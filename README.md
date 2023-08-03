# Matterhorn

## 1 General Introduction

![logo](./assets/logo.png)

Matterhorn is a general SNN framework based on PyTorch.

## 2 Installation

### Environment

Python(>=3.7.0)

CUDA(>=11.3.0, with CUDNN)

PyTorch(>=1.10.0 and <=1.13.1)

TorchVision(>=0.11.0 and <= 0.13.1)

### Environment Installation

To install PyTorch you can find the command on [https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/).

```sh
pip install numpy matplotlib scipy tqdm rich
```

### Install Matterhorn

```sh
git clone https://github.com/AmperiaWang/Matterhorn.git
cd Matterhorn
python3 setup.py install
```

Don't forget to add `sudo` if you are not the root user.

## 3 Module Explanation

### Neurons in SNN

As we're all known, the image below describes what an ANN be like.

![ANN Structure](./assets/readme_1.png)

**Operation 1** is **synapse function**, which uses weights and bias to calculate those values from previous layer to current layer. Commonly used synapse functions are including full connection layer `nn.Linear`, convolution layer `nn.Conv2D`, etc.

We use an equation to describe synapse function:

$$Y^{l}=synapse(X^{l-1})$$

Where $l$ here means the number of current layer.

**Operation 2** is **activation function**, which filters information from synapses and send the filtered information to next layer. Commonly used activation functions are including `nn.ReLU`, `nn.Sigmoid`, etc.

We use an equation to describe activation function:

$$X^{l}=activation(Y^{l})$$

This is what an ANN be like. Each of layers in ANN has 2 functions. We can build our ANN model in PyTorch by the code below:

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(28 * 28, 10),
    nn.ReLU()
)
```

This is an 1-layer MLP. It can take an image with the size of 28x28 as input and classify it into 10 classes. In this example, two equations of ANN can be represented as below:

$$Y^{l}=W^{l}X^{l-1}+\vec{b}$$

$$X^{l}=ReLU(Y^{l})$$

In SNN, the synapse equation is same as which in ANN. However, the functions in soma is no longer like what in ANN. In the soma of SNN, there exists a loop in time. The image below describes what a SNN be like.

![SNN Structure](./assets/readme_2.png)

**Operation 1**, the **synapse function**, calculates spikes from previous layer $O^{l-1}(t)$ thus generates the input potential $X^{l}(t)$.

We use an equation to describe synapse function:

$$X^{l}(t)=synapse(O^{l-1}(t))$$

By **operation 2**, the input potential, with the history potential, is calculated based on 1-order differential equation, thus generate the soma potential $U^{l}(t)$. We name it **response function**.

We use an equation to describe response function:

$$U^{l}(t)=response(H^{l}(t),X^{l}(t))$$

Each spiking neuron model has its unique response differential equation.

For example, in LIF neuron:

$$\tau \frac{du}{dt}=-(u-u_{rest})+RI$$

Discretizing it into a difference equation, we can get:

$$U^{l}(t)=(1-\frac{1}{\tau})H^{l}(t)+\frac{1}{\tau}u_{rest}+X^{l}(t)$$

**Operation 3** uses Heaviside step function and threshold potential $u_{th}$ to decide whether to generate spikes $O^{l}(t)$. We name it **spiking function**.

We use an equation to describe spiking function:

$$O^{l}(t)=spiking(U^{l}(t))$$

Generally, spiking function is like this.

$$O^{l}(t)=Heaviside(U^{l}(t)-u_{th})$$

Where Heaviside step function returns 1 when input is greater than or equal to 0, returns 0 otherwise.

The aim of **operation 4** is to set refractory time on neurons by output spikes $O^{l}(t)$. We name it **reset function**.

We use an equation to describe reset function:

$$H^{l}(t)=reset(U^{l}(t-1),O^{l}(t-1))$$

Under most occasions we use equation below to reset potential:

$$H^{l}(t)=U^{l}(t-1)[1-O^{l}(t-1)]+u_{rest}O^{l}(t-1)$$

In brief, we use 4 equations to describe SNN neurons. This is what a SNN be like. The shape of a SNN neuron is like a trumpet. Its synapses transforms those spikes from last neuron and pass the input response to soma, in which there is a time loop awaits.

By unfolding SNN neuron in temporal dimension, we can get the spatial-temporal topology network of SNN.

![Spatial-temporal Topology Network of SNN](./assets/readme_3.png)

Like building ANN in PyTorch, we can build our SNN model in Matterhorn by the code below:

```python
import torch
import matterhorn.snn as snn

snn_model = snn.TemporalContainer(
    snn.SpatialContainer(
        snn.Linear(28 * 28, 10),
        snn.LIF()
    )
)
```

In the code, `SpatialContainer` is one Matterhorn's container to represent sequential SNN layers in spatial dimension, and `TemporalContainer` is another Matterhorn's container to repeat calculating potential and spikes in temporal dimension. By using `SpatialContainer` and `TemporalContainer`, an SNN spatial-temporal topology network is built thus used for training and evaluating.

The built network takes an $n+1$ dimensional `torch.Tensor` as input spike train. It will take the first dimension as time steps, thus claculate through each time step. after that, it will generate a `torch.Tensor` as output spike train, just like what an ANN takes and generates in PyTorch. The only difference, which is also a key point, is that we should encode our information into spike train and decode the output spike train.

### Encoding and Decoding

A spike train is a set of Dirac impulse functions on the axis of time.

$$O(t)=\sum_{t_{i}}δ(t-t_{i})$$

In other words, there will only be 0s and 1s in discrete spike train. Therefore, we can use an $n+1$ dimensional tensor to represent our spike train. For example, if neurons are flattened into a 1-dimensional vector, we can use another dimension to represent time, thus let it be a 2-dimensional matrix to represent the spike train through space and time.

$$
\begin{matrix}
 & →s \\
↓t &
\begin{bmatrix}
0 & 1 & 1 & 0 & 1 & 0 & 0 & 1 \\
1 & 0 & 0 & 1 & 0 & 0 & 1 & 1 \\
1 & 1 & 1 & 1 & 1 & 1 & 0 & 1 \\
1 & 0 & 1 & 1 & 0 & 1 & 0 & 1 \\
\end{bmatrix}
\end{matrix}
$$

The matrix above shows what a spike train looks like. It has 4 rows, representing 4 time steps. Besides, it has 8 columns, representing 8 output neurons.

To transform our traditional binary information (images, sounds, etc.) into spike train, an encoder is needed. The most commonly used encoder for non-event data is Poisson encoder, which is a kind of rate coding encoder. It sees intensity of a pixel as probability to fire a spike.

You can use Poisson encoder in Matterhorn by the code below:

```python
import torch
import matterhorn.snn as snn

encoder = snn.PoissonEncoder(
    time_steps = 32
)
```

Then, you can use it by the code below:

```python
spike_train = encoder(image)
```

An image with the shape of `[H, W, C]` would be encoded into a spike train with the shape of `[T, H, W, C]`. For example, a MNIST image which shape is `[28, 28]` would be encoded (`T=32`) into a spike train with the shape of `[32, 28, 28]`.

After encoding and processing, the network would generate an output spike train. To get the information, we need to decode. Commonly used decoding method is to count average spikes each output neuron has generated.

$$\hat{y}_{i}=\frac{1}{T}\sum_{t=1}^{T}{O_{i}^{K}(t)}$$

You can use Poisson encoder in Matterhorn by the code below:

```python
import torch
import matterhorn.snn as snn

decoder = snn.AvgDecoder()
```

It will take first dimension as temporal dimension, and generate statistic result as output. The output can be transported into ANN for further process.

By now, you have experienced how a SNN look like and how to build it by Matterhorn. For further experience, you can refer to [examples/2_layer_mlp.py](./examples/2_layer_mlp.py).

```python
cd Matterhorn
python3 examples/2_layer_mlp.py
```