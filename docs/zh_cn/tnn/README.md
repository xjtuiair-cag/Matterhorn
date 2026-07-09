# `matterhorn_pytorch.tnn`

[回到 `matterhorn_pytorch`](../README.md)

[English](../../en_us/tnn/README.md)

[中文](../../zh_cn/tnn/README.md)

## `matterhorn_pytorch.tnn.functional`：时空代数的算子

该模块存储了 Temporal Neural Networks（TNNs）的基础——时空代数（Space-time Algebra）的运算符，以及竞争逻辑（RL）算子。

您可以阅读[文档](./1_functional.md)以了解更多函数的使用方法。

## `matterhorn_pytorch.tnn.volley`：脉冲簇编解码

实现时间簇（volley）的归一化、值↔时间转换、OnOff 编码等工具，是 TNN 输入/输出的基础表示。

阅读[文档](./2_volley.md)。

## `matterhorn_pytorch.tnn.response` 与 `matterhorn_pytorch.tnn.neuron`：兴奋性神经元

实现 Smith 书 Model II 的分段线性复合响应、膜电位求和，以及可反向传播的 `ExcitatoryNeuron`。

阅读[文档](./3_neuron.md)。

## `matterhorn_pytorch.tnn.inhibition`：抑制柱

实现前馈抑制（FFI）与侧向抑制（LI / tk-WTA），对应认知柱中的两种抑制机制。

阅读[文档](./4_inhibition.md)。

## `matterhorn_pytorch.tnn.column`、`cognitive`、`container`：柱结构与网络容器

- `ExcitatoryColumn`：并行兴奋性神经元阵列（兴奋柱 EC）；
- `CognitiveColumn`：FFI → EC → LI 认知柱（CC）；
- `Sequential`：多层 TNN 串联容器。

阅读[文档](./5_column_cognitive.md)。

## `matterhorn_pytorch.tnn.stdp` 与 `matterhorn_pytorch.tnn.training`：学习与训练

提供 STDP 无监督学习、Model II 均值权重设置，以及基于输出脉冲时间的监督训练器 `TNNTrainer`。

阅读[文档](./6_stdp_training.md)。

## `matterhorn_pytorch.tnn.component`：TNN 组件

该模块含有 TNN 的基础组件，如双调排序网络（`Bitonic`）与发射单元（`Firing`）。可通过 `from matterhorn_pytorch.tnn import Bitonic, Firing` 或 `matterhorn_pytorch.tnn.component` 导入。

## 快速入门

```python
import torch
from matterhorn_pytorch.tnn import CognitiveColumn, Sequential, TNNTrainer

cc = CognitiveColumn(num_inputs=8, num_neurons=4, seed=42)
spikes = torch.tensor([[1.0, 1.2, 1.5, 2.0,
                        float("inf"), float("inf"), float("inf"), float("inf")]])
out, states = cc(merged_input=spikes)

net = Sequential(cc)
trainer = TNNTrainer(net)
```

## 参考文献

[1] Smith J E, Martonosi M. Space-time computing with temporal neural networks[M]. Morgan & Claypool, 2017.
