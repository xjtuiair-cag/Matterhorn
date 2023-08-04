import torch
from torch import Tensor
import torch.nn as nn
from rich import print


from matterhorn.snn import container


@torch.jit.script
def stdp_py(weight_mat: torch.Tensor, input_shape: int, output_shape: int, time_steps: int, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> None:
    """
    STDP的python版本实现，不到万不得已不会调用（性能是灾难级别的）
    @params:
        weight_mat: torch.Tensor 权重矩阵，形状为[output_shape, input_shape]
        input_shape: int 输入长度
        output_shape: int 输出长度
        time_steps: int 时间步长
        input_spike_train: torch.Tensor 输入脉冲序列，形状为[input_shape, time_steps]
        output_spike_train: torch.Tensor 输出脉冲序列，形状为[output_shape, time_steps]
        a_pos: float STDP参数A+
        tau_pos: float STDP参数tau+
        a_neg: float STDP参数A-
        tau_neg: float STDP参数tau-
    """
    for i in range(output_shape):
        for j in range(input_shape):
            weight = 0.0
            for ti in range(time_steps):
                if not output_spike_train[ti, i]:
                    continue
                for tj in range(time_steps):
                    if not input_spike_train[tj, j]:
                        continue
                    dt = ti - tj
                    if dt > 0:
                        weight += a_pos * torch.exp(-dt / tau_pos)
                    else:
                        weight += -a_neg * torch.exp(dt / tau_neg)
            weight_mat[i, j] += weight
    return


if torch.cuda.is_available():
    try:
        from matterhorn_cuda_extensions import stdp as stdp_cuda
    except:
        stdp_cuda = None
try:
    from matterhorn_cpp_extensions import stdp as stdp_cpp
except:
    stdp_cpp = None


def stdp(weight_mat: torch.Tensor, input_shape: int, output_shape: int, time_steps: int, input_spike_train: torch.Tensor, output_spike_train: torch.Tensor, a_pos: float, tau_pos: float, a_neg: float, tau_neg: float) -> None:
    """
    STDP总函数，视情况调用函数
    @params:
        weight_mat: torch.Tensor 权重矩阵，形状为[output_shape, input_shape]
        input_shape: int 输入长度
        output_shape: int 输出长度
        time_steps: int 时间步长
        input_spike_train: torch.Tensor 输入脉冲序列，形状为[input_shape, time_steps]
        output_spike_train: torch.Tensor 输出脉冲序列，形状为[output_shape, time_steps]
        a_pos: float STDP参数A+
        tau_pos: float STDP参数tau+
        a_neg: float STDP参数A-
        tau_neg: float STDP参数tau-
    """
    w_type = weight_mat.device.type
    w_idx = weight_mat.device.index
    i_type = input_spike_train.device.type
    i_idx = input_spike_train.device.index
    o_type = output_spike_train.device.type
    o_idx = output_spike_train.device.index
    assert (w_type == i_type and i_type == o_type) and (w_idx == i_idx and i_idx == o_idx), "The type of weight matrix, input spike train and output spike train should be the same."
    device_type = w_type
    device_idx = w_idx
    if device_type == "cuda" and stdp_cuda is not None:
        stdp_cuda(weight_mat, input_shape, output_shape, time_steps, input_spike_train, output_spike_train, a_pos, tau_pos, a_neg, tau_neg)
        return
    weight_mat_cpu = weight_mat.cpu()
    input_spike_train_cpu = input_spike_train.cpu()
    output_spike_train_cpu = output_spike_train.cpu()
    if stdp_cpp is not None:
        stdp_cpp(weight_mat, input_shape, output_shape, time_steps, input_spike_train, output_spike_train, a_pos, tau_pos, a_neg, tau_neg)
        return
    stdp_py(weight_mat, input_shape, output_shape, time_steps, input_spike_train, output_spike_train, a_pos, tau_pos, a_neg, tau_neg)
    return


class STDPLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, soma: nn.Module, a_pos: float = 0.4, tau_pos: float = 1.0, a_neg: float = 0.4, tau_neg: float = 1.0, device = None, dtype = None) -> None:
        """
        使用STDP学习机制时的全连接层
        @params:
            in_features: int 输入长度，用法同nn.Linear
            out_features: int 输出长度，用法同nn.Linear
            soma: nn.Module 使用的脉冲神经元胞体，在matterhorn.snn.soma中选择
            a_pos: float STDP参数A+
            tau_pos: float STDP参数tau+
            a_neg: float STDP参数A-
            tau_neg: float STDP参数tau-
        """
        super().__init__(
            in_features = in_features, 
            out_features = out_features,
            bias = False,
            device = device,
            dtype = dtype
        )
        self.soma = soma
        self.a_pos = a_pos
        self.tau_pos = tau_pos
        self.a_neg = a_neg
        self.tau_neg = tau_neg
        self.n_reset()


    def n_reset(self):
        """
        重置整个神经元
        """
        self.input_spike_seq = []
        self.output_spike_seq = []
        if hasattr(self.soma, "n_reset"):
            self.soma.n_reset()


    def l_step(self):
        """
        对整个神经元应用STDP使其更新
        """
        input_spike_train = torch.stack(self.input_spike_seq)
        output_spike_train = torch.stack(self.output_spike_seq)
        if len(input_spike_train.shape) == 3:
            batch_size = input_spike_train.shape[1]
            for b in range(batch_size):
                delta_weight = torch.zeros_like(self.weight)
                stdp(delta_weight, self.in_features, self.out_features, len(self.input_spike_seq), input_spike_train[:, b], output_spike_train[:, b], self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
                self.weight += delta_weight
        else:
            delta_weight = torch.zeros_like(self.weight)
            stdp(delta_weight, self.in_features, self.out_features, len(self.input_spike_seq), input_spike_train, output_spike_train, self.a_pos, self.tau_pos, self.a_neg, self.tau_neg)
            self.weight += delta_weight
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数
        @params:
            x: torch.Tensor 上一层脉冲$O_{j}^{l-1}(t)$
        @return:
            o: torch.Tensor 当前层脉冲$O_{i}^{l}(t)$
        """
        self.input_spike_seq.append(x.clone())
        x = super().forward(x)
        x = self.soma(x)
        self.output_spike_seq.append(x.clone())
        return x


class STDPSpatial(container.Spatial):
    def __init__(self, *args):
        """
        使用STDP学习机制时所使用的空间容器
        @params:
            *args: [nn.Module] 按空间顺序传入的各个模块
        """
        super().__init__(*args)
    

    def l_step(self):
        """
        一次部署所有结点的STDP学习
        """
        for module in self:
            if hasattr(module, "l_step"):
                module.l_step()



class STDPTemporal(container.Temporal):
    def __init__(self, model: nn.Module, reset_after_process = True) -> None:
        """
        使用STDP学习机制时所使用的时间容器
        @params:
            model: nn.Module 所用来执行的单步模型
            reset_after_process: bool 是否在执行完后自动重置，若为False则需要手动重置
        """
        super().__init__(model, reset_after_process)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数，默认接受的张量形状为[T,B,...]（需要将时间维度通过permute等函数转到最外）
        @params:
            x: torch.Tensor 输入张量
        @return:
            y: torch.Tensor 输出张量
        """
        time_steps = x.shape[0]
        result = []
        for t in range(time_steps):
            result.append(self.model(x[t]))
        y = torch.stack(result)
        if hasattr(self.model, "l_step"):
            self.model.l_step()
        if self.reset_after_process and hasattr(self.model, "n_reset"):
            self.model.n_reset()
        return y