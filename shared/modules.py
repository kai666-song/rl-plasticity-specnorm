import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math


def gen_encoder(obs_size, h_size, depth, enc_type, activation="relu", layernorm=False, rmsnorm=False, specnorm=False):
    """
    生成编码器网络 (Encoder Network Generator)
    
    这是网络架构的核心工厂函数，支持多种归一化和激活函数配置。
    
    Args:
        obs_size: 观测空间大小
        h_size: 隐藏层大小
        depth: 输入通道数（对于图像输入）
        enc_type: 编码器类型 ("conv32", "conv64", "conv11", "linear")
        activation: 激活函数 ("relu", "leaky_relu", "mish", "gelu", "tanh")
        layernorm: 是否使用 Layer Normalization
        rmsnorm: 是否使用 RMS Normalization
        specnorm: 是否使用 Spectral Normalization（谱归一化）【推荐】
                  通过约束权重矩阵的谱范数（最大奇异值）来：
                  1. 防止特征秩崩溃 (Feature Rank Collapse)
                  2. 稳定训练过程，避免梯度爆炸
                  3. 保持网络的表达能力和可塑性
    
    Returns:
        nn.Module: 编码器网络
    
    Example:
        >>> encoder = gen_encoder(obs_size=64*64*3, h_size=256, depth=3, 
        ...                       enc_type="conv64", specnorm=True)
    """
    encoders = {
        "conv32": lambda: ConvEncoder(h_size, depth, 32, activation, layernorm, rmsnorm, specnorm),
        "conv64": lambda: ConvEncoder(h_size, depth, 64, activation, layernorm, rmsnorm, specnorm),
        "conv11": lambda: ConvEncoder(h_size, depth, 11, activation, layernorm, rmsnorm, specnorm),
        "linear": lambda: LinearEncoder(obs_size, h_size, activation, layernorm, rmsnorm, specnorm),
    }

    return encoders.get(enc_type, lambda: raise_not_implemented_error())()


def raise_not_implemented_error():
    raise NotImplementedError


def calculate_l2_norm(parameters):
    l2_norm_squared = 0.0
    num_params = 0
    for param in parameters:
        l2_norm_squared += torch.sum(param**2).item()
        num_params += param.numel()
    return torch.sqrt(torch.tensor(l2_norm_squared) / num_params).item()


def save_param_state(model):
    initial_state = {
        name: param.clone().detach() for name, param in model.named_parameters()
    }
    return initial_state


def compute_l2_norm_difference(model, initial_state):
    l2_diff = 0.0
    num_params = 0

    for name, param in model.named_parameters():
        if name in initial_state:
            param_diff = param - initial_state[name]
            l2_diff += torch.sum(param_diff**2).item()
            num_params += param.numel()
        else:
            raise ValueError(
                f"Saved initial state does not contain parameters '{name}'."
            )

    return torch.sqrt(torch.tensor(l2_diff) / num_params).item()


def clone_model_state(model: nn.Module):
    return {key: value.clone().detach() for key, value in model.state_dict().items()}


def restore_model_state(model: nn.Module, state_dict: dict):
    model.load_state_dict(state_dict)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CReLu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if len(x.shape) == 2:
            # FF layer
            return torch.cat([F.relu(x), F.relu(-x)], dim=-1)
        else:
            # Conv layer
            return torch.cat([F.relu(x), F.relu(-x)], dim=1)


class Mish(nn.Module):
    """
    Mish激活函数: x * tanh(softplus(x))
    - 平滑、非单调、无上界、有下界
    - 负值区域有小的负输出，能更好地保持梯度流动
    - 理论上能缓解plasticity loss问题
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Injector(nn.Module):
    def __init__(self, original, in_size=256, out_size=10):
        super(Injector, self).__init__()
        if type(original) == nn.Linear:
            self.original = original
        elif type(original) == Injector:
            self.original = nn.Linear(in_size, out_size)
            aw = original.original.weight
            bw = original.new_a.weight
            cw = original.new_b.weight
            self.original.weight = nn.Parameter(aw + bw - cw)
            ab = original.original.bias
            bb = original.new_a.bias
            cb = original.new_b.bias
            self.original.bias = nn.Parameter(ab + bb - cb)
        else:
            raise NotImplementedError
        self.new_a = nn.Linear(in_size, out_size)
        self.new_b = copy.deepcopy(self.new_a)

    def forward(self, x):
        return self.original(x) + self.new_a(x) - self.new_b(x).detach()


class LnReLU(nn.Module):
    # Simple layernorm followed by RELU
    def __init__(self, h_size):
        super().__init__()
        self.norm = nn.LayerNorm(h_size)

    def forward(self, x):
        return F.relu(self.norm(x))


class GnReLU(nn.Module):
    # Simple groupnorm followed by RELU
    def __init__(self, h_size, groups=1):
        super().__init__()
        self.norm = nn.GroupNorm(groups, h_size)

    def forward(self, x):
        return F.relu(self.norm(x))


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    比LayerNorm更轻量，不需要计算均值，只做缩放
    公式: x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return (x / rms) * self.weight


def compute_effective_rank(features):
    """
    计算特征矩阵的有效秩 (Effective Rank)
    
    有效秩是衡量特征多样性的指标，基于奇异值的归一化熵：
        eff_rank = exp(-sum(p_i * log(p_i)))
    其中 p_i = sigma_i / sum(sigma_j) 是归一化的奇异值分布
    
    Args:
        features: 特征矩阵，形状为 (batch_size, feature_dim)
    
    Returns:
        float: 有效秩，范围从 1（所有信息集中在一个方向）到 min(batch, dim)（均匀分布）
    
    Reference:
        Roy & Bhattacharya, "Effective Rank: A Measure of Effective Dimensionality", 2007
    """
    # 中心化特征
    features = features - features.mean(dim=0, keepdim=True)
    
    # 计算奇异值
    try:
        _, S, _ = torch.svd(features, compute_uv=False)
    except:
        return torch.tensor(0.0)
    
    # 归一化奇异值得到概率分布
    S = S + 1e-10  # 避免除零
    p = S / S.sum()
    
    # 计算熵
    entropy = -(p * torch.log(p)).sum()
    
    # 有效秩 = exp(熵)
    eff_rank = torch.exp(entropy)
    
    return eff_rank


def compute_singular_values(features):
    """
    计算特征矩阵的奇异值分布
    
    Args:
        features: 特征矩阵，形状为 (batch_size, feature_dim)
    
    Returns:
        torch.Tensor: 奇异值，按降序排列
    """
    features = features - features.mean(dim=0, keepdim=True)
    try:
        _, S, _ = torch.svd(features, compute_uv=False)
        return S
    except:
        return torch.zeros(min(features.shape))


class ConvEncoder(nn.Module):
    """
    卷积编码器，支持多种归一化方法
    
    输入/输出维度说明：
    - 输入: (batch_size, obs_size) 其中 obs_size = C * H * W
    - 内部使用 nn.Unflatten 恢复为 (batch_size, C, H, W) 进行卷积
    - 输出: (batch_size, h_size) 特征向量
    
    设计说明：
    - 这种"先压扁再还原"的设计是为了统一 Linear/Conv Encoder 的接口
    - 虽然不是最优雅的设计，但保证了模块的可替换性
    - 如需完全消除这个模式，需要重构整个 Encoder 接口和调用方
    
    Args:
        h_size: 输出特征维度
        depth: 输入通道数 (C)
        conv_size: 输入图像尺寸 (H=W)
        activation: 激活函数类型
        layernorm: 是否使用 LayerNorm (在卷积层用 GroupNorm(1) 替代)
        rmsnorm: 是否使用 RMSNorm
        specnorm: 是否使用 Spectral Normalization
                  SN 通过约束权重的谱范数（最大奇异值）来稳定训练
                  可以防止特征秩崩溃，保持网络的表达能力
    """
    def __init__(self, h_size, depth, conv_size, activation="relu", layernorm=False, rmsnorm=False, specnorm=False):
        super().__init__()
        self.depth = depth
        self.h_size = h_size
        self.activation = activation
        self.layernorm = layernorm
        self.rmsnorm = rmsnorm
        self.specnorm = specnorm
        self.encoder = getattr(self, f"conv{conv_size}")()

    def _maybe_spectral_norm(self, layer):
        """
        条件性应用谱归一化 (Conditional Spectral Normalization)
        
        Spectral Normalization 的数学原理 (Miyato et al., ICLR 2018):
        
        对于权重矩阵 W，谱归一化将其归一化为：
            W_SN = W / σ(W)
        
        其中 σ(W) 是 W 的最大奇异值（谱范数）。
        
        这确保了每一层的 Lipschitz 常数被约束为 1：
            ||f(x) - f(y)|| ≤ ||x - y||
        
        在深度 RL 中的作用：
        1. 防止特征秩崩溃：约束权重增长，保持特征多样性
        2. 稳定训练：避免梯度爆炸，使训练曲线更平滑
        3. 保持可塑性：网络能持续学习新任务，而非僵化
        
        与 ReDo 的对比：
        - ReDo 是"外科手术"：周期性砍掉休眠神经元，造成训练震荡
        - SN 是"基因疗法"：从根本上约束网络，让其健康生长
        
        Reference:
            Miyato et al., "Spectral Normalization for GANs", ICLR 2018
        """
        if self.specnorm:
            return nn.utils.spectral_norm(layer)
        return layer

    def map_conv_activation(self, out_channels):
        """
        为卷积层生成激活函数（可选归一化）
        
        实现细节说明：
        - 当 layernorm=True 时，在卷积层使用 GroupNorm(groups=1, channels) 替代 LayerNorm
        - 这是因为标准 LayerNorm 期望输入形状为 (B, D)，无法直接处理 4D 卷积输出 (B, C, H, W)
        - GroupNorm(1, C) 在数学上等价于对每个样本的所有通道做归一化，
          类似于 LayerNorm 但能正确处理空间维度
        - 在全连接层（Flatten 之后）则使用标准的 nn.LayerNorm
        
        Reference:
            Wu & He, "Group Normalization", ECCV 2018
            - GroupNorm(1, C) 也被称为 "Layer Normalization for CNNs"
        """
        use_groupnorm = self.layernorm  # LayerNorm 在卷积层用 GroupNorm(1) 替代
        return map_activation(self.activation, out_channels, False, use_groupnorm, False)

    def conv11(self):
        if self.activation != "crelu":
            conv_depths = [16, 16, 32, 32, 64, self.h_size]
        else:
            conv_depths = [8, 16, 16, 32, 32, self.h_size // 2]
        return nn.Sequential(
            nn.Unflatten(1, (self.depth, 11, 11)),
            self._maybe_spectral_norm(nn.Conv2d(self.depth, conv_depths[0], 3, 1, 0)),
            self.map_conv_activation(conv_depths[0]),
            self._maybe_spectral_norm(nn.Conv2d(conv_depths[1], conv_depths[2], 3, 1, 0)),
            self.map_conv_activation(conv_depths[2]),
            self._maybe_spectral_norm(nn.Conv2d(conv_depths[3], conv_depths[4], 3, 1, 0)),
            self.map_conv_activation(conv_depths[4]),
            nn.Flatten(1, -1),
            self._maybe_spectral_norm(nn.Linear(1600, conv_depths[5])),
            map_activation(self.activation, self.h_size, self.layernorm, False, self.rmsnorm),
        )

    def conv32(self):
        if self.activation != "crelu":
            conv_depths = [16, 16, 32, 32, 64, self.h_size]
        else:
            conv_depths = [8, 16, 16, 32, 32, self.h_size // 2]
        return nn.Sequential(
            nn.Unflatten(1, (self.depth, 32, 32)),
            self._maybe_spectral_norm(nn.Conv2d(self.depth, conv_depths[0], 4, 2, 1)),
            self.map_conv_activation(conv_depths[0]),
            self._maybe_spectral_norm(nn.Conv2d(conv_depths[1], conv_depths[2], 4, 2, 1)),
            self.map_conv_activation(conv_depths[2]),
            self._maybe_spectral_norm(nn.Conv2d(conv_depths[3], conv_depths[4], 4, 2, 1)),
            self.map_conv_activation(conv_depths[4]),
            nn.Flatten(1, -1),
            self._maybe_spectral_norm(nn.Linear(1024, conv_depths[5])),
            map_activation(self.activation, self.h_size, self.layernorm, False, self.rmsnorm),
        )

    def conv64(self):
        if self.activation != "crelu":
            conv_depths = [32, 32, 64, 64, 128, 128, 128, self.h_size]
        else:
            conv_depths = [16, 32, 32, 64, 64, 128, 64, self.h_size // 2]
        return nn.Sequential(
            nn.Unflatten(1, (self.depth, 64, 64)),
            self._maybe_spectral_norm(nn.Conv2d(self.depth, conv_depths[0], 4, 2, 1)),
            self.map_conv_activation(conv_depths[0]),
            self._maybe_spectral_norm(nn.Conv2d(conv_depths[1], conv_depths[2], 4, 2, 1)),
            self.map_conv_activation(conv_depths[2]),
            self._maybe_spectral_norm(nn.Conv2d(conv_depths[3], conv_depths[4], 4, 2, 1)),
            self.map_conv_activation(conv_depths[4]),
            self._maybe_spectral_norm(nn.Conv2d(conv_depths[5], conv_depths[6], 4, 2, 1)),
            self.map_conv_activation(conv_depths[6]),
            nn.Flatten(1, -1),
            self._maybe_spectral_norm(nn.Linear(2048, conv_depths[7])),
            map_activation(self.activation, self.h_size, self.layernorm, False, self.rmsnorm),
        )

    def calc_dead_units(self, x):
        if self.activation in ["ln-relu", "relu", "crelu"]:
            return torch.mean((x == 0).float())
        elif self.activation == "leaky_relu":
            # Leaky ReLU: 负区间有0.01斜率，数学上不存在死神经元
            return torch.tensor(0.0)
        elif self.activation == "tanh":
            return torch.mean((torch.abs(x) > 0.99).float())
        elif self.activation == "sigmoid":
            return torch.mean(((x < 0.01) | (x > 0.99)).float())
        elif self.activation == "gelu":
            # GELU: 检测输出是否在死区（接近0或负饱和区）
            return torch.mean((torch.abs(x) < 0.01).float())
        elif self.activation == "mish":
            # Mish: 最小值约为-0.31，检测神经元是否卡在负饱和区
            # 或者方差极低（神经元不活跃）
            # 使用更宽松的阈值，只有当输出非常接近Mish的最小值时才算"死亡"
            return torch.mean((x < -0.28).float())
        else:
            raise NotImplementedError

    def forward(self, x, check=False):
        x = self.encoder(x)

        if check:
            dead_units = self.calc_dead_units(x)
            eff_rank = compute_effective_rank(x)
            return x, dead_units, eff_rank
        else:
            return x


def get_activation(activation_name):
    activations = {
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=0.01),
        "tanh": nn.Tanh(),
        "gelu": nn.GELU(),
        "sigmoid": nn.Sigmoid(),
        "crelu": CReLu(),
        "mish": Mish(),
    }

    if activation_name not in activations:
        raise NotImplementedError
    return activations[activation_name]


def map_activation(activation, h_size, layernorm=False, groupnorm=False, rmsnorm=False):
    layers = []

    if layernorm:
        layers.append(nn.LayerNorm(h_size))
    elif rmsnorm:
        layers.append(RMSNorm(h_size))
    elif groupnorm:
        if activation == "crelu":
            h_size = h_size * 2
        layers.append(nn.GroupNorm(1, h_size))

    layers.append(get_activation(activation))
    return nn.Sequential(*layers)


class LinearEncoder(nn.Module):
    def __init__(self, obs_size, h_size, activation="relu", layernorm=False, rmsnorm=False, specnorm=False):
        super().__init__()

        if activation == "crelu":
            h_size = h_size // 2
            h_size_mid = h_size * 2
        else:
            h_size_mid = h_size

        # 根据 specnorm 参数决定是否应用谱归一化
        linear_a = nn.Linear(obs_size, h_size)
        linear_b = nn.Linear(h_size_mid, h_size)
        
        if specnorm:
            linear_a = nn.utils.spectral_norm(linear_a)
            linear_b = nn.utils.spectral_norm(linear_b)

        self.enc_a = nn.Sequential(
            linear_a,
            map_activation(activation, h_size, layernorm, False, rmsnorm),
        )

        self.enc_b = nn.Sequential(
            linear_b,
            map_activation(activation, h_size, layernorm, False, rmsnorm),
        )

        self.activation = activation

    def calc_dead_units(self, x):
        if self.activation in ["ln-relu", "gn-relu", "relu", "crelu"]:
            return torch.mean((x == 0).float())
        elif self.activation == "leaky_relu":
            # Leaky ReLU: 负区间有0.01斜率，数学上不存在死神经元
            return torch.tensor(0.0)
        elif self.activation == "tanh":
            return torch.mean((torch.abs(x) > 0.99).float())
        elif self.activation == "sigmoid":
            return torch.mean(((x < 0.01) | (x > 0.99)).float())
        elif self.activation == "gelu":
            return torch.mean((torch.abs(x) < 0.01).float())
        elif self.activation == "mish":
            # Mish: 最小值约为-0.31，检测神经元是否卡在负饱和区
            return torch.mean((x < -0.28).float())
        else:
            raise NotImplementedError

    def forward(self, x, check=False):
        x = self.enc_a(x)
        x = self.enc_b(x)

        if check:
            dead_units = self.calc_dead_units(x)
            eff_rank = compute_effective_rank(x)
            return x, dead_units, eff_rank
        else:
            return x


def sp_module(current_module, init_module, shrink_factor, epsilon):
    use_device = next(current_module.parameters()).device
    init_params = list(init_module.to(use_device).parameters())
    for idx, current_param in enumerate(current_module.parameters()):
        current_param.data *= shrink_factor
        current_param.data += epsilon * init_params[idx].data


def mix_reset_module(current_module, init_module, mix_factor):
    # Randomly replaced units from the current module with units from the init module
    init_params = list(init_module.parameters())
    for idx, current_param in enumerate(current_module.parameters()):
        init_param = init_params[idx]
        mask = torch.rand_like(current_param.data).to(current_param.device) < mix_factor
        current_param.data = torch.where(
            mask, init_param.data.to(current_param.device), current_param.data
        )


def reinitialize_weights(module, reset_mask, next_module):
    """
    重新初始化休眠神经元的权重 (Reinitialize Dormant Neuron Weights)
    
    根据重置掩码重新初始化模块的权重和偏置。这是 ReDo 算法的核心操作之一。
    
    Args:
        module (torch.nn.Module): 需要重新初始化的模块（Linear 或 Conv2d）
        reset_mask (torch.Tensor): 布尔张量，标记哪些神经元需要重置
        next_module: 下一层模块，用于将重置神经元的输出权重置零
    
    Note:
        - 使用 Kaiming 初始化重置神经元的输入权重
        - 将重置神经元的输出权重置零，确保重置不会立即影响网络输出
    """
    # 使用 Kaiming 均匀分布重新初始化权重
    new_weights = torch.empty_like(module.weight.data)
    torch.nn.init.kaiming_uniform_(new_weights, a=math.sqrt(5))
    module.weight.data[reset_mask] = new_weights[reset_mask].to(module.weight.device)

    # 将重置神经元的输出权重置零，防止重置立即影响网络输出
    if type(module) == type(next_module):
        next_module.weight.data[:, reset_mask] = 0.0


def redo_reset(model, input, temp):
    """
    ReDo 算法：选择性重置休眠神经元 (ReDo: Reactivate Dormant Neurons)
    
    该算法通过周期性检测并重置"休眠"神经元来维持网络的可塑性。
    休眠神经元是指激活值持续较低、对网络输出贡献很小的神经元。
    
    算法流程:
        1. 前向传播计算每个神经元的 s-score（相对激活分数）
        2. 识别 s-score 低于阈值 temp 的休眠神经元
        3. 重新初始化休眠神经元的权重
    
    Args:
        model (torch.nn.Module): 需要处理的神经网络模型
        input (torch.Tensor): 用于计算激活值的输入数据
        temp (float): 阈值温度，s-score 低于此值的神经元将被重置
                      典型值: 0.025
    
    Reference:
        Sokar et al., "Dormant Neuron Phenomenon in Deep Reinforcement Learning", ICML 2023
    """
    with torch.no_grad():
        s_scores_dict = calculate_s_scores_multilayer(model, input)

        modules = [
            m
            for m in model.named_modules()
            if isinstance(m[1], (torch.nn.Linear, torch.nn.Conv2d))
        ]

        # check if there are any conv layers in the network
        has_conv = any(isinstance(m[1], torch.nn.Conv2d) for m in modules)

        for i, (name, module) in enumerate(modules):
            # Skip the first entry, which is the model itself in named_modules()
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue  # Skip non-relevant modules
            if not has_conv:
                base_name_parts = name.split(".")[:-1] + ["1.0"]
                base_name = ".".join(base_name_parts)
            elif ("policy" not in name) and ("value" not in name):
                base_name_parts = name.split(".")
                base_name_parts[-1] = str(int(base_name_parts[-1]) + 1)
                base_name_parts.append("0")
                base_name = ".".join(base_name_parts)
            else:
                continue
            if base_name in s_scores_dict:
                s_scores = s_scores_dict[base_name]
                reset_mask = s_scores <= temp

                # Check if there is a next module in the list and get it
                next_module = modules[i + 1][1] if i + 1 < len(modules) else None
                # Assuming reinitialize_weights is modified to handle the next_module
                # You would need to adjust reinitialize_weights to apply the necessary changes
                # to both the current and next modules based on reset_mask.
                reinitialize_weights(module, reset_mask, next_module)


def calculate_s_scores_multilayer(model, inputs):
    """
    计算多层神经网络中每个神经元的 s-score (Calculate S-Scores for Multi-Layer Network)
    
    S-score 是衡量神经元活跃程度的指标，定义为：
        s_i = mean(activation_i) / mean(all_activations)
    
    s-score 较低的神经元被认为是"休眠"的，对网络输出贡献很小。
    
    Args:
        model (torch.nn.Module): 多层神经网络模型
        inputs (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)
    
    Returns:
        dict: 字典，键为层名称，值为对应的 s-score 张量，形状为 (num_neurons,)
    
    Note:
        - 仅计算 ReLU 激活层后的 s-score
        - 使用 forward hook 捕获中间激活值
    """
    # Create a dictionary to store the s scores for each layer
    s_scores_dict = {}

    # Register a forward hook to capture the activations of each layer
    activations = {}
    hooks = []

    def hook(module, input, output):
        activations[module] = output.detach()

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            handle = module.register_forward_hook(hook)
            hooks.append(handle)

    # Forward pass through the model
    model(inputs)

    # Calculate the s scores for each layer
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ReLU):
            layer_activations = activations[module]
            s_scores = layer_activations / torch.mean(
                layer_activations, axis=1, keepdim=True
            )
            s_scores = torch.mean(s_scores, axis=0)
            if len(s_scores.shape) > 1:
                s_scores = torch.mean(
                    s_scores, axis=tuple(range(1, len(s_scores.shape)))
                )
            s_scores_dict[name] = s_scores

    # Remove the hooks to prevent memory leaks
    for handle in hooks:
        handle.remove()

    return s_scores_dict
