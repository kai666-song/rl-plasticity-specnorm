from typing import Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from shared.modules import (
    gen_encoder,
    sp_module,
    calculate_l2_norm,
    Injector,
    mix_reset_module,
)


class PPOModel(torch.nn.Module):
    """
    PPO 模型
    
    包含共享编码器、策略头和价值头的 Actor-Critic 网络。
    支持多种归一化方法（LayerNorm, RMSNorm, Spectral Norm）和激活函数。
    """
    
    def __init__(
        self,
        obs_size: int,
        act_size: int,
        depth: int,
        model_params: dict[str, Any]
    ) -> None:
        super().__init__()
        self.split_encoder = False
        self.enc_type = model_params["enc_type"]
        self.obs_size = obs_size
        self.act_size = act_size
        self.h_size = model_params["h_size"]
        self.adapt_method, self.adapt_params = model_params["adapt_info"]
        self.activation = model_params["activation"]
        self.l2_norm = model_params["l2_norm"]
        self.lr = model_params["lr"]
        self.layernorm = model_params["layernorm"]
        self.rmsnorm = model_params.get("rmsnorm", False)
        self.specnorm = model_params.get("specnorm", False)
        self.l2_init = model_params["l2_init"]
        self.w2_init = model_params["w2_init"]
        self.redo_weight = model_params["redo_weight"]
        self.redo_freq = model_params["redo_freq"]
        self.depth = depth
        if self.split_encoder:
            self.policy_encoder = self.gen_encoder()
            self.value_encoder = self.gen_encoder()
        else:
            self.encoder = self.gen_encoder()
        self.policy = self.gen_policy(self.h_size, act_size)
        self.value = self.gen_value(self.h_size)
        self.injected = False
        self.optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.l2_norm
        )

    def gen_policy(self, h_size, act_size):
        return nn.Linear(h_size, act_size)

    def gen_value(self, h_size):
        return nn.Linear(h_size, 1)

    def gen_encoder(self):
        return gen_encoder(
            self.obs_size,
            self.h_size,
            self.depth,
            self.enc_type,
            self.activation,
            self.layernorm,
            self.rmsnorm,
            self.specnorm,
        )

    def adapt(self):
        if self.adapt_method == "sp":
            self._shrink_perturb()
        elif self.adapt_method == "inject":
            self._inject()
        elif self.adapt_method == "mix":
            self._mix()
        else:
            pass
        self.optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.l2_norm
        )

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False,
        compute_rank: bool = False
    ) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入观测，支持 4D (B, C, H, W) 或 2D (B, obs_size)
            return_stats: 是否返回诊断信息（dead_units, eff_rank）
            compute_rank: 是否计算 eff_rank（SVD 计算开销大，仅在需要时启用）
                          当 return_stats=True 但 compute_rank=False 时，
                          eff_rank 返回 -1.0 占位符
        
        Returns:
            如果 return_stats=False: (logits, value)
            如果 return_stats=True: (logits, value, dead_units, eff_rank)
        
        Note:
            - 训练时建议 return_stats=True, compute_rank=False（仅获取 dead_units）
            - 评估/日志时可设 return_stats=True, compute_rank=True（完整诊断）
            - SVD 计算在 batch_size < h_size 时结果无意义，真实秩分析请用 analyze_features.py
        """
        # 智能维度处理：保持 4D 输入不变，其他情况 reshape 为 2D
        if x.dim() != 4:
            x = x.view(-1, self.obs_size)
        
        if self.split_encoder:
            x_policy = self.policy_encoder(x)
            x_value = self.value_encoder(x)
            logits = self.policy(x_policy)
            value = self.value(x_value)
        else:
            x = self.encoder(x, return_stats=return_stats, compute_rank=compute_rank)
            if return_stats:
                x, dead_units, eff_rank = x
            logits = self.policy(x)
            value = self.value(x)
        if return_stats:
            return logits, value.view(-1), dead_units, eff_rank
        return logits, value.view(-1)

    def _inject(self):
        self.policy = Injector(self.policy, self.h_size, self.act_size)
        self.value = Injector(self.value, self.h_size, 1)
        self.policy.to(next(self.parameters()).device)
        self.value.to(next(self.parameters()).device)

    def _mix(self):
        factor = self.adapt_params
        new_encoder = self.gen_encoder()
        new_value = self.gen_value(self.h_size)
        new_policy = self.gen_policy(self.h_size, self.act_size)
        mix_reset_module(self.encoder, new_encoder, factor)
        mix_reset_module(self.value, new_value, factor)
        mix_reset_module(self.policy, new_policy, factor)

    def _shrink_perturb(self):
        shrink_targets, shrink_p, perturb_p = self.adapt_params
        shrink_encoder, shrink_value, shrink_policy = shrink_targets
        if shrink_encoder:
            if self.split_encoder:
                new_policy_enc = self.gen_encoder()
                sp_module(self.policy_encoder, new_policy_enc, shrink_p, perturb_p)
                new_value_enc = self.gen_encoder()
                sp_module(self.value_encoder, new_value_enc, shrink_p, perturb_p)
            else:
                new_enc = self.gen_encoder()
                sp_module(self.encoder, new_enc, shrink_p, perturb_p)
        if shrink_value:
            new_value = self.gen_value(self.h_size)
            sp_module(self.value, new_value, shrink_p, perturb_p)
        if shrink_policy:
            new_policy = self.gen_policy(self.h_size, self.act_size)
            sp_module(self.policy, new_policy, shrink_p, perturb_p)

    def sample_action(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从策略中采样动作
        
        Args:
            obs: 观测张量
        
        Returns:
            (action, logits, value) 元组
        """
        logits, value = self.forward(obs)
        action = torch.distributions.Categorical(logits=logits).sample()
        return action, logits, value.view(-1)

    @property
    def weight_magnitude(self):
        return calculate_l2_norm(self.parameters())
