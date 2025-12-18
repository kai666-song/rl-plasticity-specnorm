#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
层范数分析脚本 (Layer Norm Analysis Script)
==========================================

分析 Spectral Normalization 对深层网络特征范数的影响。

【理论背景】
Spectral Normalization 通过约束每层权重的谱范数（最大奇异值）为 1，
确保网络的 Lipschitz 常数 <= 1。这意味着：
    ||f(x) - f(y)|| <= ||x - y||

潜在问题：
如果网络很深，且每层都严格约束 Lipschitz <= 1，
特征范数可能会随层数指数级衰减，导致梯度消失。

本脚本检测这一现象，并提供分析结果。

Usage:
    python analyze_layer_norms.py

Output:
    results/feature_analysis/layer_norms.png
    results/feature_analysis/layer_norms_comparison.png
"""

from typing import Any
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from envs.mdps import ProcGenEnv
from algos.ppo.model import PPOModel

# 设置
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})

# 实验配置
EXPERIMENTS = {
    'Baseline (ReLU)': {
        'checkpoint': 'results/baseline/checkpoints/baseline_0.pt',
        'color': '#7F8C8D',
        'specnorm': False,
    },
    'Spectral Norm (Ours)': {
        'checkpoint': 'results/specnorm_experiment/checkpoints/specnorm_0.pt',
        'color': '#E74C3C',
        'specnorm': True,
    },
}


def create_env(seed: int = 0) -> ProcGenEnv:
    """创建 ProcGen 环境"""
    return ProcGenEnv(
        shift_type='permute',
        task='coinrun_100',
        train=False,
        seed=seed
    )


def collect_observations(num_samples: int = 256, seed: int = 0) -> torch.Tensor:
    """收集真实环境观测数据"""
    env = create_env(seed)
    observations = []
    obs = env.reset()
    
    for _ in range(num_samples):
        observations.append(obs)
        action = np.random.randint(0, env.action_space.n)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    
    return torch.stack(observations)


def load_model(checkpoint_path: str, specnorm: bool = False, device: str = 'cpu') -> PPOModel:
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    
    model_params = {
        'enc_type': 'conv64',
        'h_size': 256,
        'lr': 0.0005,
        'l2_norm': 0.0,
        'l2_init': 0.0,
        'w2_init': 0.0,
        'redo_weight': 0.0,
        'redo_freq': 10,
        'activation': 'relu',
        'layernorm': False,
        'rmsnorm': False,
        'specnorm': specnorm,
        'adapt_info': ['none', None],
    }
    
    model = PPOModel(64 * 64 * 3, 9, 3, model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def analyze_layer_norms(
    model: nn.Module,
    observations: torch.Tensor,
    device: str = 'cpu'
) -> dict[str, float]:
    """
    分析每层的特征范数
    
    检测 Spectral Normalization 是否导致深层特征范数指数衰减。
    
    Args:
        model: 训练好的模型
        observations: 输入观测数据
        device: 计算设备
    
    Returns:
        每层的平均特征范数
    """
    layer_norms = {}
    hooks = []
    
    def hook_fn(name: str):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # 计算输出的 L2 范数（按样本平均）
                norm = output.view(output.size(0), -1).norm(dim=-1).mean().item()
                layer_norms[name] = norm
        return hook
    
    # 为所有卷积层和线性层注册 hooks
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(f"L{layer_idx}_{name}")))
            layer_idx += 1
    
    # 前向传播
    with torch.no_grad():
        model(observations.to(device))
    
    # 移除 hooks
    for h in hooks:
        h.remove()
    
    return layer_norms


def check_exponential_decay(norms: list[float], threshold: float = 0.1) -> tuple[bool, float]:
    """
    检测特征范数是否指数衰减
    
    Args:
        norms: 各层的特征范数列表
        threshold: 衰减阈值，如果最后一层/第一层 < threshold 则认为存在问题
    
    Returns:
        (is_decaying, decay_ratio)
    """
    if len(norms) < 2:
        return False, 1.0
    
    decay_ratio = norms[-1] / norms[0] if norms[0] > 0 else 0
    is_decaying = decay_ratio < threshold
    
    return is_decaying, decay_ratio


def plot_layer_norms(results: dict[str, dict], output_path: Path) -> None:
    """绘制层范数对比图"""
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    
    for name, data in results.items():
        norms = data['norms']
        config = data['config']
        
        layers = list(norms.keys())
        values = list(norms.values())
        
        ax.plot(range(len(values)), values,
                label=f"{name} (decay={data['decay_ratio']:.3f})",
                color=config['color'],
                marker='o',
                linewidth=2.0,
                markersize=6)
    
    ax.set_xlabel('Layer Index', fontsize=13)
    ax.set_ylabel('Feature Norm (L2)', fontsize=13)
    ax.set_title('Feature Norm Across Layers', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    print("=" * 60)
    print("Layer Norm Analysis for Spectral Normalization")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path('results/feature_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集观测数据
    print("\n[1/3] Collecting observations...")
    observations = collect_observations(num_samples=256)
    print(f"Collected {len(observations)} observations")
    
    # 分析所有实验
    print("\n[2/3] Analyzing layer norms...")
    results = {}
    
    for name, config in EXPERIMENTS.items():
        print(f"\n--- {name} ---")
        
        try:
            model = load_model(config['checkpoint'], config['specnorm'], device)
            norms = analyze_layer_norms(model, observations, device)
            
            # 检测指数衰减
            norm_values = list(norms.values())
            is_decaying, decay_ratio = check_exponential_decay(norm_values)
            
            results[name] = {
                'norms': norms,
                'config': config,
                'is_decaying': is_decaying,
                'decay_ratio': decay_ratio,
            }
            
            print(f"  Layers: {len(norms)}")
            print(f"  First layer norm: {norm_values[0]:.4f}")
            print(f"  Last layer norm: {norm_values[-1]:.4f}")
            print(f"  Decay ratio: {decay_ratio:.4f}")
            
            if is_decaying:
                print(f"  ⚠️ WARNING: Exponential decay detected!")
                print(f"     This may indicate gradient vanishing due to SN's Lipschitz constraint.")
                print(f"     Consider adding learnable scalar multipliers or reducing network depth.")
            else:
                print(f"  ✓ No significant decay detected")
                
        except Exception as e:
            print(f"  Error: {e}")
    
    # 生成图表
    print("\n[3/3] Generating plots...")
    if results:
        plot_layer_norms(results, output_dir / 'layer_norms_comparison.png')
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, data in results.items():
        status = "⚠️ DECAY" if data['is_decaying'] else "✓ OK"
        print(f"{name}: decay_ratio={data['decay_ratio']:.4f} [{status}]")
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
