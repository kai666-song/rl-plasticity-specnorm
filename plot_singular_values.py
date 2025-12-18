#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
奇异值谱对比图生成脚本 (Singular Value Spectrum Plot Generator)
===============================================================

生成 Baseline 和 Spectral Norm 的奇异值分布对比图，
直接展示"防止秩崩溃"的数学证据。

使用方法:
    python plot_singular_values.py

输出:
    docs/assets/singular_value_spectrum.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from algos.ppo.model import PPOModel
from shared.modules import compute_singular_values, compute_effective_rank

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})

# 实验配置
EXPERIMENTS = {
    'Baseline (ReLU)': {
        'checkpoint': 'results/baseline/checkpoints/baseline_0.pt',
        'color': '#7F8C8D',
        'linestyle': '-',
    },
    'ReDo Reset': {
        'checkpoint': 'results/ablation_studies/redo_experiment/results/redo_reset/checkpoints/redo-reset_0.pt',
        'color': '#27AE60',
        'linestyle': '--',
    },
    'Spectral Norm (Ours)': {
        'checkpoint': 'results/specnorm_experiment/checkpoints/specnorm_0.pt',
        'color': '#E74C3C',
        'linestyle': '-',
    },
}


def load_model_and_get_features(checkpoint_path, sample_obs):
    """加载模型并获取特征"""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    
    # 从checkpoint获取模型参数
    model_state = checkpoint['model_state_dict']
    
    # 推断模型配置
    # 检查是否有specnorm (通过检查权重名称中是否有weight_orig)
    has_specnorm = any('weight_orig' in k for k in model_state.keys())
    
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
        'specnorm': has_specnorm,
        'adapt_info': ['none', None],
    }
    
    obs_size = 64 * 64 * 3
    act_size = 9  # CoinRun has 9 actions
    depth = 3
    
    model = PPOModel(obs_size, act_size, depth, model_params)
    model.load_state_dict(model_state)
    model.eval()
    
    # 获取编码器输出特征
    with torch.no_grad():
        x = sample_obs.view(-1, obs_size)
        features = model.encoder(x)
    
    return features


def generate_sample_observations(batch_size=256):
    """生成样本观测（随机噪声，用于分析特征空间）"""
    # 使用随机噪声作为输入
    obs = torch.randn(batch_size, 64 * 64 * 3)
    return obs


def plot_singular_value_spectrum(output_path):
    """绘制奇异值谱对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    sample_obs = generate_sample_observations(batch_size=256)
    
    all_sv = {}
    all_eff_rank = {}
    
    for name, config in EXPERIMENTS.items():
        try:
            features = load_model_and_get_features(config['checkpoint'], sample_obs)
            sv = compute_singular_values(features)
            eff_rank = compute_effective_rank(features)
            all_sv[name] = sv.numpy()
            all_eff_rank[name] = eff_rank.item()
            print(f"✓ {name}: Effective Rank = {eff_rank.item():.2f}")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
            continue
    
    # 左图：奇异值分布（对数坐标）
    ax1 = axes[0]
    for name, sv in all_sv.items():
        config = EXPERIMENTS[name]
        ax1.semilogy(np.arange(1, len(sv) + 1), sv,
                     label=f"{name} (eff_rank={all_eff_rank[name]:.1f})",
                     color=config['color'],
                     linestyle=config['linestyle'],
                     linewidth=2.0)
    
    ax1.set_xlabel('Singular Value Index', fontsize=12)
    ax1.set_ylabel('Singular Value (log scale)', fontsize=12)
    ax1.set_title('A) Singular Value Spectrum', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右图：归一化奇异值分布（累积贡献）
    ax2 = axes[1]
    for name, sv in all_sv.items():
        config = EXPERIMENTS[name]
        sv_normalized = sv / sv.sum()
        sv_cumsum = np.cumsum(sv_normalized)
        ax2.plot(np.arange(1, len(sv) + 1), sv_cumsum,
                 label=name,
                 color=config['color'],
                 linestyle=config['linestyle'],
                 linewidth=2.0)
    
    ax2.axhline(0.9, linestyle='--', color='gray', alpha=0.5, linewidth=1)
    if all_sv:
        first_sv = list(all_sv.values())[0]
        ax2.text(len(first_sv) * 0.7, 0.92, '90% variance', fontsize=10, color='gray')
    
    ax2.set_xlabel('Number of Components', fontsize=12)
    ax2.set_ylabel('Cumulative Variance Ratio', fontsize=12)
    ax2.set_title('B) Cumulative Variance Explained', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    print("=" * 55)
    print("Singular Value Spectrum Analysis")
    print("=" * 55)
    
    output_dir = Path('docs/assets')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_singular_value_spectrum(output_dir / 'singular_value_spectrum.png')
    
    print("\n" + "=" * 55)
    print("Analysis complete!")
    print("=" * 55)


if __name__ == '__main__':
    main()
