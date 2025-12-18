#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征分析脚本 (Feature Analysis Script)
======================================

【这是正确的特征秩分析方法】

使用真实环境数据分析网络特征，包括：
1. 奇异值谱分布 (Singular Value Spectrum)
2. 有效秩 (Effective Rank)
3. 死神经元统计 (Dead Neuron Statistics)

关键改进（解决之前的方法论错误）：
1. 【数据源】使用真实 ProcGen 环境数据而非随机噪声
   - 随机噪声无法代表网络对真实环境的处理能力
   - 网络对噪声的响应与对结构化图像的响应完全不同
   
2. 【样本量】大批量采样确保 N >> D
   - N = 2500 样本, D = 256 特征维度
   - N/D ratio ≈ 9.8，避免秩被截断
   - 训练时 batch_size=64 < h_size=256 会导致秩计算失真
   
3. 【统计口径】累积统计死神经元（全数据集永久死亡率）
   - 只有在所有样本上都从未激活的神经元才计为"死神经元"
   - 区分 ReLU 的正常稀疏性与真正的神经元死亡

Usage:
    python analyze_features.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from envs.mdps import ProcGenEnv, GroupEnv
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
    'ReDo Reset': {
        'checkpoint': 'results/ablation_studies/redo_experiment/results/redo_reset/checkpoints/redo-reset_0.pt',
        'color': '#27AE60',
        'specnorm': False,
    },
    'Spectral Norm (Ours)': {
        'checkpoint': 'results/specnorm_experiment/checkpoints/specnorm_0.pt',
        'color': '#E74C3C',
        'specnorm': True,
    },
}

# 环境配置
ENV_CONFIG = {
    'name': 'procgen',
    'task': 'coinrun_100',
    'shift_type': 'permute',
    'obs_type': 'conv64',
}


def create_env(seed=0):
    """创建 ProcGen 环境"""
    env = ProcGenEnv(
        shift_type=ENV_CONFIG['shift_type'],
        task=ENV_CONFIG['task'],
        train=False,
        seed=seed
    )
    return env


def collect_real_observations(env, model, num_episodes=10, device='cpu'):
    """
    从真实环境中收集观测数据
    
    Args:
        env: ProcGen 环境
        model: 训练好的模型
        num_episodes: 收集的 episode 数量
        device: 计算设备
    
    Returns:
        observations: 收集的观测数据 (N, obs_dim)
    """
    observations = []
    model.eval()
    
    with torch.no_grad():
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            while not done:
                observations.append(obs)
                # 使用模型采样动作
                action, _, _ = model.sample_action(obs.unsqueeze(0).to(device))
                obs, reward, done, _ = env.step(action.item())
    
    return torch.stack(observations)


def collect_large_batch_observations(num_samples=2500, num_envs=8, seed=0):
    """
    收集大批量真实观测数据
    
    确保 N >> D (N=样本数, D=特征维度256)
    
    Args:
        num_samples: 目标样本数量 (建议 >= 2000)
        num_envs: 并行环境数量
        seed: 随机种子
    
    Returns:
        observations: (N, obs_dim) 的观测数据
    """
    envs = [create_env(seed + i) for i in range(num_envs)]
    
    observations = []
    obs_list = [env.reset() for env in envs]
    
    pbar = tqdm(total=num_samples, desc="Collecting observations")
    
    while len(observations) < num_samples:
        for i, env in enumerate(envs):
            observations.append(obs_list[i])
            # 随机动作探索环境
            action = np.random.randint(0, env.action_space.n)
            obs, reward, done, _ = env.step(action)
            if done:
                obs = env.reset()
            obs_list[i] = obs
        pbar.update(num_envs)
    
    pbar.close()
    return torch.stack(observations[:num_samples])


def load_model(checkpoint_path, specnorm=False, device='cpu'):
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
    
    obs_size = 64 * 64 * 3
    act_size = 9  # CoinRun
    depth = 3
    
    model = PPOModel(obs_size, act_size, depth, model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def extract_features(model, observations, device='cpu'):
    """
    提取编码器输出的特征
    
    Args:
        model: PPO 模型
        observations: (N, obs_dim) 观测数据
        device: 计算设备
    
    Returns:
        features: (N, h_size) 特征矩阵
    """
    model.eval()
    features = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(observations), batch_size):
            batch = observations[i:i+batch_size].to(device)
            x = batch.view(-1, model.obs_size)
            feat = model.encoder(x)
            features.append(feat.cpu())
    
    return torch.cat(features, dim=0)


def compute_effective_rank(features):
    """
    计算有效秩 (Effective Rank)
    
    使用归一化熵公式：
        eff_rank = exp(-sum(p_i * log(p_i)))
    其中 p_i = sigma_i / sum(sigma_j)
    
    Args:
        features: (N, D) 特征矩阵，要求 N >> D
    
    Returns:
        eff_rank: 有效秩
        singular_values: 奇异值数组
    """
    N, D = features.shape
    print(f"  Feature matrix shape: N={N}, D={D} (N/D ratio: {N/D:.1f})")
    
    if N < D:
        print(f"  Warning: N < D, effective rank will be truncated!")
    
    # 中心化
    features = features - features.mean(dim=0, keepdim=True)
    
    # SVD 分解
    try:
        _, S, _ = torch.svd(features)
    except Exception as e:
        print(f"  SVD failed: {e}")
        return 0.0, np.zeros(min(N, D))
    
    # 归一化奇异值
    S = S + 1e-10
    p = S / S.sum()
    
    # 计算熵
    entropy = -(p * torch.log(p)).sum()
    eff_rank = torch.exp(entropy).item()
    
    return eff_rank, S.numpy()


def compute_dead_neurons_cumulative(model, observations, device='cpu'):
    """
    累积统计死神经元（全数据集永久死亡率）
    
    只有在所有样本上都从未激活过的神经元才计为"死神经元"
    
    Args:
        model: PPO 模型
        observations: (N, obs_dim) 观测数据
        device: 计算设备
    
    Returns:
        dead_ratio: 死神经元比例
        activation_counts: 每个神经元的激活次数
    """
    model.eval()
    h_size = model.h_size
    activation_counts = torch.zeros(h_size)
    total_samples = 0
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(observations), batch_size):
            batch = observations[i:i+batch_size].to(device)
            x = batch.view(-1, model.obs_size)
            feat = model.encoder(x).cpu()
            
            # 统计每个神经元是否激活 (> 0)
            activated = (feat > 0).float()
            activation_counts += activated.sum(dim=0)
            total_samples += len(batch)
    
    # 计算从未激活过的神经元比例
    never_activated = (activation_counts == 0).float()
    dead_ratio = never_activated.mean().item()
    
    # 计算平均激活率
    avg_activation_rate = activation_counts / total_samples
    
    return dead_ratio, avg_activation_rate


def analyze_all_experiments(observations, device='cpu'):
    """分析所有实验"""
    results = {}
    
    for name, config in EXPERIMENTS.items():
        print(f"\n{'='*50}")
        print(f"Analyzing: {name}")
        print(f"{'='*50}")
        
        try:
            # 加载模型
            model = load_model(config['checkpoint'], config['specnorm'], device)
            
            # 提取特征
            print("Extracting features...")
            features = extract_features(model, observations, device)
            
            # 计算有效秩
            print("Computing effective rank...")
            eff_rank, singular_values = compute_effective_rank(features)
            print(f"  Effective Rank: {eff_rank:.2f}")
            
            # 计算死神经元（累积统计）
            print("Computing dead neurons (cumulative)...")
            dead_ratio, activation_rate = compute_dead_neurons_cumulative(
                model, observations, device
            )
            print(f"  Dead Neurons (never activated): {dead_ratio*100:.2f}%")
            print(f"  Average activation rate: {activation_rate.mean()*100:.2f}%")
            
            results[name] = {
                'eff_rank': eff_rank,
                'singular_values': singular_values,
                'dead_ratio': dead_ratio,
                'activation_rate': activation_rate.numpy(),
                'config': config,
            }
            
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    return results


def plot_singular_value_spectrum(results, output_path):
    """绘制奇异值谱对比图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # 左图：奇异值分布（对数坐标）
    ax1 = axes[0]
    for name, data in results.items():
        sv = data['singular_values']
        config = data['config']
        eff_rank = data['eff_rank']
        ax1.semilogy(
            np.arange(1, len(sv) + 1), sv,
            label=f"{name} (eff_rank={eff_rank:.1f})",
            color=config['color'],
            linewidth=2.0
        )
    
    ax1.set_xlabel('Singular Value Index', fontsize=12)
    ax1.set_ylabel('Singular Value (log scale)', fontsize=12)
    ax1.set_title('A) Singular Value Spectrum (Real Data)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # 右图：归一化奇异值累积分布
    ax2 = axes[1]
    for name, data in results.items():
        sv = data['singular_values']
        config = data['config']
        sv_normalized = sv / sv.sum()
        sv_cumsum = np.cumsum(sv_normalized)
        ax2.plot(
            np.arange(1, len(sv) + 1), sv_cumsum,
            label=name,
            color=config['color'],
            linewidth=2.0
        )
    
    ax2.axhline(0.9, linestyle='--', color='gray', alpha=0.5, linewidth=1)
    ax2.axhline(0.95, linestyle=':', color='gray', alpha=0.5, linewidth=1)
    ax2.text(len(sv) * 0.7, 0.92, '90% variance', fontsize=10, color='gray')
    
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


def plot_activation_distribution(results, output_path):
    """绘制神经元激活率分布图"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    for name, data in results.items():
        activation_rate = data['activation_rate']
        config = data['config']
        
        # 绘制激活率直方图
        ax.hist(
            activation_rate, bins=50, alpha=0.5,
            label=f"{name} (dead={data['dead_ratio']*100:.1f}%)",
            color=config['color']
        )
    
    ax.axvline(0, linestyle='--', color='red', alpha=0.7, linewidth=2)
    ax.set_xlabel('Activation Rate per Neuron', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Neuron Activation Rate Distribution', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def print_summary_table(results):
    """打印结果汇总表"""
    print("\n" + "=" * 70)
    print("Feature Analysis Summary (Using Real Environment Data)")
    print("=" * 70)
    print(f"{'Method':<25} {'Eff. Rank':>12} {'Dead Neurons':>15} {'Avg Act. Rate':>15}")
    print("-" * 70)
    
    for name, data in results.items():
        print(f"{name:<25} {data['eff_rank']:>12.2f} {data['dead_ratio']*100:>14.2f}% {data['activation_rate'].mean()*100:>14.2f}%")
    
    print("=" * 70)


def main():
    print("=" * 60)
    print("Feature Analysis with Real Environment Data")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path('results/feature_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集大批量真实观测数据 (N >> D=256)
    print("\n[1/4] Collecting real observations from ProcGen...")
    num_samples = 2500  # 确保 N >> D (256)
    observations = collect_large_batch_observations(num_samples=num_samples)
    print(f"Collected {len(observations)} observations")
    
    # 分析所有实验
    print("\n[2/4] Analyzing all experiments...")
    results = analyze_all_experiments(observations, device)
    
    if not results:
        print("Error: No results to analyze!")
        return
    
    # 打印汇总表
    print("\n[3/4] Generating summary...")
    print_summary_table(results)
    
    # 生成图表
    print("\n[4/4] Generating plots...")
    plot_singular_value_spectrum(results, output_dir / 'singular_value_spectrum_real.png')
    plot_activation_distribution(results, output_dir / 'activation_distribution.png')
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
