#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Effective Rank 对比图生成脚本
============================

从多种子实验的 checkpoint 中提取特征，计算 effective rank 并生成对比图。

使用方法:
    python plot_effective_rank.py

输出:
    results/comparison_figures/effective_rank_comparison.png
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})

# 实验配置
EXPERIMENTS = {
    'Baseline (ReLU)': {
        'color': '#7F8C8D',
        'specnorm': False,
        'layernorm': False,
    },
    'LayerNorm': {
        'color': '#3498DB',
        'specnorm': False,
        'layernorm': True,
    },
    'ReDo Reset': {
        'color': '#27AE60',
        'specnorm': False,
        'layernorm': False,
    },
    'Spectral Norm (Ours)': {
        'color': '#E74C3C',
        'specnorm': True,
        'layernorm': False,
    },
}

# 方法名到目录名的映射
METHOD_DIR_MAP = {
    'Baseline (ReLU)': 'baseline',
    'LayerNorm': 'layernorm',
    'ReDo Reset': 'redo',
    'Spectral Norm (Ours)': 'specnorm',
}


def find_checkpoint(method_name: str, seed: int) -> Path:
    """查找指定方法和种子的 checkpoint 文件（选择 epoch 最高的）"""
    method_dir = METHOD_DIR_MAP[method_name]
    
    best_checkpoint = None
    best_epoch = -1
    
    # 优先查找 multiseed 目录
    multiseed_dir = Path(f'results/multiseed/{method_dir}/seed_{seed}')
    if multiseed_dir.exists():
        # 递归查找所有 .pt 文件
        for pt_file in multiseed_dir.rglob('*.pt'):
            try:
                ckpt = torch.load(pt_file, map_location='cpu', weights_only=False)
                epoch = ckpt.get('epoch', 0)
                if epoch > best_epoch:
                    best_epoch = epoch
                    best_checkpoint = pt_file
            except:
                pass
    
    # 备选：查找 results/{method}_seed{seed} 目录
    if best_checkpoint is None:
        alt_dir = Path(f'results/{method_dir}_seed{seed}')
        if alt_dir.exists():
            for pt_file in alt_dir.rglob('*.pt'):
                try:
                    ckpt = torch.load(pt_file, map_location='cpu', weights_only=False)
                    epoch = ckpt.get('epoch', 0)
                    if epoch > best_epoch:
                        best_epoch = epoch
                        best_checkpoint = pt_file
                except:
                    pass
    
    return best_checkpoint


def create_model(specnorm=False, layernorm=False):
    """创建模型实例"""
    from algos.ppo.model import PPOModel
    
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
        'layernorm': layernorm,
        'rmsnorm': False,
        'specnorm': specnorm,
        'adapt_info': ['none', None],
    }
    
    obs_size = 64 * 64 * 3
    act_size = 9
    depth = 3
    
    return PPOModel(obs_size, act_size, depth, model_params)


def collect_observations(num_samples=2560, seed=0):
    """收集真实环境观测数据"""
    from envs.mdps import ProcGenEnv
    
    print(f"Collecting {num_samples} real observations from ProcGen CoinRun...")
    
    env = ProcGenEnv(
        shift_type='permute',
        task='coinrun_100',
        train=False,
        seed=seed
    )
    
    observations = []
    obs = env.reset()
    
    pbar = tqdm(total=num_samples, desc="Collecting observations")
    while len(observations) < num_samples:
        observations.append(obs)
        action = np.random.randint(0, env.action_space.n)
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
        pbar.update(1)
    pbar.close()
    
    return torch.stack(observations[:num_samples])


def compute_effective_rank(features: torch.Tensor) -> float:
    """计算有效秩"""
    N, D = features.shape
    
    # 中心化
    features = features - features.mean(dim=0, keepdim=True)
    
    try:
        S = torch.linalg.svdvals(features)
    except:
        return 0.0
    
    # 归一化
    S = S + 1e-10
    p = S / S.sum()
    
    # 熵
    entropy = -(p * torch.log(p)).sum()
    eff_rank = torch.exp(entropy).item()
    
    return eff_rank


def extract_features(model, observations, device='cpu'):
    """提取编码器特征"""
    model.eval()
    features = []
    batch_size = 256
    
    with torch.no_grad():
        for i in range(0, len(observations), batch_size):
            batch = observations[i:i+batch_size].to(device)
            feat = model.encoder(batch)
            features.append(feat.cpu())
    
    return torch.cat(features, dim=0)


def load_model_flexible(checkpoint_path: Path, specnorm: bool, layernorm: bool, device='cpu'):
    """灵活加载模型，处理结构不匹配的情况"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 创建匹配的模型
    model = create_model(specnorm=specnorm, layernorm=layernorm)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    except RuntimeError as e:
        # 尝试 strict=False
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"  Warning: Loaded with strict=False")
            return model
        except:
            return None


def main():
    print("=" * 60)
    print("Effective Rank Comparison Plot Generator")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    output_dir = Path('results/comparison_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集观测数据
    observations = collect_observations(num_samples=2560)
    print(f"Collected {len(observations)} observations")
    
    # 计算每个方法的 effective rank
    results = {}
    num_seeds = 5
    
    for method_name, config in EXPERIMENTS.items():
        print(f"\nProcessing: {method_name}")
        eff_ranks = []
        
        for seed in range(num_seeds):
            checkpoint_path = find_checkpoint(method_name, seed)
            
            if checkpoint_path is None:
                print(f"  Seed {seed}: checkpoint not found")
                continue
            
            print(f"  Seed {seed}: {checkpoint_path}")
            
            model = load_model_flexible(
                checkpoint_path,
                specnorm=config['specnorm'],
                layernorm=config['layernorm'],
                device=device
            )
            
            if model is None:
                print(f"  Seed {seed}: failed to load model")
                continue
            
            model.to(device)
            model.eval()
            
            # 提取特征并计算 effective rank
            features = extract_features(model, observations, device)
            eff_rank = compute_effective_rank(features)
            eff_ranks.append(eff_rank)
            print(f"  Seed {seed}: eff_rank = {eff_rank:.2f}")
        
        if eff_ranks:
            results[method_name] = {
                'mean': np.mean(eff_ranks),
                'std': np.std(eff_ranks),
                'values': eff_ranks,
                'config': config,
            }
    
    if not results:
        print("\nError: No results to plot!")
        return
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("Effective Rank Summary (Mean ± Std)")
    print("=" * 60)
    print(f"{'Method':<25} {'Effective Rank':>20}")
    print("-" * 60)
    for name, data in results.items():
        print(f"{name:<25} {data['mean']:>10.2f} ± {data['std']:.2f}")
    print("=" * 60)
    
    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    methods = list(results.keys())
    means = [results[m]['mean'] for m in methods]
    stds = [results[m]['std'] for m in methods]
    colors = [results[m]['config']['color'] for m in methods]
    
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Method', fontsize=13)
    ax.set_ylabel('Effective Rank', fontsize=13)
    ax.set_title('Effective Rank Comparison (Multi-Seed, Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 添加数值标签
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / 'effective_rank_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n✓ Saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
