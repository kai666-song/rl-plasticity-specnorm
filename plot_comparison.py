#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比图生成脚本 (Comparison Plot Generator)
==========================================

生成所有实验方法的对比图，包含完整的消融实验结果。

使用方法:
    python plot_comparison.py

输出:
    results/comparison_figures/
    ├── test_reward_comparison.png    # 测试奖励对比图
    ├── dead_units_comparison.png     # 死神经元比例对比图
    └── summary_comparison.png        # 综合对比图 (2x1 子图)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# 设置字体支持
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})


# 核心方法对比配置 - Baseline, LayerNorm, ReDo, Spectral Norm (3000 epochs)
EXPERIMENTS = {
    # 基准方法
    'Baseline (ReLU)': {
        'checkpoint': 'results/baseline/checkpoints/baseline_0.pt',
        'color': '#7F8C8D',
        'linestyle': '-',
        'linewidth': 2.0,
        'zorder': 1,
    },
    # LayerNorm (工业界标准)
    'LayerNorm': {
        'checkpoint': 'results/ablation_studies/layernorm_experiment/results/layernorm_2/checkpoints/layernorm_0.pt',
        'color': '#3498DB',
        'linestyle': '--',
        'linewidth': 2.0,
        'zorder': 2,
    },
    # ReDo 方法 (3000 epochs)
    'ReDo Reset': {
        'checkpoint': 'results/ablation_studies/redo_experiment/results/redo_reset/checkpoints/redo-reset_0.pt',
        'color': '#27AE60',
        'linestyle': '-.',
        'linewidth': 2.0,
        'zorder': 3,
    },
    # 我们的方法
    'Spectral Norm (Ours)': {
        'checkpoint': 'results/specnorm_experiment/checkpoints/specnorm_0.pt',
        'color': '#E74C3C',
        'linestyle': '-',
        'linewidth': 2.5,
        'zorder': 10,
    },
}

# 完整消融实验配置
EXPERIMENTS_FULL = {
    'Baseline (ReLU)': {
        'checkpoint': 'results/baseline/checkpoints/baseline_0.pt',
        'color': '#7F8C8D',
        'linestyle': '-',
        'linewidth': 1.5,
        'zorder': 1,
    },
    'Leaky ReLU': {
        'checkpoint': 'results/ablation_studies/leaky_relu_experiment/checkpoints/leaky_relu_0.pt',
        'color': '#9B59B6',
        'linestyle': '--',
        'linewidth': 1.5,
        'zorder': 2,
    },
    'Mish': {
        'checkpoint': 'results/ablation_studies/mish_experiment/checkpoints/mish_0.pt',
        'color': '#3498DB',
        'linestyle': '--',
        'linewidth': 1.5,
        'zorder': 2,
    },
    'RMSNorm': {
        'checkpoint': 'results/ablation_studies/rmsnorm_experiment/checkpoints/rmsnorm_0.pt',
        'color': '#E67E22',
        'linestyle': ':',
        'linewidth': 1.5,
        'zorder': 2,
    },
    'ReDo Reset': {
        'checkpoint': 'results/ablation_studies/redo_experiment/results/redo_reset/checkpoints/redo-reset_0.pt',
        'color': '#27AE60',
        'linestyle': '-.',
        'linewidth': 1.8,
        'zorder': 3,
    },
    'Spectral Norm (Ours)': {
        'checkpoint': 'results/specnorm_experiment/checkpoints/specnorm_0.pt',
        'color': '#E74C3C',
        'linestyle': '-',
        'linewidth': 2.5,
        'zorder': 10,
    },
}

SHIFT_POINTS = [1000, 2000]


def load_checkpoint_data(checkpoint_path):
    """从checkpoint文件加载数据"""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    result_dict = checkpoint['result_dict']
    
    data = {}
    for key, values in result_dict.items():
        if isinstance(values, list) and len(values) > 0:
            if hasattr(values[0], 'item'):
                data[key] = np.array([v.item() for v in values])
            else:
                data[key] = np.array(values)
    return data


def smooth_data(data, window=20):
    """平滑数据"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='same')
    smoothed[:window//2] = data[:window//2]
    smoothed[-window//2:] = data[-window//2:]
    return smoothed


def load_all_experiments(experiments=None):
    """加载所有实验数据"""
    if experiments is None:
        experiments = EXPERIMENTS
    all_data = {}
    for name, config in experiments.items():
        try:
            data = load_checkpoint_data(config['checkpoint'])
            print(f"✓ Loaded: {name} (epochs: {len(data.get('dead_units', []))})")
            all_data[name] = {'data': data, 'config': config}
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    return all_data



def plot_metric_comparison(all_data, metric, ylabel, title, output_path, 
                           test_interval=50, smooth_window=20):
    """绘制单个指标的对比图"""
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    
    sorted_items = sorted(all_data.items(), key=lambda x: x[1]['config'].get('zorder', 1))
    
    for name, exp_data in sorted_items:
        data = exp_data['data']
        config = exp_data['config']
        
        if metric not in data:
            continue
        
        values = data[metric]
        
        if metric == 'test_r':
            x = np.arange(len(values))
            xp = np.linspace(0, len(values) - 1, len(values) * test_interval)
            values = np.interp(xp, x, values)
        
        smoothed = smooth_data(values, smooth_window)
        epochs = np.arange(len(smoothed))
        
        ax.plot(epochs, smoothed, 
                label=name,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                zorder=config.get('zorder', 1),
                alpha=0.9)
    
    for i, sp in enumerate(SHIFT_POINTS):
        ax.axvline(sp, linestyle='--', color='gray', alpha=0.5, linewidth=1)
        ax.text(sp + 30, ax.get_ylim()[1] * 0.95, f'Task {i + 2}', fontsize=10, color='gray')
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_summary_comparison(all_data, output_path):
    """绘制综合对比图 (2x1 子图)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    metrics = [
        ('test_r', 'Test Reward', 'A) Test Reward Comparison'),
        ('dead_units', 'Dead Units Ratio', 'B) Dead Units Comparison'),
    ]
    
    sorted_items = sorted(all_data.items(), key=lambda x: x[1]['config'].get('zorder', 1))
    
    for ax, (metric, ylabel, title) in zip(axes, metrics):
        for name, exp_data in sorted_items:
            data = exp_data['data']
            config = exp_data['config']
            
            if metric not in data:
                continue
            
            values = data[metric]
            
            if metric == 'test_r':
                x = np.arange(len(values))
                xp = np.linspace(0, len(values) - 1, len(values) * 50)
                values = np.interp(xp, x, values)
            
            smoothed = smooth_data(values, 20)
            epochs = np.arange(len(smoothed))
            
            ax.plot(epochs, smoothed,
                    label=name,
                    color=config['color'],
                    linestyle=config['linestyle'],
                    linewidth=config['linewidth'],
                    zorder=config.get('zorder', 1),
                    alpha=0.9)
        
        for sp in SHIFT_POINTS:
            ax.axvline(sp, linestyle='--', color='gray', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    axes[0].legend(loc='lower right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")



def generate_results_table(all_data):
    """生成结果汇总表格"""
    print("\n" + "="*75)
    print("实验结果汇总 (Experiment Results Summary)")
    print("="*75)
    print(f"{'Method':<25} {'Test Reward':>15} {'Dead Units':>15} {'vs Baseline':>15}")
    print("-"*75)
    
    baseline_reward = None
    if 'Baseline (ReLU)' in all_data:
        baseline_data = all_data['Baseline (ReLU)']['data']
        test_r = baseline_data.get('test_r', [])
        baseline_reward = np.mean(test_r[-10:]) if len(test_r) >= 10 else np.mean(test_r)
    
    results = []
    for name, exp_data in all_data.items():
        data = exp_data['data']
        
        test_r = data.get('test_r', [])
        dead_units = data.get('dead_units', [])
        
        avg_r = np.mean(test_r[-10:]) if len(test_r) >= 10 else np.mean(test_r)
        avg_dead = np.mean(dead_units[-10:]) if len(dead_units) >= 10 else np.mean(dead_units)
        
        if baseline_reward and name != 'Baseline (ReLU)':
            improvement = ((avg_r - baseline_reward) / baseline_reward) * 100
            imp_str = f"{improvement:+.1f}%"
        else:
            imp_str = "-"
        
        results.append((name, avg_r, avg_dead, imp_str))
        print(f"{name:<25} {avg_r:>15.2f} {avg_dead*100:>14.1f}% {imp_str:>15}")
    
    print("="*75)
    return results


def main():
    """主函数"""
    print("="*55)
    print("Deep RL Plasticity - Full Ablation Comparison Generator")
    print("="*55)
    
    output_dir = Path('results/comparison_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n[1/4] Loading all experiment data...")
    all_data = load_all_experiments()
    
    if not all_data:
        print("Error: No experiment data found!")
        return
    
    print("\n[2/4] Generating results summary...")
    generate_results_table(all_data)
    
    print("\n[3/5] Generating individual comparison plots...")
    plot_metric_comparison(
        all_data, 'test_r', 'Test Reward', 
        'Test Reward Comparison (All Methods)',
        output_dir / 'test_reward_comparison.png'
    )
    
    plot_metric_comparison(
        all_data, 'dead_units', 'Dead Units Ratio',
        'Dead Units Comparison (All Methods)', 
        output_dir / 'dead_units_comparison.png'
    )
    
    # 检查是否有 eff_rank 数据
    has_eff_rank = any('eff_rank' in exp['data'] for exp in all_data.values())
    if has_eff_rank:
        print("\n[4/5] Generating effective rank comparison plot...")
        plot_metric_comparison(
            all_data, 'eff_rank', 'Effective Rank',
            'Effective Rank Comparison (All Methods)',
            output_dir / 'effective_rank_comparison.png'
        )
    else:
        print("\n[4/5] Skipping effective rank plot (no data available)")
    
    print("\n[5/5] Generating summary comparison plot...")
    plot_summary_comparison(all_data, output_dir / 'summary_comparison.png')
    
    print("\n" + "="*55)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*55)


if __name__ == '__main__':
    main()
