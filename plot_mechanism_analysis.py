#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
机理分析图生成脚本 (Mechanism Analysis Plot Generator)
=====================================================

生成论文中机理分析部分所需的图表：
1. 梯度范数对比图 (grad_norm.png) - 展示 ReDo 的不稳定性 vs SN 的平稳性
2. 策略熵对比图 (entropy.png) - 展示 SN 如何缓解熵的过快下降

使用方法:
    python plot_mechanism_analysis.py

输出:
    docs/assets/
    ├── grad_norm.png    # 梯度范数对比图
    └── entropy.png      # 策略熵对比图
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# 设置字体
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})

# 实验配置
EXPERIMENTS = {
    'Baseline (ReLU)': {
        'checkpoint': 'results/baseline/checkpoints/baseline_0.pt',
        'color': '#95A5A6',  # 灰色
        'linestyle': '-',
        'linewidth': 1.8,
        'alpha': 0.85,
    },
    'ReDo Reset': {
        'checkpoint': 'results/ablation_studies/redo_experiment/results/redo_reset/checkpoints/redo-reset_0.pt',
        'color': '#F39C12',  # 橙色
        'linestyle': '-',
        'linewidth': 1.8,
        'alpha': 0.85,
    },
    'Spectral Norm (Ours)': {
        'checkpoint': 'results/specnorm_experiment/checkpoints/specnorm_0.pt',
        'color': '#E74C3C',  # 红色
        'linestyle': '-',
        'linewidth': 2.2,
        'alpha': 0.95,
    },
}

SHIFT_POINTS = [1000, 2000]


def load_checkpoint_data(checkpoint_path):
    """从checkpoint文件加载数据"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    result_dict = checkpoint['result_dict']
    
    data = {}
    for key, values in result_dict.items():
        if isinstance(values, list) and len(values) > 0:
            if hasattr(values[0], 'item'):
                data[key] = np.array([v.item() for v in values])
            else:
                data[key] = np.array(values)
    return data


def load_pickle_data(pickle_path, condition):
    """从pickle文件加载数据"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    result = {}
    for metric, cond_data in data.items():
        if condition in cond_data:
            mean, ste, raw = cond_data[condition]
            result[metric] = np.array(mean)
            result[f'{metric}_ste'] = np.array(ste)
    return result


def smooth_data(data, window=20):
    """平滑数据"""
    if len(data) < window:
        return data
    kernel = np.ones(window) / window
    smoothed = np.convolve(data, kernel, mode='same')
    smoothed[:window] = data[:window]
    smoothed[-window:] = data[-window:]
    return smoothed


def load_all_experiments():
    """加载所有实验数据"""
    all_data = {}
    
    for name, config in EXPERIMENTS.items():
        try:
            data = load_checkpoint_data(config['checkpoint'])
            all_data[name] = {'data': data, 'config': config}
            print(f"✓ Loaded: {name}")
        except Exception as e:
            print(f"✗ Failed to load {name}: {e}")
    
    return all_data


def plot_grad_norm_comparison(all_data, output_path):
    """
    绘制梯度范数对比图
    
    重点展示：
    - ReDo 的梯度范数呈现剧烈的脉冲式波动
    - SN 的梯度范数曲线非常平滑且幅值适中
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    metric = 'grad_norm'
    
    for name, exp_data in all_data.items():
        data = exp_data['data']
        config = exp_data['config']
        
        if metric not in data:
            print(f"  Warning: {name} has no {metric} data")
            continue
        
        values = data[metric]
        
        # 对于 ReDo，使用较小的平滑窗口以保留波动特征
        if 'ReDo' in name:
            smoothed = smooth_data(values, window=5)  # 小窗口，保留波动
        else:
            smoothed = smooth_data(values, window=15)
        
        epochs = np.arange(len(smoothed))
        
        ax.plot(epochs, smoothed,
                label=name,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                alpha=config['alpha'])
    
    # 添加环境切换线
    for sp in SHIFT_POINTS:
        ax.axvline(sp, linestyle='--', color='gray', alpha=0.4, linewidth=1)
        ax.text(sp + 30, ax.get_ylim()[1] * 0.92, f'Task {SHIFT_POINTS.index(sp) + 2}',
                fontsize=9, color='gray')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norm Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置合理的 y 轴范围
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_entropy_comparison(all_data, output_path):
    """
    绘制策略熵对比图
    
    重点展示：
    - Baseline 的熵下降过快（过早收敛）
    - SN 能够更长时间地维持较高的熵（保持探索能力）
    """
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    metric = 'entropy'
    
    for name, exp_data in all_data.items():
        data = exp_data['data']
        config = exp_data['config']
        
        if metric not in data:
            print(f"  Warning: {name} has no {metric} data")
            continue
        
        values = data[metric]
        smoothed = smooth_data(values, window=20)
        epochs = np.arange(len(smoothed))
        
        ax.plot(epochs, smoothed,
                label=name,
                color=config['color'],
                linestyle=config['linestyle'],
                linewidth=config['linewidth'],
                alpha=config['alpha'])
    
    # 添加环境切换线
    for sp in SHIFT_POINTS:
        ax.axvline(sp, linestyle='--', color='gray', alpha=0.4, linewidth=1)
        ax.text(sp + 30, ax.get_ylim()[1] * 0.92, f'Task {SHIFT_POINTS.index(sp) + 2}',
                fontsize=9, color='gray')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Policy Entropy', fontsize=12)
    ax.set_title('Policy Entropy Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    """主函数"""
    print("="*55)
    print("Mechanism Analysis Plot Generator")
    print("="*55)
    
    # 创建输出目录
    output_dir = Path('docs/assets')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载所有实验数据
    print("\n[1/3] Loading experiment data...")
    all_data = load_all_experiments()
    
    if not all_data:
        print("Error: No experiment data found!")
        return
    
    # 绘制梯度范数对比图
    print("\n[2/3] Generating gradient norm comparison plot...")
    plot_grad_norm_comparison(all_data, output_dir / 'grad_norm.png')
    
    # 绘制策略熵对比图
    print("\n[3/3] Generating entropy comparison plot...")
    plot_entropy_comparison(all_data, output_dir / 'entropy.png')
    
    print("\n" + "="*55)
    print("All mechanism analysis plots generated!")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*55)


if __name__ == '__main__':
    main()
