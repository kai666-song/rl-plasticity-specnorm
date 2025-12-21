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

# 多种子实验配置
# 路径模式: results/multiseed/{method}/seed_{seed}/results/{method}_seed{seed}/checkpoints/{method}_0.pt
# 注意：由于之前 num_sessions 配置问题，checkpoint 文件名可能是 {method}_0.pt 而非 {method}_seed{seed}_0.pt
MULTISEED_EXPERIMENTS = {
    'Baseline (ReLU)': {
        'checkpoint_pattern': 'results/multiseed/baseline/seed_{seed}/results/baseline_seed{seed}/checkpoints/baseline_0.pt',
        # 备选路径模式（自动扫描时使用）
        'alt_patterns': [
            'results/multiseed/baseline/seed_{seed}/checkpoints/baseline_seed{seed}_0.pt',
            'results/multiseed/baseline/seed_{seed}/checkpoints/baseline_0.pt',
        ],
        'color': '#7F8C8D',
        'linestyle': '-',
        'linewidth': 2.0,
        'zorder': 1,
    },
    'ReDo Reset': {
        'checkpoint_pattern': 'results/multiseed/redo/seed_{seed}/results/redo_seed{seed}/checkpoints/redo_0.pt',
        'alt_patterns': [
            'results/multiseed/redo/seed_{seed}/checkpoints/redo_seed{seed}_0.pt',
            'results/multiseed/redo/seed_{seed}/checkpoints/redo_0.pt',
        ],
        'color': '#27AE60',
        'linestyle': '-.',
        'linewidth': 2.0,
        'zorder': 3,
    },
    'Spectral Norm (Ours)': {
        'checkpoint_pattern': 'results/multiseed/specnorm/seed_{seed}/results/specnorm_seed{seed}/checkpoints/specnorm_0.pt',
        'alt_patterns': [
            'results/multiseed/specnorm/seed_{seed}/checkpoints/specnorm_seed{seed}_0.pt',
            'results/multiseed/specnorm/seed_{seed}/checkpoints/specnorm_0.pt',
        ],
        'color': '#E74C3C',
        'linestyle': '-',
        'linewidth': 2.5,
        'zorder': 10,
    },
    'LayerNorm': {
        'checkpoint_pattern': 'results/multiseed/layernorm/seed_{seed}/results/layernorm_seed{seed}/checkpoints/layernorm_0.pt',
        'alt_patterns': [
            'results/multiseed/layernorm/seed_{seed}/checkpoints/layernorm_seed{seed}_0.pt',
            'results/multiseed/layernorm/seed_{seed}/checkpoints/layernorm_0.pt',
        ],
        'color': '#3498DB',
        'linestyle': '--',
        'linewidth': 2.0,
        'zorder': 2,
    },
}


def load_checkpoint_data(checkpoint_path: str) -> dict:
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


def load_multi_seed_data(
    method_name: str,
    checkpoint_pattern: str,
    num_seeds: int = 5,
    alt_patterns: list = None
) -> dict:
    """
    加载多个种子的实验数据（支持多种路径模式自动扫描）
    
    Args:
        method_name: 方法名称（用于日志输出）
        checkpoint_pattern: 主 checkpoint 路径模式，使用 {seed} 占位符
        num_seeds: 种子数量
        alt_patterns: 备选路径模式列表，当主模式找不到文件时尝试
    
    Returns:
        包含每个 metric 的 'mean' 和 'std' 的字典，如果没有数据则返回 None
    
    Example:
        >>> data = load_multi_seed_data(
        ...     'Baseline',
        ...     'results/multiseed/baseline/seed_{seed}/checkpoints/baseline_0.pt',
        ...     num_seeds=5,
        ...     alt_patterns=['results/baseline/checkpoints/baseline_{seed}.pt']
        ... )
        >>> print(data['test_r']['mean'].shape)  # (num_epochs,)
    """
    all_seed_data = []
    loaded_seeds = []
    
    # 收集所有可能的路径模式
    patterns_to_try = [checkpoint_pattern]
    if alt_patterns:
        patterns_to_try.extend(alt_patterns)
    
    for seed in range(num_seeds):
        loaded = False
        for pattern in patterns_to_try:
            path = pattern.format(seed=seed)
            if Path(path).exists():
                try:
                    data = load_checkpoint_data(path)
                    all_seed_data.append(data)
                    loaded_seeds.append(seed)
                    loaded = True
                    break  # 找到一个就停止
                except Exception as e:
                    print(f"  ⚠ Failed to load seed {seed} from {path}: {e}")
        
        if not loaded:
            # 尝试自动扫描目录中的文件（包括子目录）
            method_lower = method_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            # 提取方法名的关键词（支持多种命名方式）
            method_keyword_map = {
                'baseline': 'baseline',
                'specnorm': 'specnorm',
                'spectral': 'specnorm',  # "Spectral Norm" -> specnorm
                'redo': 'redo',
                'layernorm': 'layernorm',
            }
            method_key = None
            for kw, folder in method_keyword_map.items():
                if kw in method_lower:
                    method_key = folder
                    break
            
            if method_key:
                # 扫描 results/multiseed/{method}/seed_{seed} 目录
                seed_dir = Path(f'results/multiseed/{method_key}/seed_{seed}')
                if seed_dir.exists():
                    # 递归查找所有 .pt 文件
                    for pt_file in seed_dir.rglob('*.pt'):
                        try:
                            ckpt = torch.load(pt_file, map_location='cpu', weights_only=False)
                            epoch = ckpt.get('epoch', -1)
                            # 只加载完成的实验 (epoch >= 2999)
                            if epoch >= 2999:
                                data = load_checkpoint_data(str(pt_file))
                                all_seed_data.append(data)
                                loaded_seeds.append(seed)
                                loaded = True
                                print(f"    Found: {pt_file} (epoch {epoch})")
                                break
                        except Exception as e:
                            pass
    
    if not all_seed_data:
        print(f"  ✗ No data found for {method_name}")
        return None
    
    print(f"  ✓ Loaded {method_name}: {len(all_seed_data)}/{num_seeds} seeds (seeds: {loaded_seeds})")
    
    # 计算均值和标准差
    result = {}
    for key in all_seed_data[0].keys():
        values = [d[key] for d in all_seed_data if key in d]
        if values:
            # 对齐长度（取最短的）
            min_len = min(len(v) for v in values)
            values = np.array([v[:min_len] for v in values])
            result[key] = {
                'mean': np.mean(values, axis=0),
                'std': np.std(values, axis=0),
                'n_seeds': len(values)
            }
    
    return result


def load_all_multiseed_experiments(experiments=None, num_seeds: int = 5):
    """
    加载所有多种子实验数据（支持自动扫描多种路径模式）
    
    Args:
        experiments: 实验配置字典，默认使用 MULTISEED_EXPERIMENTS
        num_seeds: 每个方法的种子数量
    
    Returns:
        包含所有方法数据的字典
    """
    if experiments is None:
        experiments = MULTISEED_EXPERIMENTS
    
    all_data = {}
    print("Loading multi-seed experiment data...")
    
    for name, config in experiments.items():
        data = load_multi_seed_data(
            name,
            config['checkpoint_pattern'],
            num_seeds,
            alt_patterns=config.get('alt_patterns', [])
        )
        if data is not None:
            all_data[name] = {'data': data, 'config': config}
    
    return all_data


def plot_with_confidence_interval(
    ax: plt.Axes,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    color: str,
    label: str,
    linestyle: str = '-',
    linewidth: float = 2.0,
    alpha: float = 0.2,
    zorder: int = 1
) -> None:
    """
    绘制带置信区间（标准差阴影）的曲线
    
    Args:
        ax: matplotlib axes
        x: x 轴数据
        mean: 均值
        std: 标准差
        color: 颜色
        label: 图例标签
        linestyle: 线条样式
        linewidth: 线条宽度
        alpha: 阴影透明度
        zorder: 绘制顺序
    """
    ax.plot(x, mean, color=color, label=label, linestyle=linestyle, 
            linewidth=linewidth, zorder=zorder)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=alpha, zorder=zorder-1)


def plot_multiseed_comparison(
    all_data: dict,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    test_interval: int = 50,
    smooth_window: int = 20
) -> None:
    """
    绘制多种子实验的对比图（带置信区间）
    
    Args:
        all_data: 多种子实验数据
        metric: 要绘制的指标
        ylabel: y 轴标签
        title: 图表标题
        output_path: 输出路径
        test_interval: 测试间隔（用于插值 test_r）
        smooth_window: 平滑窗口大小
    """
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    
    sorted_items = sorted(all_data.items(), key=lambda x: x[1]['config'].get('zorder', 1))
    
    for name, exp_data in sorted_items:
        data = exp_data['data']
        config = exp_data['config']
        
        if metric not in data:
            continue
        
        mean = data[metric]['mean']
        std = data[metric]['std']
        n_seeds = data[metric]['n_seeds']
        
        # 对 test_r 进行插值
        if metric == 'test_r':
            x = np.arange(len(mean))
            xp = np.linspace(0, len(mean) - 1, len(mean) * test_interval)
            mean = np.interp(xp, x, mean)
            std = np.interp(xp, x, std)
        
        # 平滑
        mean = smooth_data(mean, smooth_window)
        std = smooth_data(std, smooth_window)
        epochs = np.arange(len(mean))
        
        # 在标签中显示种子数量
        label = f"{name} (n={n_seeds})"
        
        plot_with_confidence_interval(
            ax, epochs, mean, std,
            color=config['color'],
            label=label,
            linestyle=config['linestyle'],
            linewidth=config['linewidth'],
            zorder=config.get('zorder', 1)
        )
    
    # 添加任务切换点标记
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
    
    print("\n[1/6] Loading single-seed experiment data...")
    all_data = load_all_experiments()
    
    if not all_data:
        print("Warning: No single-seed experiment data found!")
    else:
        print("\n[2/6] Generating results summary...")
        generate_results_table(all_data)
        
        print("\n[3/6] Generating single-seed comparison plots...")
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
            plot_metric_comparison(
                all_data, 'eff_rank', 'Effective Rank',
                'Effective Rank Comparison (All Methods)',
                output_dir / 'effective_rank_comparison.png'
            )
        
        print("\n[4/6] Generating summary comparison plot...")
        plot_summary_comparison(all_data, output_dir / 'summary_comparison.png')
    
    # 尝试加载多种子数据
    print("\n[5/6] Loading multi-seed experiment data...")
    multiseed_data = load_all_multiseed_experiments(num_seeds=5)
    
    if multiseed_data:
        print("\n[6/6] Generating multi-seed comparison plots (with confidence intervals)...")
        
        plot_multiseed_comparison(
            multiseed_data, 'test_r', 'Test Reward',
            'Test Reward Comparison (Multi-Seed, Mean ± Std)',
            output_dir / 'test_reward_multiseed.png'
        )
        
        plot_multiseed_comparison(
            multiseed_data, 'dead_units', 'Dead Units Ratio',
            'Dead Units Comparison (Multi-Seed, Mean ± Std)',
            output_dir / 'dead_units_multiseed.png'
        )
    else:
        print("\n[6/6] Skipping multi-seed plots (no data available)")
        print("  To generate multi-seed data, run: ./run_seeds.sh all 5")
    
    print("\n" + "="*55)
    print("All plots generated successfully!")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*55)


if __name__ == '__main__':
    main()
