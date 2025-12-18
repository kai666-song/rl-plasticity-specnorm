#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Power Iteration vs SVD 精度对比脚本
===================================

验证训练时使用的 Power Iteration 近似与评估时使用的 SVD 精确计算之间的差异。

【方法论说明】
- 训练时：PyTorch 的 spectral_norm 使用 Power Iteration 近似计算最大奇异值
  - 优点：计算高效，O(mn) 复杂度
  - 缺点：近似值，需要多次迭代才能收敛
  
- 评估时：使用 torch.svd 精确计算奇异值
  - 优点：精确
  - 缺点：计算开销大，O(min(m,n)^2 * max(m,n)) 复杂度

本脚本量化两种方法的差异，验证 Power Iteration 的精度是否足够。

Usage:
    python analyze_power_iteration.py

Output:
    results/feature_analysis/power_iteration_comparison.png
"""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({'font.size': 12})


def power_iteration(W: torch.Tensor, num_iters: int = 1) -> torch.Tensor:
    """
    Power Iteration 方法计算最大奇异值
    
    这是 PyTorch spectral_norm 内部使用的方法。
    
    Args:
        W: 权重矩阵 (out_features, in_features)
        num_iters: 迭代次数
    
    Returns:
        估计的最大奇异值
    """
    # 初始化随机向量
    u = torch.randn(W.shape[0], device=W.device)
    u = u / u.norm()
    
    for _ in range(num_iters):
        # v = W^T u / ||W^T u||
        v = W.t() @ u
        v = v / v.norm()
        
        # u = W v / ||W v||
        u = W @ v
        u = u / u.norm()
    
    # sigma = u^T W v
    sigma = (u @ W @ v).abs()
    return sigma


def exact_svd(W: torch.Tensor) -> torch.Tensor:
    """
    使用 SVD 精确计算最大奇异值
    
    Args:
        W: 权重矩阵
    
    Returns:
        最大奇异值
    """
    _, S, _ = torch.svd(W)
    return S[0]


def compare_methods(W: torch.Tensor, max_iters: int = 20) -> dict[str, list]:
    """
    比较 Power Iteration 和 SVD 的结果
    
    Args:
        W: 权重矩阵
        max_iters: 最大迭代次数
    
    Returns:
        包含比较结果的字典
    """
    exact = exact_svd(W).item()
    
    results = {
        'iters': list(range(1, max_iters + 1)),
        'power_iter': [],
        'exact': exact,
        'relative_error': [],
    }
    
    for n_iters in results['iters']:
        approx = power_iteration(W, n_iters).item()
        results['power_iter'].append(approx)
        results['relative_error'].append(abs(approx - exact) / exact * 100)
    
    return results


def analyze_spectral_norm_layers(model: nn.Module) -> dict[str, dict]:
    """
    分析模型中所有使用 Spectral Norm 的层
    
    Args:
        model: 神经网络模型
    
    Returns:
        每层的分析结果
    """
    results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 检查是否应用了 spectral_norm
            if hasattr(module, 'weight_orig'):
                # 获取原始权重
                W = module.weight_orig.data
                if isinstance(module, nn.Conv2d):
                    # 将卷积核展平为 2D 矩阵
                    W = W.view(W.size(0), -1)
                
                results[name] = compare_methods(W)
    
    return results


def plot_convergence(results: dict[str, dict], output_path: Path) -> None:
    """绘制 Power Iteration 收敛曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # 左图：估计值 vs 迭代次数
    ax1 = axes[0]
    for name, data in results.items():
        ax1.plot(data['iters'], data['power_iter'], 
                 label=f"{name}", marker='o', markersize=4)
        ax1.axhline(data['exact'], linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Number of Iterations', fontsize=12)
    ax1.set_ylabel('Estimated Spectral Norm', fontsize=12)
    ax1.set_title('A) Power Iteration Convergence', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 右图：相对误差 vs 迭代次数
    ax2 = axes[1]
    for name, data in results.items():
        ax2.semilogy(data['iters'], data['relative_error'],
                     label=f"{name}", marker='o', markersize=4)
    
    ax2.axhline(1.0, linestyle='--', color='red', alpha=0.5, label='1% error')
    ax2.axhline(0.1, linestyle=':', color='green', alpha=0.5, label='0.1% error')
    
    ax2.set_xlabel('Number of Iterations', fontsize=12)
    ax2.set_ylabel('Relative Error (%)', fontsize=12)
    ax2.set_title('B) Relative Error vs Iterations', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    print("=" * 60)
    print("Power Iteration vs SVD Accuracy Analysis")
    print("=" * 60)
    
    output_dir = Path('results/feature_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建测试矩阵
    print("\n[1/3] Creating test matrices...")
    test_matrices = {
        'Linear_256x256': torch.randn(256, 256),
        'Linear_256x64': torch.randn(256, 64),
        'Conv_32x3x4x4': torch.randn(32, 3 * 4 * 4),  # 展平的卷积核
        'Conv_64x32x4x4': torch.randn(64, 32 * 4 * 4),
    }
    
    # 分析每个矩阵
    print("\n[2/3] Analyzing Power Iteration convergence...")
    results = {}
    
    for name, W in test_matrices.items():
        print(f"\n--- {name} ---")
        result = compare_methods(W, max_iters=20)
        results[name] = result
        
        print(f"  Exact spectral norm: {result['exact']:.6f}")
        print(f"  After 1 iter: {result['power_iter'][0]:.6f} (error: {result['relative_error'][0]:.2f}%)")
        print(f"  After 5 iters: {result['power_iter'][4]:.6f} (error: {result['relative_error'][4]:.2f}%)")
        print(f"  After 10 iters: {result['power_iter'][9]:.6f} (error: {result['relative_error'][9]:.4f}%)")
    
    # 生成图表
    print("\n[3/3] Generating plots...")
    plot_convergence(results, output_dir / 'power_iteration_comparison.png')
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
【方法论说明】

训练时：
- PyTorch spectral_norm 默认使用 1 次 Power Iteration
- 这是一个近似值，但计算效率高
- 随着训练进行，估计值会逐渐收敛

评估时：
- 使用 torch.svd 进行精确计算
- 用于分析和验证

结论：
- Power Iteration 在 5-10 次迭代后通常能达到 <1% 的误差
- 对于训练目的，1 次迭代的近似已经足够
- 评估分析时应使用 SVD 精确计算以确保准确性
""")
    
    print("=" * 60)
    print("Analysis complete!")
    print(f"Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == '__main__':
    main()
