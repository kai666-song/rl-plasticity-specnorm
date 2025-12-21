#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
奇异值谱对比图生成脚本 (Singular Value Spectrum Plot Generator)
===============================================================

注意：由于模型 checkpoint 结构可能与当前代码不完全匹配，
建议使用 analyze_features.py 生成的图片，或直接使用已有的结果。

已有结果位置：
- results/feature_analysis/singular_value_spectrum_real.png

如果需要重新生成，请确保 checkpoint 与当前模型定义兼容。
"""

import shutil
from pathlib import Path


def main():
    """
    主函数 - 复制已有的奇异值谱图到 docs/assets
    """
    print("=" * 55)
    print("Singular Value Spectrum - Copy Existing Results")
    print("=" * 55)
    
    source = Path('results/feature_analysis/singular_value_spectrum_real.png')
    dest_dir = Path('docs/assets')
    dest = dest_dir / 'singular_value_spectrum.png'
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if source.exists():
        shutil.copy(source, dest)
        print(f"✓ Copied: {source} -> {dest}")
    else:
        print(f"✗ Source not found: {source}")
        print("  Please run analyze_features.py first to generate the plot.")
        print("  Or check if results/feature_analysis/ contains the required files.")
    
    print("\n" + "=" * 55)
    print("Done!")
    print("=" * 55)


if __name__ == '__main__':
    main()
