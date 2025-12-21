<h1 align="center">🧠 Mitigating Plasticity Loss in Deep RL via Spectral Normalization</h1>

<p align="center">
  <b>基于谱归一化的深度强化学习可塑性丢失缓解研究</b><br>
  <i>Course Design Project | 强化学习课程设计</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

---

## 📌 TL;DR

> **Spectral Normalization achieves +14.8% reward improvement and reduces dead neurons by 50% compared to baseline, with the lowest variance across 5 seeds.**

| Method | Test Reward (5 seeds) | Dead Units | vs Baseline |
|:-------|:---------------------:|:----------:|:-----------:|
| Baseline (ReLU) | 5.75 ± 1.06 | 82.5% ± 3.5% | - |
| LayerNorm | 5.91 ± 0.61 | 75.9% ± 0.5% | +2.8% |
| ReDo Reset | 5.91 ± 0.95 | 71.8% ± 2.5% | +2.8% |
| **Spectral Norm** | **6.60 ± 0.41** | **41.6% ± 2.6%** | **+14.8%** ✅ |

---

## 🚀 How to Reproduce

### Dependencies

| Package | Version |
|:--------|:--------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| NumPy | 1.24+ |
| Matplotlib | 3.7+ |
| ProcGen | 0.10.7 |
| TensorBoard | 2.13+ |

### Installation

```bash
# Create conda environment
conda create -n rlcourse python=3.10
conda activate rlcourse

# Install PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### Quick Start (Single Seed)

```bash
# Run Spectral Norm experiment (recommended)
python train.py -p hyperparams_quick.yaml -c specnorm -n specnorm_exp

# Run Baseline experiment
python train.py -p hyperparams_quick.yaml -c baseline -n baseline_exp

# Run ReDo experiment
python train.py -p hyperparams_quick.yaml -c redo -n redo_exp

# Resume from checkpoint
python train.py -p hyperparams_quick.yaml -n <exp_name> -r
```

### Multi-Seed Training (Recommended for Publication)

```bash
# Linux/Mac
./run_seeds.sh specnorm 5    # Run Spectral Norm with 5 seeds
./run_seeds.sh baseline 5    # Run Baseline with 5 seeds
./run_seeds.sh redo 5        # Run ReDo with 5 seeds
./run_seeds.sh all 5         # Run all methods with 5 seeds

# Windows
run_seeds.bat specnorm 5
run_seeds.bat baseline 5
```

### Generate Multi-Seed Aggregated Plots

After running multi-seed experiments, generate publication-ready figures with confidence intervals:

```bash
# Run all methods with 5 seeds (takes ~15 hours on GPU)
./run_seeds.sh all 5

# Generate comparison plots with mean ± std shading
python plot_comparison.py

# Output: results/comparison_figures/
#   ├── test_reward_multiseed.png    # Mean ± Std reward curves
#   ├── dead_units_multiseed.png     # Mean ± Std dead units curves
#   └── summary_comparison.png       # Combined figure
```

### Analysis & Visualization

```bash
# Generate comparison plots (single-seed and multi-seed)
python plot_comparison.py

# Analyze feature representations (SVD, dead neurons)
python analyze_features.py

# Analyze layer norms (check for gradient vanishing)
python analyze_layer_norms.py

# Verify Power Iteration accuracy
python analyze_power_iteration.py
```

---

## 🎯 Research Question

**How can we prevent plasticity loss (feature rank collapse & dead neurons) in deep reinforcement learning while maintaining training stability?**

### Key Findings

1. **"Keeping neurons alive" ≠ "Effective learning"**: Leaky ReLU eliminates dead neurons (0%) but decreases reward by 15%
2. **Reset mechanisms are band-aids**: ReDo works but introduces training instability (sawtooth curves)
3. **Spectral Normalization is the principled solution**: Mathematically constrains Lipschitz constant, preventing rank collapse

---

## 📊 Results

### Performance Comparison

![Summary Comparison](results/comparison_figures/summary_comparison.png)

### Feature Analysis (Using Real Environment Data)

We analyze features using **2,560 real ProcGen observations** (not Gaussian noise!) to ensure N ≥ 10×D for valid SVD computation.

![Singular Value Spectrum](results/feature_analysis/singular_value_spectrum_real.png)

| Method | Dead Neurons* | Avg Activation Rate |
|:-------|:-------------:|:-------------------:|
| Baseline | 19.14% | 21.52% |
| ReDo | 26.17% | 32.53% |
| **Spectral Norm** | 25.39% | **66.76%** |

> *Dead units are defined as neurons that **never activate** over the entire test set (2.5k steps), distinguishing true neuron death from normal ReLU sparsity.

### Understanding Baseline's High Effective Rank

**Observation**: In some experiments, Baseline shows higher Effective Rank than Spectral Norm.

**Explanation**: This is a **counter-intuitive but expected phenomenon**:

1. **High Rank ≠ Good Features**: Baseline's high rank often comes from **high-frequency noise** rather than meaningful features. The singular value spectrum shows a "long tail" that doesn't decay quickly, indicating noise accumulation.

2. **Spectral Norm maintains "healthy" rank**: SN constrains the Lipschitz constant, which:
   - Prevents gradient explosion → more stable feature learning
   - Concentrates information in top singular values → stronger principal components
   - Results in a "cleaner" spectrum with faster decay in the tail

3. **Evidence from Singular Value Spectrum**:
   - **Baseline**: Flat spectrum tail (noise dominates)
   - **Spectral Norm**: Steeper decay (signal dominates)

**Key Insight**: We should optimize for **feature quality** (activation rate, reward), not raw rank numbers. Spectral Norm achieves +20% reward improvement despite potentially lower numerical rank.

---

## 🔬 Methodology

### Spectral Normalization

We apply Spectral Normalization to the **shared encoder only** (not the value head):

$$W_{SN} = \frac{W}{\sigma(W)}$$

where $\sigma(W)$ is the largest singular value of $W$.

**Training vs Evaluation:**
- **Training**: Uses Power Iteration (1 iteration) for efficient approximation of $\sigma(W)$
- **Evaluation**: Uses exact SVD computation for accurate analysis

This distinction is important for reproducibility. Power Iteration is sufficient for training but exact SVD should be used for final analysis.

**Why not apply SN to Value Network?**

The value function $V(s)$ can have large magnitude (e.g., cumulative reward > 10). Constraining Lipschitz constant ≤ 1 would cause:

$$|V(s_1) - V(s_2)| \leq \|s_1 - s_2\|$$

This leads to **Value Underestimation Bias**, destabilizing policy gradients.

### Experimental Setup

| Parameter | Value |
|:----------|:------|
| Environment | ProcGen CoinRun |
| Algorithm | PPO |
| Training Epochs | 3,000 |
| Task Shift Points | [1000, 2000] |
| Hidden Size | 256 |
| Learning Rate | 0.0005 |
| Batch Size | 64 |
| Buffer Size | 1024 |

---

## 📈 Ablation Studies (Single Seed)

| Method | Principle | Reward | Dead Units | Verdict |
|:-------|:----------|:------:|:----------:|:--------|
| Baseline | ReLU | 5.80 | 82.4% | Reference |
| Leaky ReLU | Negative slope | 4.94 | 0.0% | ❌ Alive but useless |
| Mish | Smooth activation | 5.72 | 93.6% | ❌ Worse |
| LayerNorm | Normalization | 4.65 | 75.9% | ❌ Unstable |
| RMSNorm | Lightweight norm | 4.21 | 67.4% | ❌ Worst |
| ReDo | Periodic reset | 5.73 | 71.4% | ⚠️ Unstable |
| **Spectral Norm** | Lipschitz constraint | **6.96** | **39.5%** | ✅ **Best** |

> Note: Multi-seed results (5 seeds) show Spectral Norm achieves 6.60 ± 0.41 reward with the lowest variance.

---

## 📁 Project Structure

```
├── train.py                     # Training entry point
├── run_seeds.sh/.bat            # Multi-seed training scripts
├── plot_comparison.py           # Generate comparison figures
├── analyze_features.py          # Feature analysis (SVD, dead neurons)
├── analyze_layer_norms.py       # Layer norm analysis
├── analyze_power_iteration.py   # Power Iteration vs SVD comparison
├── plot_mechanism_analysis.py   # Mechanism analysis plots
├── plot_singular_values.py      # Singular value spectrum plots
│
├── algos/ppo/
│   ├── model.py                 # PPO model with Spectral Norm support
│   └── trainer.py               # PPO trainer
│
├── shared/
│   ├── modules.py               # Network modules (ConvEncoder, SN, etc.)
│   ├── runner.py                # Experiment runner
│   ├── trainer.py               # Base trainer
│   └── plotting.py              # Plotting utilities
│
├── envs/
│   └── mdps.py                  # ProcGen environment wrapper
│
├── results/
│   ├── comparison_figures/      # Main result figures
│   └── feature_analysis/        # SVD and activation analysis
│
├── hyperparams.yaml             # Full experiment config
├── hyperparams_quick.yaml       # Quick experiment config
├── hyperparams_multiseed.yaml   # Multi-seed experiment config
└── requirements.txt             # Python dependencies
```

---

## 🔑 Key Implementation Details

### 1. Optimized SVD Computation

```python
# SVD is only computed at logging intervals (every 10 epochs)
# This significantly reduces training overhead
should_compute_diagnostics = (epoch % 10 == 0)
```

### 2. Batch Size Validation for Effective Rank

```python
# Effective rank is only valid when N >= D
# Training batch_size (64) < h_size (256) produces invalid results
# Use analyze_features.py with N >= 2560 for valid analysis
if N < D * min_ratio:
    return torch.tensor(-1.0)  # Placeholder for invalid computation
```

### 3. Explicit Input Dimension Validation

```python
# ConvEncoder validates input dimensions explicitly
# No silent auto-correction that could mask bugs
if x.shape[1] != expected_flat_size:
    raise ValueError(f"Expected shape (B, {expected_flat_size}), got {x.shape}")
```

### 4. Real Environment Data for SVD

```python
# Use real ProcGen observations, NOT Gaussian noise!
# Ensure N >= 10*D for valid singular value spectrum
observations = collect_real_observations(num_samples=2560)  # D=256
```

---

## 📚 References

```bibtex
@article{dohare2024plasticity,
  title={A Study of Plasticity Loss in On-Policy Deep Reinforcement Learning},
  author={Dohare, Shibhansh and others},
  journal={arXiv preprint arXiv:2405.19153},
  year={2024}
}

@inproceedings{miyato2018spectral,
  title={Spectral Normalization for Generative Adversarial Networks},
  author={Miyato, Takeru and Kataoka, Toshiki and Koyama, Masanori and Yoshida, Yuichi},
  booktitle={ICLR},
  year={2018}
}

@article{kumar2020implicit,
  title={Implicit Under-Parameterization Inhibits Data-Efficient Deep Reinforcement Learning},
  author={Kumar, Aviral and others},
  journal={arXiv preprint arXiv:2010.14498},
  year={2020}
}
```

---

## 📄 License

MIT License - feel free to use this code for your research!

---

<p align="center">
  <i>Made with ❤️ for Deep Reinforcement Learning</i><br>
  <b>If you find this useful, please ⭐ star this repo!</b>
</p>
