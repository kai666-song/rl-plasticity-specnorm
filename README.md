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

> **Spectral Normalization achieves +20% reward improvement and reduces dead neurons by 52% compared to baseline, outperforming all other methods including LayerNorm, ReDo, and activation function modifications.**

| Method | Test Reward | Dead Units | vs Baseline |
|:-------|:-----------:|:----------:|:-----------:|
| Baseline (ReLU) | 5.80 | 82.4% | - |
| LayerNorm | 4.65 | 75.9% | -19.8% ❌ |
| ReDo Reset | 5.73 | 71.4% | -1.1% |
| **Spectral Norm** | **6.96** | **39.5%** | **+20.0%** ✅ |

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

---

## 🚀 Quick Start

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

### Training

```bash
# Run Spectral Norm experiment (recommended)
python train.py -p hyperparams_quick.yaml -n specnorm_exp

# Run Baseline experiment
python train.py -p hyperparams.yaml -n baseline_exp

# Resume from checkpoint
python train.py -p hyperparams_quick.yaml -n <exp_name> -r
```

### Analysis

```bash
# Generate comparison plots
python plot_comparison.py

# Analyze feature representations (SVD, dead neurons)
python analyze_features.py

# Generate singular value spectrum
python plot_singular_values.py
```

---

## 📁 Project Structure

```
├── train.py                     # Training entry point
├── analyze_features.py          # Feature analysis (SVD, dead neurons)
├── plot_comparison.py           # Generate comparison figures
├── plot_singular_values.py      # Singular value spectrum analysis
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
├── hyperparams_quick.yaml       # Quick experiment config (3000 epochs)
├── hyperparams.yaml             # Full experiment config
└── requirements.txt             # Python dependencies
```

---

## 🔬 Methodology

### Spectral Normalization

We apply Spectral Normalization to the **shared encoder only** (not the value head):

$$W_{SN} = \frac{W}{\sigma(W)}$$

where $\sigma(W)$ is the largest singular value of $W$.

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

---

## 📈 Ablation Studies

| Method | Principle | Reward | Dead Units | Verdict |
|:-------|:----------|:------:|:----------:|:--------|
| Baseline | ReLU | 5.80 | 82.4% | Reference |
| Leaky ReLU | Negative slope | 4.94 | 0.0% | ❌ Alive but useless |
| Mish | Smooth activation | 5.72 | 93.6% | ❌ Worse |
| LayerNorm | Normalization | 4.65 | 75.9% | ❌ Industry standard fails |
| RMSNorm | Lightweight norm | 4.21 | 67.4% | ❌ Worst |
| ReDo | Periodic reset | 5.73 | 71.4% | ⚠️ Unstable |
| **Spectral Norm** | Lipschitz constraint | **6.96** | **39.5%** | ✅ **Best** |

---

## 🔑 Key Implementation Details

### 1. Flexible Input Handling

```python
# ConvEncoder supports both 4D (B,C,H,W) and 2D (B, C*H*W) inputs
def forward(self, x, check=False):
    if x.dim() == 2:
        x = x.view(-1, self.depth, self.conv_size, self.conv_size)
    # ... convolution layers
```

### 2. Cumulative Dead Neuron Statistics

```python
# Only neurons that NEVER activate are counted as "dead"
# This distinguishes normal ReLU sparsity from true neuron death
never_activated = (activation_counts == 0).float()
dead_ratio = never_activated.mean().item()
```

### 3. Real Environment Data for SVD

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
