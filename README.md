# Learning Rate Schedulers in Deep Learning: A Comprehensive Comparison

## The Learning Rate Problem

When training deep neural networks, the learning rate is one of the most critical hyperparameters. A fixed learning rate throughout training can lead to several issues:
- **Too high**: The model overshoots optimal weights, causing divergence or erratic loss curves
- **Too low**: Training becomes painfully slow, and the model may get stuck in local minima

**The solution?** Learning rate schedulers—algorithms that dynamically adjust the learning rate during training. In this project, we explore three popular schedulers through the lens of training ResNet18 on CIFAR-10.

---

## 🎯 Project Overview

This repository contains a comprehensive empirical study comparing three learning rate scheduling strategies:

1. **StepLR** — The traditional, rule-based scheduler
2. **ReduceLROnPlateau** — The adaptive, patience-based scheduler
3. **CosineAnnealingLR** — The modern, mathematically elegant scheduler

We train a ResNet18 model on CIFAR-10 using each scheduler and analyze the results to understand their strengths and weaknesses.

---

## 📚 Understanding the Schedulers

### 1. **StepLR** — The Classic Approach

**How it works:**
```
LR = LR × gamma^(epoch / step_size)
```

In our implementation:
- Step size: 10 epochs
- Gamma: 0.1
- This means: every 10 epochs, multiply the learning rate by 0.1

**Why use it?**
- Simple and predictable
- Good for understanding when LR decay is needed
- Works well when you know exactly when learning plateaus

**The trade-off:**
- Rigid schedule regardless of actual training progress
- May reduce LR when the model still has room to improve
- May keep LR high when the model has converged

### 2. **ReduceLROnPlateau** — The Adaptive Champion

**How it works:**
```
IF no improvement for N epochs:
    LR = LR × factor
```

In our implementation:
- Mode: 'min' (monitors validation loss)
- Factor: 0.1
- Patience: 3 epochs with no improvement

**Why use it?**
- Responds to actual training dynamics, not a fixed schedule
- Reduces LR only when needed, preserving optimization when progress is being made
- Patient enough to avoid false reductions from temporary loss spikes

**The advantage:**
- Maximizes training flexibility
- Often achieves the best validation accuracy
- Can handle variations in convergence speed

### 3. **CosineAnnealingLR** — The Modern Approach

**How it works:**
```
LR = eta_min + 0.5 × (LR_initial - eta_min) × (1 + cos(π × current_epoch / T_max))
```

This creates a smooth cosine curve that gracefully decays the learning rate.

In our implementation:
- T_max: 30 epochs (full training duration)
- Minimum LR: 1e-6
- Creates a smooth warm-down effect

**Why use it?**
- Theoretically motivated by optimization theory
- Provides smooth, continuous decay rather than discrete steps
- Works well as a default choice without much tuning

**The benefit:**
- Eliminates the need for patience parameters
- Encourages exploration early, exploitation late
- Excellent for reproducibility

---

## 🏗️ Architecture & Setup

### Model Configuration

We use **ResNet18** — a lightweight residual network with 18 layers. Why ResNet?
- Fast training iterations for comparison
- Sufficient complexity for CIFAR-10
- Well-established baseline

**Key modifications:**
- Base: `torchvision.models.resnet18` with pretrained ImageNet weights
- Replace final FC layer: `nn.Linear(512, 10)` for CIFAR-10 classes
- Transfer learning: Fine-tune pretrained weights

### Dataset & Training Setup

**CIFAR-10:**
- 60,000 images (32×32 pixels)
- 10 object classes
- Train/Val split: 50,000 / 5,000 (we further split test into 50/50)
- Normalization: Mean=(0.5, 0.5, 0.5), Std=(0.5, 0.5, 0.5)

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Epochs: 30
- Batch size: 32
- Device: GPU (CUDA/MPS) if available, else CPU

---

## 📊 Results & Analysis

### Peak Performance Comparison

| Scheduler | Peak Accuracy | Epoch | Behavior |
|-----------|---------------|-------|----------|
| **StepLR** | 82.6% | Epoch 20 | Step-like jumps |
| **ReduceLROnPlateau** | 83.0% | Epoch 28 | Smooth, adaptive |
| **CosineAnnealingLR** | 82.1% | Epoch 22-27 | Smooth decay |

### Key Observations

#### StepLR: Predictable but Rigid
```
Epoch 10: LR reduced from 0.001 → 0.0001
Epoch 20: LR reduced from 0.0001 → 0.00001
```
- **Strength**: Very predictable, useful for reproducing results
- **Weakness**: The aggressive reduction at epoch 10 causes accuracy to dip, then recover
- **Verdict**: Works well but requires manual tuning of step_size and gamma

#### ReduceLROnPlateau: Best Overall Performance
```
Patience window tracks: "Has validation loss improved in last 3 epochs?"
```
- **Strength**: Achieved the highest peak accuracy (83.0%) by being responsive to actual progress
- **Weakness**: Requires monitoring validation loss, slightly more computational overhead
- **Verdict**: Best choice for maximizing performance when you can afford validation checks

#### CosineAnnealingLR: Elegant Simplicity
```
LR smoothly decays along a cosine curve
No threshold parameters, pure mathematical elegance
```
- **Strength**: No patience parameters to tune, reproducible, smooth training curve
- **Weakness**: Fixed schedule means some epochs may have too-high or too-low LR
- **Verdict**: Excellent default choice, minimal hyperparameter tuning needed

---

## 💻 Running the Code

### Prerequisites

```bash
pip install torch torchvision
```

### Basic Workflow

1. **Data Loading**: CIFAR-10 is downloaded automatically on first run
2. **Model Initialization**: ResNet18 with pretrained ImageNet weights
3. **Scheduler Configuration**: Choose one of three schedulers
4. **Training Loop**: 30 epochs with validation and checkpointing

### Key Code Snippets

**StepLR Setup:**
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 
    step_size=10, 
    gamma=0.1
)
```

**ReduceLROnPlateau Setup:**
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=3,
    verbose=True
)
# Called with validation loss: scheduler.step(avg_val_loss)
```

**CosineAnnealingLR Setup:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=30,
    eta_min=1e-6
)
```

---

## 📁 Project Structure

```
├── learning_rate_scheduler_comparison.ipynb
│   └── Demonstrates StepLR scheduler
├── learning-rate-scheduler-comparison_cosine_and_reduce.ipynb
│   └── Demonstrates ReduceLROnPlateau and CosineAnnealingLR
├── checkpoints/
│   ├── best_model.pth
│   └── epoch_1.pth to epoch_30.pth
├── data/
│   └── CIFAR-10 dataset (auto-downloaded)
└── README.md (this file)
```

---

## 🔍 When to Use Each Scheduler

### Use **StepLR** if:
- You're training on a dataset where you know the optimal learning rate decay schedule
- You want maximum reproducibility and minimal hyperparameter tuning
- You're on a limited computational budget (slight performance cost for simplicity)

### Use **ReduceLROnPlateau** if:
- You want the best validation accuracy possible
- You have computational resources for validation checks
- You're fine-tuning on new datasets and want adaptive behavior
- You want to avoid manually setting decay schedules

### Use **CosineAnnealingLR** if:
- You want a mathematically principled approach
- You prefer smooth learning curves over step functions
- You want a good default that works across many problems
- You're training from scratch and want modern best practices

---

## 🎓 Key Learnings

1. **Learning rate scheduling matters**: All three approaches improved upon a fixed learning rate
2. **Adaptive beats rigid**: ReduceLROnPlateau's 0.4% advantage comes from responding to real training dynamics
3. **No one-size-fits-all**: The "best" scheduler depends on your specific problem and constraints
4. **Modern methods shine**: CosineAnnealingLR requires minimal tuning despite being relatively new
5. **Checkpointing is essential**: Saving models at each epoch allows recovery of the best weights

---

## 📈 Reproducibility

All experiments were conducted with:
- PyTorch version: Latest stable
- Device: GPU (CUDA/MPS) with automatic fallback to CPU
- Random seeds: Not explicitly fixed (results may vary slightly)

To ensure reproducibility across runs:
```python
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
```

---

## 🚀 Future Enhancements

- WarmupLR: Gradually increasing LR at the start for better convergence
- Cyclic Learning Rates: Multiple LR cycles within one training run
- Learning rate finder: Automatic determination of optimal initial LR
- Multi-step scheduler comparisons with different CIFAR architectures

---

## 📝 License

This project is licensed under the MIT License — see the LICENSE file for details.

---

## 🤝 Contributing

Found an issue or want to improve the comparison? Fork the repository and submit a pull request!

---

**Happy training! 🎉** Remember: the learning rate scheduler you choose can be the difference between good and great results. Experiment, measure, and adapt.
• Behavior: Smooth LR decay, graceful convergence but plateaued earlier.
---
🔎 Observations
• StepLR is predictable but rigid.
• ReduceLROnPlateau adapts best to validation loss, yielding the highest accuracy.
• CosineAnnealingLR provides smooth convergence but didn’t surpass ReduceLROnPlateau in this run.
---
📌 References
• PyTorch ResNet18
• CIFAR‑10 Dataset
• Learning Rate Schedulers in PyTorch
---
🏆 Conclusion
This project highlights how different learning rate schedulers affect training dynamics and final accuracy.  
ReduceLROnPlateau gave the best results in this experiment, but each scheduler has trade‑offs depending on dataset and model.

Medium Blog post

https://medium.com/@allam.satyanarayana/learning-rate-schedulers-in-deep-learning-a-practical-comparison-with-cifar-10-066fdececfc8