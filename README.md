# Structured State Matrix Architecture (SSMA)

**SSMA** is a novel neural network framework designed to overcome the quadratic complexity of Transformers by replacing dense self-attention with a combination of dynamic sparse state transitions, low-rank factorized interactions, and a hierarchical memory module based on a Linear Recurrent Unit (LRU). SSMA also integrates quantization-aware training via ternary weight constraints to achieve efficient inference on modern hardware.

This repository contains the full source code for the SSMA library, which is fully pip‑installable. Users can build their own transformer‑like models by importing and combining SSMA components—similar to how they work with the HuggingFace Transformers library.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
  - [Sparse State Transitions](#sparse-state-transitions)
  - [Low-Rank Factorized Interactions](#low-rank-factorized-interactions)
  - [Hierarchical Memory via LRU](#hierarchical-memory-via-lru)
  - [Quantization-Aware Training](#quantization-aware-training)
- [Theoretical Foundations](#theoretical-foundations)
  - [Approximation Power](#approximation-power)
  - [Memory Stability](#memory-stability)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Importing and Building Custom Models](#importing-and-building-custom-models)
  - [Training](#training)
  - [Evaluation](#evaluation)
- [Multi-Phase Training Process](#multi-phase-training-process)
- [Hyperparameter Considerations](#hyperparameter-considerations)
- [Benchmarks and Results](#benchmarks-and-results)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Full Paper](#full-paper)
- [License](#license)
- [Contact](#contact)

---

## Overview

SSMA rethinks sequence modeling by:
- **Replacing** dense self-attention with **dynamic sparse state transitions** to focus computation only on the most important features.
- **Employing** **low-rank factorized interactions** to efficiently capture token dependencies.
- **Integrating** a **hierarchical memory module** via a Linear Recurrent Unit (LRU) that retains long-term context in a fixed-size memory bank.
- **Incorporating** **quantization-aware training** (ternary weight constraints) to produce lightweight models optimized for modern hardware.

By reducing the computational complexity from \(O(n^2d)\) to \(O(ndr)\) (with \(r \ll d\)) and ensuring constant memory usage, SSMA enables efficient and scalable modeling for very long sequences.

---

## Key Features

- **Linear Complexity:** Achieves \(O(ndr)\) complexity via sparse state transitions and low-rank factorization.
- **Constant Memory Usage:** A fixed-size LRU-based memory module ensures that memory usage does not scale with input length.
- **Hybrid Flexibility:** Easily integrate SSMA layers into your custom models or use hybrid layers that toggle between standard attention and SSMA.
- **Quantization-Aware Training:** Supports ternary quantization for improved inference efficiency.
- **Modular Design:** Build transformer‑like models by combining SSMA’s modular components.
- **Multi-Phase Training:** Supports a training pipeline with dense training, lottery ticket pruning, and dynamic sparsity for optimal performance.

---

## Architecture Overview

### Sparse State Transitions

SSMA updates a fixed-size state matrix \(S \in \mathbb{R}^{d \times m}\) with a selective mechanism:
$$
S^{(t)} = \operatorname{Top\text{-}k}\Bigl(\sigma\Bigl(S^{(t-1)} W_{\text{state}} + X^{(t)} W_{\text{in}}\Bigr)\Bigr)
$$
- **\(X^{(t)}\)**: Input token embedding at time \(t\).
- **\(W_{\text{state}}\)**: Block-diagonal weight matrix with learned sparsity masks.
- **\(W_{\text{in}}\)**: Projects the input into the state space.
- **\(\sigma\)**: Non-linear activation (e.g., ReLU).
- **\(\operatorname{Top\text{-}k}\)**: Retains the top \(k\) activations, enforcing sparsity.

### Low-Rank Factorized Interactions

SSMA employs low-rank projections to efficiently capture token interactions:
$$
Y = X + \operatorname{FFN}\Bigl(U(X) \cdot V(X)^T\Bigr)
$$
- **\(U, V \in \mathbb{R}^{d \times r}\)**: Learnable projection matrices where \(r \ll d\).
- **\(\operatorname{FFN}\)**: A feed-forward network that further processes the combined interactions.

### Hierarchical Memory via LRU

To capture long-term dependencies, SSMA maintains a fixed-size memory \(M \in \mathbb{R}^{m \times d}\) updated as:
$$
M^{(t)} = \gamma M^{(t-1)} + (1 - \gamma) S^{(t)}
$$
- **\(\gamma\)**: A learned decay factor that balances past memory with new state updates.

### Quantization-Aware Training

SSMA leverages quantization-aware training to reduce model size and increase efficiency:
$$
\mathcal{L}_{\text{quant}} = \|W_{\text{state}} - \operatorname{sign}(W_{\text{state}})\|^2
$$
- **STE (Straight-Through Estimator):** Used to allow gradient flow through non-differentiable quantization operations.
- **Regularization:** Encourages weights to adopt ternary values (\(\{-1,0,+1\}\)).

---

## Theoretical Foundations

### Approximation Power
For any attention matrix \(A\), there exist matrices \(U, V \in \mathbb{R}^{d \times r}\) (with \(r = O(\frac{\log d}{\epsilon^2})\)) such that:
$$
\|A - UV^T\|_F \le \epsilon \|A\|_F.
$$
This result shows that low-rank factorization in SSMA can closely approximate full self-attention.

### Memory Stability
The LRU update:
$$
M^{(t)} = \gamma M^{(t-1)} + (1 - \gamma) S^{(t)}
$$
ensures that the gradients with respect to the initial memory decay exponentially:
$$
\Bigl\|\frac{\partial \mathcal{L}}{\partial M^{(0)}}\Bigr\| \le C \, \gamma^t,
$$
ensuring stable training over very long sequences.

_Detailed proofs are provided in the Appendix of the full paper._

---

## Project Structure

```
SSMA/
├── .github/
│   └── workflows/
│       └── publish.yml         # GitHub Actions for CI/CD and publishing to PyPI
├── docs/                      # Extended documentation
│   ├── index.md
│   ├── architecture.md
│   └── usage.md
├── requirements.txt           # Python package dependencies
├── setup.py                   # Setup script for packaging
├── README.md                  # This file
├── LICENSE                    # License file (MIT License)
├── SSMA.pdf                   # Full technical paper
├── ssma/                      # Main Python package
│   ├── __init__.py
│   ├── model.py               # SSMA model definition
│   ├── config/                # Default configuration files
│   │   ├── default_config.yaml
│   │   └── hyperparameters.yaml
│   ├── layers/                # SSMA layer implementations
│   │   ├── __init__.py
│   │   ├── ssma_layer.py
│   │   └── hybrid_layer.py
│   ├── training/              # Training scripts and utilities
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── dataset.py
│   │   └── utils.py
│   ├── evaluation/            # Evaluation scripts
│   │   ├── __init__.py
│   │   └── evaluate.py
│   └── quantization/          # Quantization functions
│       ├── __init__.py
│       └── quantize.py
├── scripts/                   # Helper shell scripts
│   ├── run_train.sh
│   └── run_eval.sh
└── tests/                     # Unit tests
    ├── __init__.py
    ├── test_model.py
    ├── test_layers.py
    └── test_quantization.py
```

---

## Installation

### Via PyPI
After publishing on PyPI, install the package with:
```bash
pip install ssma
```

### From Source
Clone the repository and install locally:
```bash
git clone https://github.com/svector-corporation/SSMA.git
cd SSMA
pip install .
```

---

## Usage

### Importing and Building Custom Models
You can build your own transformer-like models using the SSMA components. For example:
```python
from ssma.model import SSMA_Model

# Initialize the SSMA model
model = SSMA_Model(
    d_model=512,
    num_layers=4,
    r=64,
    m=256,
    top_k=32,
    vocab_size=10000,
    max_seq_len=512
)
```
You can integrate the SSMA model into your training pipelines or use its components to build custom architectures.

### Training
A comprehensive training pipeline is provided in the `ssma/training/` module. To train the model on your dataset:
```bash
bash scripts/run_train.sh
```
This script uses a dummy dataset (`RandomTextDataset`) and includes multi-phase training (dense training, lottery ticket pruning, dynamic sparsity). Modify `ssma/training/dataset.py` and `ssma/training/utils.py` for your specific data and training strategies.

### Evaluation
Evaluate your trained model with:
```bash
bash scripts/run_eval.sh
```
The evaluation script (`ssma/evaluation/evaluate.py`) loads a saved checkpoint and computes metrics (e.g., perplexity) on a test dataset.

---

## Multi-Phase Training Process

SSMA’s training strategy involves three distinct phases:

1. **Dense Training:**  
   Train all weights (e.g., \(W_{\text{state}}\)) in a dense manner to capture full connectivity.

2. **Lottery Ticket Pruning:**  
   Prune non-critical weights based on magnitude to discover a sparse subnetwork while retaining initial weights.

3. **Dynamic Sparsity:**  
   Replace fixed masks with adaptive Gumbel-Softmax gates to enforce a top-\(k\) sparsity pattern dynamically during training.

This process enhances efficiency and ensures that the model remains both expressive and computationally efficient.

---

## Hyperparameter Considerations

Key hyperparameters include:
- **Top-\(k\) Threshold:** Controls the number of activations retained in the state update.
- **Decay Factor (\(\gamma\)):** Determines how much historical memory is retained versus new information.
- **Quantization Settings:** STE temperature and regularization strength in the quantization loss.

These parameters are defined in the configuration files located in `ssma/config/`. Fine-tuning them is crucial for achieving optimal performance on different tasks.

---

## Benchmarks and Results

SSMA has been benchmarked against standard Transformers and other linear-complexity architectures like Mamba and RetNet. For example, on WikiText-103:

| **Model**      | **PPL** | **Memory (8k)** | **Throughput (t/s)** |
|----------------|---------|-----------------|----------------------|
| Transformer    | 18.7    | 24.5 GB         | 1.2k                 |
| Mamba          | 19.1    | 12.1 GB         | 3.8k                 |
| RetNet         | 18.9    | 10.5 GB         | 4.1k                 |
| **SSMA**       | **17.9**| **8.4 GB**      | **4.5k**             |

*Observations:*  
- **Efficiency:** SSMA uses constant memory and achieves higher throughput due to its linear complexity.
- **Quality:** Lower perplexity indicates better language modeling performance.
- **Scalability:** SSMA can handle very long sequences (up to 1M tokens) with a fixed memory footprint.

For detailed experimental results, please refer to the full paper.

---

## Documentation

Further documentation is available in the `docs/` directory:
- **`docs/index.md`:** Overview and table of contents.
- **`docs/architecture.md`:** Detailed explanation of the SSMA architecture.
- **`docs/usage.md`:** Usage examples and API reference.

---

## Contributing

We welcome contributions from the community. If you would like to contribute:
- Fork the repository and create your feature branch.
- Write tests and update documentation as necessary.
- Submit a pull request with a detailed description of your changes.
See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

---

## Full Paper

For the complete technical paper with rigorous proofs, extended experiments, and comprehensive discussions, please visit:  
[Full Paper](SAMA.pdf)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For questions, support, or further discussion, please contact us at [research@svector.co.in](mailto:research@svector.co.in).

---
