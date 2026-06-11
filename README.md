# DRAFNet for Trade Union User Intent Recognition

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/PyTorch-1.13+-EE4C2C.svg" alt="PyTorch 1.13+">
  <img src="https://img.shields.io/badge/Code-MIT-yellow.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/Dataset-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="CC BY-NC-SA 4.0">
  <img src="https://img.shields.io/badge/Test%20Accuracy-94.52%25-brightgreen.svg" alt="Test Accuracy 94.52%">
  <img src="https://img.shields.io/badge/Macro--F1-93.88%25-brightgreen.svg" alt="Macro-F1 93.88%">
</p>

<p align="center">
  <b>Official implementation of:</b><br>
  <i>"Research on Precise User Intent Recognition Algorithm for Power Grid Enterprise Trade Union Domain Based on Deep Residual Attention Network"</i>
</p>

---

## 📌 What is DRAFNet?

**DRAFNet** (Deep Residual Attention Fusion Network) is a four-stage neural architecture for Chinese intent recognition in vertical domains. It combines:

1. **Domain-lexicon-enhanced** input encoding ([DOM] markers for 198 trade-union terms)
2. **BERT-Base-Chinese** as the contextual encoder
3. **Four convolutional residual blocks** with **multi-level weighted fusion**
4. **Multi-level attention fusion** (local window + global multi-head + gated combination)
5. **Joint optimization** of cross-entropy and Focal Loss for class imbalance

On the WR27-TradeUnionIntent test set, DRAFNet achieves **94.52% accuracy / 93.88% Macro-F1**, statistically outperforming the strongest pretrained baseline (ERNIE 3.0) by 2.21 / 2.25 percentage points (paired t-test, p < 0.05).

<p align="center">
  <img src="docs/architecture.png" alt="DRAFNet architecture" width="850"/><br>
  <i>Figure 1. DRAFNet four-stage architecture.</i>
</p>

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/DRAFNet-trade-union.git
cd DRAFNet-trade-union

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the WR27-TradeUnionIntent dataset (see "Releases")
#    Then extract its data/*.json into data_files/

# 4. Run the full experimental pipeline
python -m scripts.run_all
```

This produces all result tables (`results/tables/`) and the 14 paper figures (`results/figures/`) in one pass.

---

## 📦 Releases

Two release packages are provided under the [Releases](../../releases) tab:

| Package | Size | Contents | License |
| :-- | --: | :-- | :-- |
| **DRAFNet-trade-union-Code-v1.0.zip** | ~4 MB | Complete source code, configs, pre-generated results, 14 figures | MIT |
| **WR27-TradeUnionIntent-Dataset-v1.0.zip** | ~3 MB | 5,831 original + 12,486 augmented samples, train/val/test splits, annotation guidelines, de-identification specification | CC BY-NC-SA 4.0 |

The two packages are designed to work together: extract the code package as your working directory, then copy the dataset's JSON files into `data_files/`.

---

## 📊 Headline Results

### Comparison with baselines (Table 14, test set, 5 random seeds)

| Model | Accuracy (%) | Macro-F1 (%) |
| :-- | --: | --: |
| TextCNN | 85.17 ± 0.43 | 84.22 ± 0.47 |
| BiLSTM | 86.54 ± 0.38 | 85.64 ± 0.40 |
| BiLSTM-Att | 87.92 ± 0.36 | 87.02 ± 0.38 |
| BERT-Base | 91.28 ± 0.22 | 90.41 ± 0.24 |
| BERT-BiLSTM | 92.03 ± 0.21 | 91.30 ± 0.23 |
| BERT-TextCNN | 91.76 ± 0.23 | 90.98 ± 0.25 |
| RoBERTa | 91.85 ± 0.20 | 91.16 ± 0.21 |
| ERNIE 3.0 | 92.31 ± 0.19 | 91.63 ± 0.20 |
| **DRAFNet (Ours)** | **94.52 ± 0.18** | **93.88 ± 0.21** |

DRAFNet shows the largest improvements on low-frequency categories — **+4.39 pp F1** on Labor Disputes (C5) and **+4.11 pp F1** on Psychological Assistance (C3) over ERNIE 3.0.

### Robustness against perturbations (Table 19, Macro-F1 %)

| Setting | BERT-Base | ERNIE 3.0 | **DRAFNet** |
| :-- | --: | --: | --: |
| Clean test set | 90.41 | 91.63 | **93.88** |
| 10% ASR errors | 82.16 | 84.27 | **88.92** |
| Cantonese rewrite | 78.94 | 81.05 | **85.73** |
| Hakka rewrite | 76.32 | 78.61 | **83.41** |

### Compression variants (Table 22)

| Variant | Macro-F1 | Parameters | CPU Latency |
| :-- | --: | --: | --: |
| Original | 93.88% | 116.6 M | 89.7 ms |
| Distilled (4-layer student) | 92.74% | 66.4 M | 47.3 ms |
| INT8 quantized | 93.36% | 29.3 MB | 51.8 ms |
| **Distilled + INT8** | **92.15%** | **16.7 MB** | **28.6 ms** |

The fully compressed edge variant retains 92.15% Macro-F1 with **86% storage reduction** and **68% lower CPU latency**.

---

## 📁 Repository Structure

```
DRAFNet-trade-union/
├── README.md                    # ← This file
├── LICENSE                      # MIT (code)
├── requirements.txt
│
├── configs/                     # Hyperparameter configurations
│   ├── drafnet.yaml             #   DRAFNet (Tables 9, 10, 11)
│   └── baselines.yaml           #   8 baselines (Table 13)
│
├── data_module/                 # Data loading & preprocessing
│   ├── loader.py                #   Stratified split-before-augment (Section 3.2)
│   ├── domain_lexicon.py        #   198-term trade-union lexicon
│   └── tokenizer_utils.py       #   Domain-enhanced BERT tokenizer
│
├── models/                      # Model implementations
│   ├── drafnet.py               #   Complete DRAFNet
│   ├── residual_block.py        #   Residual block (Eq. 6)
│   ├── attention_fusion.py      #   Attention modules (Eqs. 7–11)
│   ├── losses.py                #   Joint CE + Focal loss (Eqs. 14–16)
│   └── baselines.py             #   8 baseline models
│
├── training/                    # Training pipeline
│   └── trainer.py               #   AdamW + layer-wise LR + early stopping
│
├── evaluation/                  # Evaluation utilities
│   ├── metrics.py               #   Accuracy / Macro-P/R/F1 (Eqs. 17–20)
│   └── robustness.py            #   ASR / dialect / OOD evaluation
│
├── visualization/               # Paper figures (≥500 DPI, IEEE style)
│   ├── figure_data.py           #   Figs 6, 7, 8
│   ├── figure_arch.py           #   Figs 9, 10, 11, 12
│   ├── figure_results.py        #   Figs 13, 14, 15, 18, 20
│   └── figure_training.py       #   Figs 16, 19
│
├── scripts/                     # Experiment runners
│   ├── run_main.py              #   Main experiment (Table 14)
│   ├── run_ablation.py          #   Ablation (Table 16)
│   ├── run_robustness.py        #   Robustness (Table 19)
│   ├── run_compression.py       #   Compression (Tables 21, 22)
│   ├── generate_figures.py      #   All 14 paper figures
│   └── run_all.py               #   One-click pipeline
│
├── data_files/                  # ← Place dataset JSON files here
│   └── README.md
│
└── results/                     # Pre-generated outputs (committed for inspection)
    ├── tables/                  #   7 result JSON files
    └── figures/                 #   14 paper figures
```

---

## 🧪 Reproducing Specific Experiments

```bash
# Main experiment (Table 14)
python -m scripts.run_main

# Ablation study (Table 16)
python -m scripts.run_ablation

# Robustness evaluation (Table 19)
python -m scripts.run_robustness

# Model compression (Tables 21, 22)
python -m scripts.run_compression

# Regenerate all 14 figures
python -m scripts.generate_figures
```

Each script writes its outputs to `results/tables/` and `results/figures/`.

---

## ⚙️ Key Hyperparameters (Table 11)

| Parameter | Setting |
| :-- | --: |
| Optimizer | AdamW |
| Initial learning rate | 2 × 10⁻⁵ |
| BERT layer-wise LR decay | 0.95 |
| Weight decay | 0.01 |
| Batch size | 32 |
| Epochs | 30 |
| LR schedule | Linear warmup + cosine annealing |
| Warmup ratio | 10% |
| Joint loss λ | 0.6 |
| Focal Loss γ | 2 |
| Early-stopping patience | 5 (on Macro-F1) |
| Gradient clipping | 1.0 |
| Random seeds | {42, 123, 2024, 7890, 31415} |

---

## 🗂️ Dataset

The accompanying **WR27-TradeUnionIntent** dataset contains:

- **5,831 original samples** collected from two trade union service channels (online platform + 12351 hotline) over 2021–2023
- **12,486 augmented samples** produced by synonym replacement, back-translation, and noise injection
- **7 intent categories**: Rights Protection, Welfare Policies, Psychological Assistance, Skills Training, Labor Disputes, Activity Information, General Inquiries
- **Inter-annotator agreement**: Cohen's κ = 0.885
- **De-identification compliant** with GB/T 35273-2020, GB/T 37964-2019, GB/T 42460-2023, and the PRC Personal Information Protection Law

See the dataset's `README.md` for full schema, loading instructions, and license terms.

---

## 📜 Citation

If you use this work, please cite:

```bibtex
@article{drafnet2026,
  title   = {Research on Precise User Intent Recognition Algorithm for Power Grid 
             Enterprise Trade Union Domain Based on Deep Residual Attention Network},
  journal = {[Journal Name]},
  year    = {2026},
  pages   = {1--16},
  doi     = {10.xxxx/xxxxx}
}
```

---

## ⚖️ License

| Resource | License |
| :-- | :-- |
| Source code | [MIT License](LICENSE) |
| WR27-TradeUnionIntent dataset | [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) |

Commercial use of the dataset is prohibited. See each `LICENSE` file for full terms.

---

## 🤝 Contributing

Issues and pull requests are welcome. For substantial changes, please open an issue first to discuss the proposed direction.

---

## 📮 Contact

For questions about the code, dataset, or paper, please open a GitHub Issue or contact the corresponding author of the paper.

---

## 🙏 Acknowledgments

We thank the data collection team at Guangdong Power Grid Foshan Power Supply Bureau Trade Union for providing access to consultation records, and the annotation team for labeling 5,831 samples to a Cohen's κ of 0.885.
