# Cross-Sectional Equity Alpha Research: MLP, LSTM & Graph Neural Networks

**Predicting relative equity returns using deep learning, cross-sectional ranking, and long/short portfolio research.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C) ![Research](https://img.shields.io/badge/Focus-Quantitative%20Research-263238)

## Overview

Can neural networks extract a **cross-sectional equity alpha signal** from historical price and volume data? This project compares three distinct modeling ideas:

- **Multilayer Perceptron (MLP):** learns nonlinear interactions among engineered factors.
- **Long Short-Term Memory (LSTM):** learns temporal patterns from sequences of daily features.
- **Graph Isomorphism Network (GIN/GNN):** models cross-stock interactions through sector and rolling-correlation links.

The target is **relative performance across stocks**, not a precise prediction of the absolute price of an individual stock. Evaluation emphasizes daily Spearman rank information coefficient (IC), quintile return spreads, long/short returns, walk-forward testing, and methods that account for overlapping forward returns.

## Repository structure

```text
.
├── README.md
├── mlp.ipynb          # Engineered-factor cross-sectional MLP
├── lstm.ipynb         # Sequence-based LSTM
├── gnn.ipynb          # Weighted sector/correlation-graph GIN
├── requirements.txt
└── .gitignore
```

The notebooks are independently executable research experiments. Their **universes, sample periods, label constructions, and portfolio implementations differ**, so their raw performance numbers are not a controlled head-to-head benchmark.

## Research methodology

```text
Yahoo Finance adjusted OHLCV
            |
     Feature engineering
            |
  Cross-sectional or rolling
       normalization
            |
    Date-based splits
            |
      MLP / LSTM / GIN
            |
   Cross-sectional scores
            |
  Spearman IC + quintiles
            |
 Long/short portfolio proxy
            |
 Overlap-aware diagnostics
```

### MLP: nonlinear factor interactions

**Notebook:** [`mlp.ipynb`](mlp.ipynb)

- **Universe:** 65 large US-listed equities across major sectors.
- **Downloaded history:** January 2010 through December 2025.
- **Engineered features:** 35 indicators, including momentum and reversal, realized volatility, price-to-moving-average ratios, RSI, candle geometry, 52-week ranges, and volume ratios.
- **Preprocessing:** feature z-scores across stocks for each date, clipped at ±5.
- **Target:** centered cross-sectional percentile rank of the subsequent **5-trading-day** return.
- **Architecture:** `35 → 128 → 64 → 32 → 1`, with batch normalization, ReLU, and dropout.
- **Training:** AdamW and validation-IC early stopping; predictions averaged over **five random seeds**.
- **Experiment designs:** fixed chronological split and optional annual walk-forward refits in 2023–2025.

### LSTM: learning temporal representations

**Notebook:** [`lstm.ipynb`](lstm.ipynb)

- **Universe and downloaded period:** same 65-stock selection and 2010–2025 range as the MLP.
- **Input:** eight relatively primitive daily signals, arranged into **30-trading-day sequences** for each ticker.
- **Architecture:** two LSTM layers, hidden dimension 64, followed by a prediction head.
- **Target:** centered rank of the 5-day forward return.
- **Training:** five-seed ensemble with validation-IC-based checkpoint selection.
- **Evaluation:** fixed time split; a separate optional yearly walk-forward function is included.

The LSTM tests whether temporal dependencies can replace part of the extensive manual feature engineering used by the MLP.

### GNN: modeling cross-stock relationships

**Notebook:** [`gnn.ipynb`](gnn.ipynb)

- **Universe:** a separate approximately **118-name** sector-labeled equity selection.
- **Downloaded period:** January 2010 through the end of 2024.
- **Input:** 20 engineered features normalized with trailing time-series statistics.
- **Graph:** sector edges combined with positive, rolling **60-day** return-correlation k-nearest-neighbor edges (`k=6`).
- **Target:** mean of sector-demeaned, cross-sectionally ranked **5-day and 10-day** forward returns.
- **Architecture:** a three-layer Graph Isomorphism Network with hidden dimension 192.
- **Training:** ranking-oriented listwise IC loss; three seeds and a validation-selected window of model-weight averaging.
- **Validation:** five annual walk-forward OOS folds (**2020–2024**).
- **Portfolio simulation:** five-day exponential smoothing of scores, sector-neutral rank-proportional exposures, and optional lagged-realized-volatility scaling toward **12% annualized volatility**.

## Saved notebook results

The following figures are **reported by the uploaded notebooks**. They have not been independently reproduced or audited as trade-level backtests.

| Experiment | Out-of-sample dates | Mean daily rank IC | Reported non-overlap Sharpe |
| :-- | :-- | --: | --: |
| MLP, frozen model | 2023–2025 | **+0.0210** | **1.10** |
| MLP, annual walk-forward retraining | 2023–2025 | **+0.0218** | **1.52** |
| LSTM, frozen model | 2023–2025 | **+0.0225** | **1.18** |
| GIN v5, five walk-forward folds, volatility-targeted | 2020–2024 | **~+0.011** | **1.00** |

**Reading this table:** the GNN changes data universe, target, test window, portfolio weighting, and risk scaling. Its IC and Sharpe therefore cannot establish whether GNN is intrinsically better or worse than MLP/LSTM. The GNN notebook also reports **0.54 HAC-adjusted Sharpe** for its volatility-targeted series, illustrating the effect of autocorrelation adjustments.

For the MLP and LSTM fixed experiments, the original test window covers 2023–2025, with 2021–2022 reserved for validation. The notebooks additionally report quintile spreads, model seed sensitivity, drawdowns, and portfolio diagnostics.

## Evaluation details

- **Daily Spearman IC:** rank correlation between each date's predictions and subsequent returns.
- **IC information ratio:** mean daily IC divided by standard deviation of daily IC.
- **Quintile spread:** future return of top-ranked quintile minus bottom-ranked quintile.
- **Long/short portfolios:** MLP/LSTM use top versus bottom 20%; GNN uses sector-neutral rank-based allocations.
- **Overlapping-label diagnostics:** five-day forward returns measured every day share price movements. The notebooks examine naïve, Newey–West/HAC-adjusted, and non-overlapping subsampled Sharpe estimates.
- **Walk-forward evaluation:** retraining using prior observations rather than continually evaluating one fixed historical model.

### Important limitations

1. **Survivorship/selection bias:** the predefined equity lists are not point-in-time constituents. This is a research universe, not a historically investable index reconstruction.
2. **Potential boundary leakage:** forward returns are constructed before some chronological splits; samples whose forward horizons cross a boundary should be purged for a stricter evaluation.
3. **Overlap and portfolio accounting:** sampling five-day forward returns daily does not itself define a self-financing executable portfolio; HAC and subsampling are useful but not complete solutions.
4. **Transaction costs omitted:** reported returns exclude commissions, slippage, market impact, financing and borrow costs, short availability, and capacity constraints.
5. **Not directly comparable:** training regimes, date windows, universes, labels, features, and weighting methods are not held constant across all three notebooks.
6. **Reproducibility:** Yahoo Finance data can be revised and package versions change. Recorded outputs might not match a fresh run.

A production-grade follow-up would use point-in-time security membership, purged splits, truly untouched test periods, explicit trade/holdings accounting, realistic transaction costs, and equal-universe/equal-period baselines.

## Getting started

```bash
git clone https://github.com/rishitbelani17/MLP-GNN-LSTM-for-Cross-Section-Equity-Long-Short.git
cd MLP-GNN-LSTM-for-Cross-Section-Equity-Long-Short
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Open `mlp.ipynb`, `lstm.ipynb`, or `gnn.ipynb` and run cells in order. Google Colab is also supported; the MLP/LSTM notebooks include Colab-style `!pip install` cells. A CUDA GPU speeds up training, particularly the GNN and walk-forward experiments.

The notebooks download their own market data and generate local predictions, model checkpoints, charts and summaries. Those generated artifacts are excluded by `.gitignore`.

## Technology

Python · PyTorch · pandas · NumPy · SciPy · scikit-learn · yfinance · Matplotlib · Jupyter

## Author

**Rishit Belani** | Columbia University, MS in Data Science

[GitHub](https://github.com/rishitbelani17) · [Portfolio](https://rishit-belani.vercel.app/)

*For research and educational purposes only. Not investment advice.*
