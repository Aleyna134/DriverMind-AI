# DriverMind AI

An LSTM-based driver behavior classifier (Normal / Drowsy / Aggressive) trained on the
[UAH-DriveSet](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/), with a live
Streamlit dashboard that feeds simulated sensor input directly into the model.

## Project structure

```
DriverMind-AI/
├── notebooks/          # 00-74: data exploration, feature engineering, model
│                        # experiments (Random Forest, XGBoost, LightGBM, LSTM,
│                        # ResNet50/video fusion) leading up to the final LSTM model
├── src/                 # Core library code
│   ├── config.py                # feature columns, sequence length, paths
│   ├── predictor.py              # hybrid rule + LSTM prediction pipeline
│   ├── lstm_model.py             # LSTM classifier architecture
│   ├── data_loader.py, preprocessor.py, trainer.py
│   └── ...                       # video/fusion model experiments
├── models/
│   └── lstm_window160_dropout02_macrof1_07358.pth   # trained LSTM checkpoint
├── dashboard/            # Streamlit live-monitoring dashboard
│   ├── app.py
│   ├── components/       # header, metrics, trend chart, risk factors, feature controls
│   └── styles/
└── datasets/             # not tracked in git (see below)
```

## Datasets

`datasets/` is excluded from this repository (raw UAH-DriveSet recordings, optical-flow
caches, and processed `.npz` feature dumps total several hundred GB). To reproduce:

1. Download the [UAH-DriveSet](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/)
   and place it under `datasets/raw/UAH-DRIVESET-v1/`.
2. Run the notebooks in order (00 → 74) to regenerate `datasets/processed/`, including
   `feature_defaults.json` and `lstm_feature_scaler.pkl`, which the dashboard depends on.

## Dashboard

The dashboard builds the LSTM's 13-feature input window directly from 4 user-controlled
sensor inputs (speed, front distance, lane offset, relative speed) plus dataset-mean
defaults for the remaining features. A rule-based layer sits on top of the raw model output
so that sudden braking/acceleration, sharp steering, tailgating, and sustained lane drift are
reflected consistently in the displayed risk score, class label, and warning chips.

Run it locally:

```bash
pip install -r requirements.txt   # streamlit, torch, numpy, pandas, scikit-learn, joblib
streamlit run dashboard/app.py
```

## Model

- Architecture: LSTM classifier, `SEQUENCE_LENGTH=160`, dropout 0.2
- Classes: `Normal`, `Drowsy`, `Aggressive`
- Features are z-score standardized before inference (`datasets/processed/lstm_feature_scaler.pkl`)

## License

MIT License
