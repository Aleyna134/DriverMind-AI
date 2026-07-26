# DriverMind AI

An intelligent driver monitoring and risk prediction system powered by deep learning.

DriverMind AI classifies driver behavior in real time — **Normal**, **Drowsy**, or **Aggressive** —
from vehicle sensor signals, using an LSTM sequence classifier trained on the
[UAH-DriveSet](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/). A hybrid
rule-plus-model layer sits on top of the raw classifier output, and a Streamlit dashboard
exposes the whole pipeline as a live monitoring tool driven by simulated sensor input.

## Project Structure

```
DriverMind-AI/
├── notebooks/              # 00-74: data exploration, feature engineering, and the full
│                            # model search (Random Forest, XGBoost, LightGBM, ResNet50 /
│                            # optical-flow video models, and the final LSTM) in chronological order
├── src/                     # Core library code
│   ├── config.py                  # feature columns, sequence length, training constants
│   ├── predictor.py                # hybrid rule + LSTM inference pipeline
│   ├── lstm_model.py               # final LSTM classifier architecture
│   ├── data_loader.py, preprocessor.py, trainer.py
│   └── ...                         # video/fusion model code from the exploratory phase
├── models/
│   └── lstm_window160_dropout02_macrof1_07358.pth   # final trained LSTM checkpoint
├── dashboard/                # Streamlit live-monitoring dashboard
│   ├── app.py
│   ├── components/                 # header, metrics, trend chart, risk factors, recommendations, sensor overview
│   └── styles/
└── datasets/                 # not tracked in git (see Installation)
```

## Model Architecture

The deployed model is a single-layer-stacked LSTM sequence classifier:

| Parameter | Value |
|---|---|
| Input features | 13 (accelerometer, gyroscope, GPS-derived speed/heading, lane offset, front distance, relative speed, vehicle state) |
| Sequence length | 160 timesteps |
| Hidden size | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Output classes | 3 (`Normal`, `Drowsy`, `Aggressive`) |

Input features are z-score standardized (`datasets/processed/lstm_feature_scaler.pkl`) before
being fed to the model. At inference time, `src/predictor.py` builds this 13-feature window from
4 user-controlled dashboard inputs plus dataset-mean defaults for the remaining features, and
layers rule-based checks (harsh braking/acceleration, sharp steering, tailgating, sustained lane
drift) on top of the raw softmax output so the displayed risk score, class label, and warning
chips never contradict each other.

## Results
### Model Evolution & Selection

The development process transitioned through three main phases to identify the optimal architecture:

1. **Classical Machine Learning Baselines (Notebooks 10–44)**
   - **Approach**: Extracted hand-crafted statistical features (mean, std, min, max, linear trends) over sliding windows.
   - **Models**: Evaluated Random Forest, LightGBM, and XGBoost using GroupKFold (trip-level) splits to prevent driver leakage.
   - **Result**: A tuned **XGBoost** model achieved **77.10% accuracy** (optimized with a 0.15 probability threshold, up from a **70.04%** baseline in Notebook 44). While competitive, classical models were limited because flattening windows into statistical aggregates threw away the temporal order of sensor signals (vital for identifying transitions, like a slow drowsy drift vs. an aggressive jerk).

2. **Sequence Modeling with LSTMs (Notebooks 45–49, 61–73)**
   - **Approach**: Fed raw sequence windows (timesteps $\times$ features) directly into stacked LSTMs to automatically learn time-series patterns.
   - **Window Size Sweep**: Evaluated sequence lengths to find the best balance of temporal context:
     - Window size 80 (Notebooks 67–68): Accuracy ~70.0%
     - Window size 120 (Notebook 66): Accuracy ~70.56% 
     - Window size 160 (Notebooks 69–70): **72.87% accuracy (Macro F1 0.7358)**
     - Window size 200 (Notebooks 71–72): Accuracy ~71.0%
   - **Selection**: The **160-timestep window** (representing 16 seconds of driving context at 10Hz) yielded the highest validation accuracy and was selected for the final deployment.

3. **Multimodal Video & Optical Flow (Notebooks 50–60, 64–65)**
   - **Approach**: Explored dashcam video frame embeddings (ResNet50) and dense Optical Flow CNN-LSTMs to augment sensors.
   - **Result**: Suffered from extreme overfitting (validation accuracy drop to 20–38%) due to the high dimensionality of video embeddings relative to the limited number of independent driving trips (~40). This confirmed that a sensor-only LSTM was the most robust and generalizable architecture.

### Final Model Performance

The production checkpoint (`lstm_window160_dropout02_macrof1_07358.pth`) was selected as the
best-performing configuration across a window-size and dropout sweep (see notebooks 67–73):

| Metric | Score |
|---|---|
| Accuracy | 72.87% |
| Precision (weighted avg)* | 0.73 |
| Recall (weighted avg)* | 0.74 |
| F1-Score (weighted avg)* | 0.73 |
| **Macro F1** | **0.7358** |

\* Precision/Recall/F1-Score are the weighted-average classification report from an equivalent
window-160 LSTM run in the same experiment family (notebook 73); Accuracy and Macro F1 are
reported directly for the exact saved checkpoint.

### Exploratory Approaches: Video-Based Models

Before settling on the sensor-only LSTM, the project also explored whether the UAH-DriveSet's
dashcam footage could improve or replace sensor-based classification. Frame-level embeddings were
extracted with a pretrained **ResNet50** backbone (both single-frame and short temporal
sequences) and evaluated on their own, fused with the sensor stream, and re-evaluated under a
stricter Leave-One-Group-Out (LOGO, grouped by trip) cross-validation split to test
generalization across drivers. A dense **optical-flow** representation was also computed from the
same footage and paired with a CNN-LSTM classifier under the same LOGO protocol.

None of these visual pipelines matched the sensor-only LSTM. Training accuracy routinely reached
95–99% while held-out accuracy fell to 20–38%, a clear overfitting signature driven by the small
number of independent trips (~40) relative to the dimensionality of visual embeddings; the
optical-flow model was additionally unstable across LOGO folds, with no consistent aggregate
score. Given these results, the final system relies exclusively on the sensor-based LSTM
described above.

| Approach | Best Reported Metric | Notebook(s) |
|---|---|---|
| ResNet50 + sensor fusion (LSTM) | Accuracy 31.8% · F1 33.0% (weighted) | 52 |
| ResNet50 frame features + MLP | Val. Accuracy 20.5% (train acc. 96.6% — overfit) | 53 |
| ResNet50 sequence features + LSTM (video-only) | Val. Accuracy 37.6% (train acc. 99.9% — overfit) | 57 |
| Trip-level video embeddings + Logistic Regression | LOGO Accuracy 30% · Macro F1 0.32 | 59 |
| Optical flow + CNN-LSTM (LOGO, 40-fold) | Highly unstable per fold (2%–67% accuracy), no stable aggregate | 64, 65 |

## Installation

```bash
git clone https://github.com/Aleyna134/DriverMind-AI.git
cd DriverMind-AI
pip install -r requirements.txt   # streamlit, torch, numpy, pandas, altair, scikit-learn, joblib
```

`datasets/` is excluded from this repository (raw UAH-DriveSet recordings, optical-flow caches,
and processed `.npz` feature dumps total several hundred GB). 

The pretrained checkpoint in `models/` and precomputed scaler/default configuration files are already tracked in git, so **the dashboard runs out of the box** without regenerating the dataset. 

However, if you want to rebuild the processed data and retrain the model from scratch:

1. Download the [UAH-DriveSet](http://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/) and place it under `datasets/raw/UAH-DRIVESET-v1/`.
2. Instead of running all 75 exploratory/research notebooks, you only need to run the following core notebooks in order to rebuild the production pipeline:
   - **`69_UAH_Window_Size_160_Dataset.ipynb`**: Generates the 160-timestep windowed sequences (`uah_dataset_window160.npz`).
   - **`73_UAH_LSTM_Feature_Importance.ipynb`**: Retrains the final LSTM model (`lstm_window160_dropout02_macrof1_07358.pth`). *(Note: Save the fitted scaler and feature defaults here if they are deleted).*
   - **`74_UAH_Simulation_Analysis.ipynb`**: Precomputes the simulation vectors (`simulation_vectors.npy`, `simulation_vectors_scaled.npy`) and simulation scaler (`simulation_scaler.pkl`) required for the Streamlit dashboard simulation.


## Usage

Launch the live dashboard:

```bash
streamlit run dashboard/app.py
```

Adjust the 4 sensor sliders (speed, front distance, lane offset, relative speed) and press
**Start** to stream simulated readings into the model. The dashboard shows the live risk score,
predicted class, model confidence, a risk trend chart, active risk factors, a driving
recommendation, and a live overview of each input sensor.

## License

MIT License
