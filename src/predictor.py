import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from src.config import FEATURE_COLUMNS, SEQUENCE_LENGTH
from src.lstm_model import LSTMClassifier


PROJECT_PATH = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL = None
FEATURE_SCALER = None
FEATURE_DEFAULTS = None

CLASS_NAMES = [
    "Normal",
    "Drowsy",
    "Aggressive",
]

NORMAL_IDX = CLASS_NAMES.index("Normal")

# Modelin kendisi front_distance/relative_speed gibi kural-tetikleyici
# özelliklere çok az duyarlı (bkz. duyarlılık analizi) — yani "Short
# Following Distance" gibi bariz tehlikeli bir durum bile ML skorunu
# neredeyse hiç etkilemeyebiliyor. Bu ağırlıklar, kural ihlallerinin
# risk skoruna ve etikete de yansımasını sağlıyor; böylece uyarılar
# ile gösterilen risk/aynı hikayeyi anlatıyor.
RISK_FACTOR_SEVERITY = {
    "High Speed": 25,
    "Lane Departure": 35,
}

# front_distance/relative_speed için sabit eşik yerine kademeli
# (mesafe azaldıkça / kapanma hızı arttıkça yumuşakça artan) katkı.
# Rampa geniş bir aralıkta erken başlıyor ki modelin "monotonluk"
# sinyali zayıflarken kural tarafı henüz yükselmemiş olduğu için
# ortada yapay bir "her şey normal" boşluğu oluşmasın.
CLOSING_DISTANCE_SAFE_M = 80.0
CLOSING_SPEED_MAX_MS = 10.0
CLOSING_RISK_WEIGHT = 35

# ~20 metrenin altında, kapanma hızından bağımsız olarak mesafe
# azaldıkça sürekli/yumuşak şekilde artan ek bir "yakınlık" katkısı.
# (Sert bir eşik yerine kademeli bir eğri — ör. 30m'den 4m'ye inerken
# risk aniden sıçramasın, sürekli artsın.)
PROXIMITY_RAMP_START_M = 20.0
PROXIMITY_MAX_BONUS = 40

# Ani fren/gaz (acc_x sinyali) tek bir olayken pencerede yalnızca
# birkaç adımlık bir iz bırakıyor; pencere zaten gerçek (sakin)
# ölçümlerle doluyken bu, modelin argmax kararını değiştirmeye
# yetmeyebiliyor (skor yükselir ama etiket takılı kalır). Bu yüzden
# sinyal yeterince büyükse kural tarafı da devreye giriyor.
HARSH_ACCEL_RAMP_G = 0.15
HARSH_ACCEL_MAX_G = 0.35
HARSH_ACCEL_WEIGHT = 45

# acc_x'in yanal (sağ/sol) karşılığı: ani/sert bir şerit değişimi
# (savrulma) — yavaş/kademeli kaymadan (Lane Drift, Drowsy) farklı
# olarak burada "Aggressive"i destekliyor.
HARSH_STEER_RAMP_G = 0.15
HARSH_STEER_MAX_G = 0.35
HARSH_STEER_WEIGHT = 45

# Direksiyonun belli bir açıda kalıp aracın düzeltilmeden, sürekli
# aynı yöne kayması (lane_offset zaman içinde tek yönde büyüyor)
# klasik bir dikkat dağınıklığı/uykululuk göstergesi — bu yüzden
# "Aggressive"i değil "Drowsy"yi destekleyen ayrı bir kural sinyali.
LANE_DRIFT_RAMP_SECONDS = 3.0
LANE_DRIFT_MAX_SECONDS = 8.0
LANE_DRIFT_WEIGHT = 65
DROWSY_OVERRIDE_THRESHOLD = 50


def _clamp01(value):
    return max(0.0, min(1.0, value))


def load_model():
    global MODEL

    if MODEL is not None:
        return MODEL

    checkpoint = torch.load(
        PROJECT_PATH / "models" / "lstm_window160_dropout02_macrof1_07358.pth",
        map_location=DEVICE,
    )

    MODEL = LSTMClassifier(
        input_size=checkpoint["num_features"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
        num_classes=checkpoint["num_classes"],
    )
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.to(DEVICE)
    MODEL.eval()

    return MODEL


def load_feature_scaler():
    global FEATURE_SCALER

    if FEATURE_SCALER is None:
        FEATURE_SCALER = joblib.load(
            PROJECT_PATH / "datasets" / "processed" / "lstm_feature_scaler.pkl"
        )

    return FEATURE_SCALER


def load_feature_defaults():
    global FEATURE_DEFAULTS

    if FEATURE_DEFAULTS is None:
        with open(
            PROJECT_PATH / "datasets" / "processed" / "feature_defaults.json",
            encoding="utf-8",
        ) as f:
            FEATURE_DEFAULTS = json.load(f)

    return FEATURE_DEFAULTS


def build_row(features):
    """
    Kullanıcının girdiği 7 sensör değerini ve veri setinin ortalamasından
    gelen 6 sabit değeri FEATURE_COLUMNS sırasıyla tek bir satıra
    (o anki ölçüm) dönüştürür.
    """

    defaults = load_feature_defaults()

    row = [features.get(col, defaults.get(col)) for col in FEATURE_COLUMNS]

    return np.array(row, dtype=np.float32)


def build_window(features, history=None):
    """
    LSTM girişi olacak SEQUENCE_LENGTH uzunluğundaki pencereyi oluşturur.

    `history` verilirse (dashboard'daki canlı akışın her saniye
    biriktirdiği gerçek ölçüm geçmişi), pencere bu gerçek zaman
    dizisinden kurulur — henüz SEQUENCE_LENGTH kadar ölçüm birikmediyse
    baştaki eksik kısım en eski ölçümün tekrarıyla doldurulur. Böylece
    ani bir değişiklik (ör. hızın aniden düşürülmesi) pencerenin son
    adımlarında gerçek bir sıçrama olarak görünür ve model bunu bir
    zaman-serisi örüntüsü olarak algılayabilir.

    `history` verilmezse (tek seferlik/bağımsız çağrı), o anki tek
    ölçüm pencere boyunca sabit tutulur.
    """

    if history:
        rows = list(history[-SEQUENCE_LENGTH:])

        if len(rows) < SEQUENCE_LENGTH:
            pad_count = SEQUENCE_LENGTH - len(rows)
            rows = [rows[0]] * pad_count + rows

        return np.stack(rows).astype(np.float32)

    row = build_row(features)

    return np.tile(row, (SEQUENCE_LENGTH, 1))


def predict(features, history=None, acc_x_signal=0.0, acc_y_signal=0.0, lane_drift_seconds=0.0):

    model = load_model()
    feature_scaler = load_feature_scaler()

    window = build_window(features, history)

    # Model, standardize edilmiş (z-score) özelliklerle eğitildi.
    window_scaled = feature_scaler.transform(window)

    x = (
        torch.tensor(window_scaled, dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    probs = probs.squeeze(0).cpu().numpy()

    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])

    prediction = CLASS_NAMES[pred_idx]

    # Risk skoru = "normal olmama" olasılığı.
    # Böylece Normal tahmin edilen bir durum yüksek güvenle
    # tahmin edilse bile risk skoru düşük kalır.
    ml_risk_score = int(round((1 - probs[NORMAL_IDX]) * 100))

    risk_factors = []

    if features["speed"] > 100:
        risk_factors.append("High Speed")

    if features["front_distance"] < 20:
        risk_factors.append("Short Following Distance")

    if abs(features["lane_offset"]) > 1.0:
        risk_factors.append("Lane Departure")

    if features["relative_speed"] > 8:
        risk_factors.append("High Closing Speed")

    if acc_x_signal <= -HARSH_ACCEL_RAMP_G:
        risk_factors.append("Harsh Braking")
    elif acc_x_signal >= HARSH_ACCEL_RAMP_G:
        risk_factors.append("Harsh Acceleration")

    if abs(acc_y_signal) >= HARSH_STEER_RAMP_G:
        risk_factors.append("Sharp Steering")

    if lane_drift_seconds >= LANE_DRIFT_RAMP_SECONDS:
        risk_factors.append("Lane Drift")

    if len(risk_factors) == 0:
        risk_factors.append("No Critical Risk")

    closing_distance_score = CLOSING_RISK_WEIGHT * _clamp01(
        (CLOSING_DISTANCE_SAFE_M - features["front_distance"]) / CLOSING_DISTANCE_SAFE_M
    )
    closing_speed_score = CLOSING_RISK_WEIGHT * _clamp01(
        features["relative_speed"] / CLOSING_SPEED_MAX_MS
    )
    proximity_bonus = PROXIMITY_MAX_BONUS * _clamp01(
        (PROXIMITY_RAMP_START_M - features["front_distance"]) / PROXIMITY_RAMP_START_M
    )
    harsh_accel_score = HARSH_ACCEL_WEIGHT * _clamp01(
        (abs(acc_x_signal) - HARSH_ACCEL_RAMP_G) / (HARSH_ACCEL_MAX_G - HARSH_ACCEL_RAMP_G)
    )
    harsh_steer_score = HARSH_STEER_WEIGHT * _clamp01(
        (abs(acc_y_signal) - HARSH_STEER_RAMP_G) / (HARSH_STEER_MAX_G - HARSH_STEER_RAMP_G)
    )

    rule_risk_score = min(
        100,
        sum(RISK_FACTOR_SEVERITY.get(factor, 0) for factor in risk_factors)
        + closing_distance_score
        + closing_speed_score
        + proximity_bonus
        + harsh_accel_score
        + harsh_steer_score,
    )
    rule_risk_score = round(rule_risk_score)

    drowsy_rule_score = LANE_DRIFT_WEIGHT * _clamp01(
        (lane_drift_seconds - LANE_DRIFT_RAMP_SECONDS)
        / (LANE_DRIFT_MAX_SECONDS - LANE_DRIFT_RAMP_SECONDS)
    )
    drowsy_rule_score = round(drowsy_rule_score)

    risk_score = max(ml_risk_score, rule_risk_score, drowsy_rule_score)

    # Uyarılar zaten tetiklenmişken etiketin hâlâ "Normal"/"Drowsy" demesi
    # kafa karıştırıcı olurdu — kural tarafı yeterince ciddiyse (yakınlık
    # rampası dahil, artık sert bir eşik değil sürekli bir eğri) etiketi
    # de ona göre güncelle. Model zaten "Aggressive" diyorsa dokunmuyoruz.
    # (ml_risk_score ile kıyaslamıyoruz artık: ml_risk_score sadece
    # "Normal değil" olasılığı, Drowsy'e çok güvenirken bile yüksek
    # çıkabiliyor ve kuralı haksız yere bastırabiliyordu.)
    should_override_aggressive = prediction != "Aggressive" and rule_risk_score >= 50

    if should_override_aggressive:
        prediction = "Aggressive"
        confidence = min(0.99, rule_risk_score / 100)

    # Sürekli/düzeltilmeyen şerit kayması da benzer şekilde "Drowsy"yi
    # zorlayabiliyor — ama Aggressive kuralı zaten devredeyse ona
    # dokunmuyoruz (ikisi aynı anda tetiklenirse daha ciddi olan kazanır).
    drowsy_rule_active = False

    if not should_override_aggressive and drowsy_rule_score >= DROWSY_OVERRIDE_THRESHOLD:
        prediction = "Drowsy"
        confidence = min(0.99, drowsy_rule_score / 100)
        drowsy_rule_active = True

    return {
        "risk_score": risk_score,
        "rule_risk_score": rule_risk_score,
        "drowsy_rule_active": drowsy_rule_active,
        "prediction": prediction,
        "confidence": round(confidence * 100, 2),
        "risk_factors": risk_factors,
    }
