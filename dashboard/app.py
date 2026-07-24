import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

from styles.theme import inject_theme, GOOD, TEXT_SECONDARY
from components.header import render_header
from components.metrics import render_metrics
from components.trend_chart import render_trend_chart
from components.feature_controls import render_feature_controls
from components.risk_factors import render_risk_factors
from src.predictor import predict, build_row
from src.config import SEQUENCE_LENGTH

st.set_page_config(
    page_title="Driver Risk Intelligence",
    layout="wide",
)

inject_theme()

if "prediction" not in st.session_state:
    st.session_state["prediction"] = {
        "risk_score": 0,
        "prediction": "-",
        "confidence": 0.0,
        "risk_factors": [
            "Henüz tahmin yapılmadı",
        ]
    }

if "running" not in st.session_state:
    st.session_state["running"] = False

if "risk_history" not in st.session_state:
    st.session_state["risk_history"] = []

if "elapsed_offset" not in st.session_state:
    st.session_state["elapsed_offset"] = 0.0

if "run_start_ts" not in st.session_state:
    st.session_state["run_start_ts"] = None

if "calm_streak_start" not in st.session_state:
    st.session_state["calm_streak_start"] = None

if "last_aggressive_ts" not in st.session_state:
    st.session_state["last_aggressive_ts"] = None

if "feature_history" not in st.session_state:
    st.session_state["feature_history"] = []

if "prev_speed" not in st.session_state:
    st.session_state["prev_speed"] = None

if "prev_tick_time" not in st.session_state:
    st.session_state["prev_tick_time"] = None

if "acc_x_signal" not in st.session_state:
    st.session_state["acc_x_signal"] = 0.0

if "prev_lane_offset" not in st.session_state:
    st.session_state["prev_lane_offset"] = None

if "lane_drift_start_ts" not in st.session_state:
    st.session_state["lane_drift_start_ts"] = None

if "lane_drift_last_move_ts" not in st.session_state:
    st.session_state["lane_drift_last_move_ts"] = None

if "prev_lane_offset_accel" not in st.session_state:
    st.session_state["prev_lane_offset_accel"] = None

if "prev_lane_offset_tick_time" not in st.session_state:
    st.session_state["prev_lane_offset_tick_time"] = None

if "acc_y_signal" not in st.session_state:
    st.session_state["acc_y_signal"] = 0.0

# Gerçek verideki acc_x aralığı (bkz. notebooks/74), modelin
# "Aggressive" kararını en çok etkileyen sensör.
ACC_X_BOUNDS = (-0.45, 0.5)

# Modelin bir freni/gazı fark edebilmesi için sinyalin pencerede
# yeterince uzun sürmesi gerekiyor (tek bir anlık tick yeterli değil,
# gerçek bir yolculukta da fren G-kuvveti anında sıfıra dönmez).
# Bu yüzden sinyal en büyük değere anında sıçrar, sonra ~8 saniyede
# kademeli olarak sönümlenir.
ACC_X_DECAY = 0.93


def derive_acc_x(current_speed):
    """
    Ani hız değişimlerini (sert fren/hızlanma) gerçekçi bir
    boylamsal ivme (acc_x, g biriminde) değerine çevirir.
    Kullanıcı sadece "Speed" slider'ını hareket ettirse bile,
    model bunu gerçek bir fren/gaz olayı olarak görebilsin diye.
    """

    now = time.time()
    prev_speed = st.session_state["prev_speed"]
    prev_time = st.session_state["prev_tick_time"]

    instantaneous = 0.0

    if prev_speed is not None and prev_time is not None:
        delta_speed_kmh = current_speed - prev_speed

        # +/- butonuyla tek tek (ör. 1 km/h) artırmak gerçek bir fren/gaz
        # olayı değil, kademeli/kontrollü bir değişikliktir. Küçük
        # adımları tamamen görmezden geliyoruz; ayrıca dt tabanını
        # yükseltiyoruz ki art arda hızlı tıklamalar (gerçek zaman farkı
        # çok küçükken) yapay olarak devasa bir ivmeye dönüşmesin.
        if abs(delta_speed_kmh) >= 2.0:
            dt = max(now - prev_time, 1.0)
            delta_speed_ms = delta_speed_kmh / 3.6
            accel_ms2 = delta_speed_ms / dt
            instantaneous = accel_ms2 / 9.81

    signal = st.session_state["acc_x_signal"]

    if abs(instantaneous) > abs(signal):
        signal = instantaneous
    else:
        signal *= ACC_X_DECAY

    signal = float(np.clip(signal, *ACC_X_BOUNDS))

    st.session_state["acc_x_signal"] = signal
    st.session_state["prev_speed"] = current_speed
    st.session_state["prev_tick_time"] = now

    return signal


# Gerçek verideki acc_y aralığı (bkz. duyarlılık testi), acc_x/acc_z
# gibi modelin "Aggressive" kararını çok güçlü etkileyen bir sensör.
ACC_Y_BOUNDS = (-0.6, 0.4)
ACC_Y_DECAY = 0.93


def derive_acc_y(current_lane_offset):
    """
    Ani şerit değişimlerini (sert direksiyon manevrası / savrulma)
    gerçekçi bir yanal ivme (acc_y, g biriminde) değerine çevirir —
    acc_x'in (boylamsal) yanal karşılığı. Küçük/kademeli slider
    adımları (LANE_DRIFT gibi yavaş kaymalar) burada tetiklenmez;
    sadece kısa sürede büyük bir sıçrama "sert direksiyon" sayılır.
    """

    now = time.time()
    prev = st.session_state["prev_lane_offset_accel"]
    prev_time = st.session_state["prev_lane_offset_tick_time"]

    instantaneous = 0.0

    if prev is not None and prev_time is not None:
        delta = current_lane_offset - prev

        # Tekli slider adımları (0.01) ya da yavaş kademeli kaymalar
        # sert bir manevra değildir — sadece belirgin bir sıçrama sayılır.
        if abs(delta) >= 0.15:
            dt = max(now - prev_time, 1.0)
            instantaneous = (delta / dt) / 9.81

    signal = st.session_state["acc_y_signal"]

    if abs(instantaneous) > abs(signal):
        signal = instantaneous
    else:
        signal *= ACC_Y_DECAY

    signal = float(np.clip(signal, *ACC_Y_BOUNDS))

    st.session_state["acc_y_signal"] = signal
    st.session_state["prev_lane_offset_accel"] = current_lane_offset
    st.session_state["prev_lane_offset_tick_time"] = now

    return signal


# lane_offset bu kadarın altında değişirse "gürültü" sayılır, gerçek
# bir kayma/düzeltme olarak değerlendirilmez.
LANE_DRIFT_EPSILON = 0.003

# Bir tick'te ilerleme görülmese bile (ör. slider'ın kısa süreliğine
# duraksaması) sayaç hemen sıfırlanmaz — bu kadar süre "ilerlemesiz"
# kalırsa gerçekten durduğu kabul edilip sıfırlanır. Sabit bir açıda
# (düzenli bir artış/azalış olmadan) kalmak asla "kayma" saymaz.
LANE_DRIFT_STALL_SECONDS = 1.0


def derive_lane_drift(current_lane_offset):
    """
    lane_offset'in DÜZENLİ olarak (aynı yönde, kesintisiz denecek
    şekilde) büyüyüp büyümediğini izler — yani direksiyonun sabit bir
    açıda kalıp aracın merkezden git gide uzaklaşması. Sadece belli bir
    ofsette sabit durmak (herkes şeridi tam ortalayamaz) bu sayılmaz;
    önemli olan zaman içindeki orantılı artış/azalıştır. Gerçek bir
    düzeltme (merkeze doğru geri gelme) veya yön değişimi sayacı
    sıfırlar; ilerleme yeterince uzun süre (LANE_DRIFT_STALL_SECONDS)
    durursa da sıfırlanır.
    """

    now = time.time()
    prev = st.session_state["prev_lane_offset"]
    last_move_ts = st.session_state["lane_drift_last_move_ts"]

    genuine_growth = False

    if prev is not None:
        moving_away = abs(current_lane_offset) > abs(prev) + LANE_DRIFT_EPSILON
        same_side = current_lane_offset * prev >= 0
        genuine_growth = moving_away and same_side

    if genuine_growth:
        last_move_ts = now
        if st.session_state["lane_drift_start_ts"] is None:
            st.session_state["lane_drift_start_ts"] = now
    elif last_move_ts is None or (now - last_move_ts) > LANE_DRIFT_STALL_SECONDS:
        st.session_state["lane_drift_start_ts"] = None
        last_move_ts = None

    st.session_state["lane_drift_last_move_ts"] = last_move_ts
    st.session_state["prev_lane_offset"] = current_lane_offset

    if st.session_state["lane_drift_start_ts"] is None:
        return 0.0

    return now - st.session_state["lane_drift_start_ts"]


# Hiçbir risk faktörü tetiklenmeden bu kadar saniye geçerse
# risk skoru kademeli olarak düşer ve tahmin "Normal"e döner.
CALM_RAMP_SECONDS = 15

# Aggressive'den çıkışta model bazen anlık olarak "Drowsy" okuyabiliyor
# (is_calm hesaplaması gürültüye çok yakın sınırda bir tık dalgalanabilir).
# Bu, hiçbir kuralın desteklemediği anlamsız bir ara durum olduğu için,
# son Aggressive anından bu kadar saniye geçene kadar "Drowsy" gösterimini
# tamamen bastırıp "Normal"e çeviriyoruz.
POST_AGGRESSIVE_RECOVERY_SECONDS = 20

# Start'a basıldığında henüz gerçek bir geçmiş yok — pencere hâlâ
# başlangıç değerinin tekrarıyla dolu. Tamamen düz/sabit bir sinyal
# modele "monoton" göründüğü için bu ilk anlarda gerçekte olmayan bir
# "Drowsy" okuması çıkabiliyor. Gerçek geçmiş birikene kadar sakin bir
# başlangıç durumu gösteriyoruz.
STARTUP_GRACE_TICKS = 15


def format_elapsed(seconds):
    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def current_elapsed():
    offset = st.session_state["elapsed_offset"]

    if st.session_state["running"] and st.session_state["run_start_ts"] is not None:
        offset += time.time() - st.session_state["run_start_ts"]

    return offset


def current_features():
    return {
        "speed": float(st.session_state["speed"]),
        "front_distance": float(st.session_state["front_distance"]),
        "lane_offset": float(st.session_state["lane_offset"]),
        "relative_speed": float(st.session_state["relative_speed"]),
    }


# ---------------- Header ----------------
mode = render_header()

st.write("")

# ---------------- Main Dashboard ----------------
left_col, right_col = st.columns([1, 2], gap="medium")

with left_col:
    with st.container(border=True):
        features = render_feature_controls()

if features["start"]:
    # Always a full reset — this is the only way to zero the timer/history.
    st.session_state["running"] = True
    st.session_state["elapsed_offset"] = 0.0
    st.session_state["run_start_ts"] = time.time()
    st.session_state["risk_history"] = []
    st.session_state["calm_streak_start"] = None
    st.session_state["last_aggressive_ts"] = None
    st.session_state["feature_history"] = []
    st.session_state["prev_speed"] = None
    st.session_state["prev_tick_time"] = None
    st.session_state["acc_x_signal"] = 0.0
    st.session_state["prev_lane_offset"] = None
    st.session_state["lane_drift_start_ts"] = None
    st.session_state["lane_drift_last_move_ts"] = None
    st.session_state["prev_lane_offset_accel"] = None
    st.session_state["prev_lane_offset_tick_time"] = None
    st.session_state["acc_y_signal"] = 0.0
    st.rerun()

if features["toggle"]:
    if st.session_state["running"]:
        # Button was showing "Stop" -> pause, keep timer/history as-is.
        st.session_state["elapsed_offset"] += time.time() - st.session_state["run_start_ts"]
        st.session_state["running"] = False
        st.session_state["run_start_ts"] = None
    else:
        # Button was showing "Continue" -> resume without resetting anything.
        st.session_state["running"] = True
        st.session_state["run_start_ts"] = time.time()
    # The button lives outside the auto-refreshing fragment, so force an
    # immediate rerun to flip its label (Stop <-> Continue) right away.
    st.rerun()


@st.fragment(run_every=0.2)
def live_panel():

    if st.session_state["running"]:
        current = current_features()

        row_features = dict(current)
        acc_x_signal = derive_acc_x(current["speed"])
        row_features["acc_x"] = acc_x_signal
        acc_y_signal = derive_acc_y(current["lane_offset"])
        row_features["acc_y"] = acc_y_signal
        lane_drift_seconds = derive_lane_drift(current["lane_offset"])

        st.session_state["feature_history"].append(build_row(row_features))
        st.session_state["feature_history"] = st.session_state["feature_history"][-SEQUENCE_LENGTH:]

        if len(st.session_state["feature_history"]) < STARTUP_GRACE_TICKS:
            result = {
                "risk_score": 0,
                "rule_risk_score": 0,
                "prediction": "Normal",
                "confidence": 0.0,
                "risk_factors": ["No Critical Risk"],
            }
            st.session_state["calm_streak_start"] = None
            st.session_state["prediction"] = result
            st.session_state["risk_history"].append(result["risk_score"])
            st.session_state["risk_history"] = st.session_state["risk_history"][-300:]
            return_early = True
        else:
            return_early = False
            result = dict(predict(
                current,
                history=st.session_state["feature_history"],
                acc_x_signal=acc_x_signal,
                acc_y_signal=acc_y_signal,
                lane_drift_seconds=lane_drift_seconds,
            ))

        if not return_early:
            # "Sakin" kararını etikete (prediction) göre vermiyoruz — model
            # bazen gerçekten sakin bir durumda bile inatla "Drowsy" diyebiliyor
            # (sentetik pencerenin kısıtı). Bunun yerine gerçek tehlike
            # göstergelerine bakıyoruz: kural tabanlı skor düşük mü ve
            # yakın zamanda gerçek bir fren/gaz olayı var mı (acc_x_signal
            # hâlâ sönümlenmemiş mi).
            #
            # Eşik 20 değil 30: front_distance'ın kademeli "yakınlık"
            # rampası, tamamen güvenli/tipik bir takip mesafesinde bile
            # (ör. varsayılan 25m) hiçbir ayrık uyarı tetiklenmeden ~24
            # puan katkı yapıyor. Eşik çok düşük olursa (20) normal bir
            # takip mesafesi tek başına "sakin" sayılmayı engelliyor ve
            # sakin sürüş mekanizması hiç devreye giremiyordu.
            is_calm = (
                result["risk_factors"] == ["No Critical Risk"]
                and result["rule_risk_score"] < 30
                and abs(st.session_state["acc_x_signal"]) < 0.05
                and abs(st.session_state["acc_y_signal"]) < 0.05
            )
            now = time.time()

            if result["prediction"] == "Aggressive":
                st.session_state["last_aggressive_ts"] = now

            recently_aggressive = (
                st.session_state["last_aggressive_ts"] is not None
                and (now - st.session_state["last_aggressive_ts"]) < POST_AGGRESSIVE_RECOVERY_SECONDS
            )

            if is_calm:
                if st.session_state["calm_streak_start"] is None:
                    st.session_state["calm_streak_start"] = now
                calm_duration = now - st.session_state["calm_streak_start"]
            else:
                st.session_state["calm_streak_start"] = None
                calm_duration = 0.0

            progress = min(calm_duration / CALM_RAMP_SECONDS, 1.0)

            if progress > 0:
                result["risk_score"] = round(result["risk_score"] * (1 - progress))

            # Bu bastırma sadece modelin HAM/desteksiz "Drowsy" okuması
            # için geçerli. Şerit kayması (lane drift) artık "Drowsy"yi
            # gerçekten destekleyen bir kural olduğu için, o durumda
            # dokunmuyoruz — sadece is_calm/recently_aggressive'den
            # kaynaklanan (kural desteksiz) "Drowsy" durumlarını
            # bastırıyoruz. Etiketi doğrudan "Normal"e çekmek yerine
            # skora bakıyoruz: skor hâlâ yüksekse (az önce Aggressive'ken
            # kademeli düşüyor demektir) etiket de "Aggressive" kalmaya
            # devam etsin — "risk skoru 83 ama etiket Normal" gibi
            # tutarsız bir görüntü olmasın.
            if (
                result["prediction"] == "Drowsy"
                and not result["drowsy_rule_active"]
                and (is_calm or recently_aggressive)
            ):
                if recently_aggressive and result["risk_score"] >= 50:
                    result["prediction"] = "Aggressive"
                else:
                    result["prediction"] = "Normal"
                # Ham modelin (artık gösterilmeyen) "Drowsy" güveni burada
                # kalıp göstergeyi yeni etiketle çelişir hale getirmesin —
                # gösterilen sayı, kararı gerçekten verdiğimiz risk_score
                # ile birebir uyumlu olsun.
                result["confidence"] = float(result["risk_score"])

            if progress >= 1.0:
                result["risk_factors"] = ["Sürekli güvenli sürüş"]

            st.session_state["prediction"] = result

            st.session_state["risk_history"].append(result["risk_score"])
            st.session_state["risk_history"] = st.session_state["risk_history"][-300:]

    with st.container(border=True):

        status_col, timer_col = st.columns([2, 1])

        with status_col:
            if st.session_state["running"]:
                st.markdown(
                    f'<div class="status-pill" style="border-color:{GOOD}55; color:{GOOD};"><span class="status-dot" style="background:{GOOD};"></span>Monitoring active</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="status-pill"><span class="status-dot" style="background:{TEXT_SECONDARY};"></span>Monitoring stopped</div>',
                    unsafe_allow_html=True,
                )

        with timer_col:
            st.markdown(
                f'<div class="timer-box">{format_elapsed(current_elapsed())}</div>',
                unsafe_allow_html=True,
            )

        st.write("")

        render_metrics(st.session_state["prediction"])

        st.write("")

        render_trend_chart(st.session_state["risk_history"])

        st.write("")

        render_risk_factors(st.session_state["prediction"]["risk_factors"])


with right_col:
    live_panel()
