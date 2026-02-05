# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# sklearn (KMeans)
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception:
    KMeans = None
    StandardScaler = None

st.set_page_config(layout="wide", page_title="Sync + Markov/KMeans por série — tabelas de parâmetros")
st.title("Sync por salto + Segmentação (K-means/Markov por série no ROI) — parâmetros cinemática e giroscópio")

FS_TARGET = 100.0
TRIGGER_VIEW_SEC = 20.0
CUTOFF_HZ = 1.5

# -------------------------
# Helpers
# -------------------------
def read_table(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file, sep=None, engine="python")

def safe_numeric(series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(x).sum() < 5:
        raise ValueError("A coluna selecionada não parece numérica (muitos NaN/inf).")
    return x

def fix_nans_1d(x: np.ndarray) -> tuple[np.ndarray, int]:
    s = pd.Series(x.astype(float))
    n_nan = int(s.isna().sum())
    if n_nan > 0:
        s = s.interpolate(method="linear", limit_direction="both")
        s = s.bfill().ffill()
    return s.to_numpy(dtype=float), n_nan

def time_vector(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=float) / float(fs)

def guess_time_unit_and_convert_to_seconds(t: np.ndarray) -> np.ndarray:
    t = t.astype(float)
    t_sorted = np.sort(t[np.isfinite(t)])
    if len(t_sorted) < 5:
        return t
    dt_med = np.nanmedian(np.diff(t_sorted))
    if dt_med > 1.0:  # provável ms
        return t / 1000.0
    return t

def interpolate_to_fs(t_in: np.ndarray, x_in: np.ndarray, fs_out: float):
    order = np.argsort(t_in)
    t_in = t_in[order]
    x_in = x_in[order]

    dt = np.diff(t_in)
    keep = np.hstack(([True], dt > 0))
    t_in = t_in[keep]
    x_in = x_in[keep]

    m = np.isfinite(t_in) & np.isfinite(x_in)
    t_in = t_in[m]
    x_in = x_in[m]

    if len(t_in) < 5:
        raise ValueError("Tempo do giroscópio inválido (poucos pontos válidos).")

    t_out = np.arange(t_in[0], t_in[-1], 1.0 / fs_out)
    x_out = np.interp(t_out, t_in, x_in)
    return t_out, x_out

def butter_lowpass_filtfilt(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    b, a = signal.butter(order, wn, btype="lowpass")
    return signal.filtfilt(b, a, x)

def plot_with_lines(
    t, x, title,
    vlines_black=None, labels_black=None,
    vlines_red=None, labels_red=None,
    xlabel="Tempo (s)"
):
    fig = plt.figure()
    plt.plot(t, x)

    if vlines_black is not None:
        for i, vl in enumerate(vlines_black):
            lab = labels_black[i] if labels_black and i < len(labels_black) else None
            plt.axvline(vl, linestyle="--", label=lab)

    if vlines_red is not None:
        for i, vl in enumerate(vlines_red):
            lab = labels_red[i] if labels_red and i < len(labels_red) else None
            plt.axvline(vl, linestyle="--", color="red", label=lab)

    if (labels_black and len(labels_black) > 0) or (labels_red and len(labels_red) > 0):
        plt.legend(loc="upper left", fontsize=8)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def segment_by_time(t: np.ndarray, x: np.ndarray, t0: float, t1: float):
    if t0 is None or t1 is None:
        return np.array([]), np.array([]), np.array([], dtype=bool)
    if t0 > t1:
        t0, t1 = t1, t0
    m = (t >= t0) & (t <= t1)
    return t[m], x[m], m

def mode_int(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(cnts)])

def baseline_from_start(states: np.ndarray, fs: float, baseline_sec: float = 1.0) -> int:
    n = int(round(baseline_sec * fs))
    n = max(1, min(n, len(states)))
    return mode_int(states[:n])

def baseline_from_end_marker(t: np.ndarray, states: np.ndarray, fs: float, end_time: float, baseline_sec: float = 1.0) -> tuple[int, int]:
    if len(t) == 0:
        return None, None
    end_idx = int(np.argmin(np.abs(t - float(end_time))))
    w = int(round(baseline_sec * fs))
    i0 = max(0, end_idx - w)
    end_baseline_state = mode_int(states[i0:end_idx + 1])
    return end_baseline_state, end_idx

def is_run(arr: np.ndarray, idx: int, length: int, value=None, not_value=None) -> bool:
    if idx < 0 or idx + length > len(arr):
        return False
    w = arr[idx: idx + length]
    if value is not None:
        return bool(np.all(w == value))
    if not_value is not None:
        return bool(np.all(w != not_value))
    return False

def detect_start_10_5(t: np.ndarray, states: np.ndarray, baseline_state: int, baseline_run: int = 10, other_run: int = 5):
    if baseline_state is None:
        return None
    for i in range(baseline_run, len(states) - other_run):
        if is_run(states, i - baseline_run, baseline_run, value=baseline_state) and \
           is_run(states, i, other_run, not_value=baseline_state):
            return float(t[i])
    return None

def detect_end_from_marker_10_5(t: np.ndarray, states: np.ndarray, fs: float,
                               end_time: float, end_baseline_sec: float = 1.0,
                               baseline_run: int = 10, other_run: int = 5):
    end_baseline_state, end_idx = baseline_from_end_marker(t, states, fs, end_time, baseline_sec=end_baseline_sec)
    if end_baseline_state is None or end_idx is None:
        return None, None

    last_possible_i = min(end_idx - (baseline_run + other_run), len(states) - (baseline_run + other_run))
    if last_possible_i < 0:
        return end_baseline_state, None

    for i in range(last_possible_i, -1, -1):
        if is_run(states, i, other_run, not_value=end_baseline_state) and \
           is_run(states, i + other_run, baseline_run, value=end_baseline_state):
            return end_baseline_state, float(t[i + other_run])

    return end_baseline_state, None

def kmeans_states_1d(x_common: np.ndarray, fs: float, n_states: int, seed: int):
    if KMeans is None or StandardScaler is None:
        raise RuntimeError("sklearn não disponível.")
    x = np.asarray(x_common, dtype=float)
    dx = np.gradient(x) * fs
    Xfeat = np.column_stack([x, dx])
    Xs = StandardScaler().fit_transform(Xfeat)
    km = KMeans(n_clusters=n_states, random_state=int(seed), n_init=20)
    return km.fit_predict(Xs).astype(int)

def top2_peaks_in_window(t: np.ndarray, x: np.ndarray, t0: float, t1: float, fs: float):
    if t0 is None or t1 is None:
        return []
    if t0 > t1:
        t0, t1 = t1, t0
    tt, xx, _ = segment_by_time(t, x, t0, t1)
    if len(xx) < 10:
        return []
    dist = int(round(0.2 * fs))
    peaks, _ = signal.find_peaks(xx, distance=max(1, dist))
    if len(peaks) == 0:
        return []
    amps = xx[peaks]
    order = np.argsort(amps)[::-1]
    sel = order[:2]
    out = [(float(tt[int(peaks[i])]), float(xx[int(peaks[i])])) for i in sel]
    # ordena por tempo também, útil para "menor tempo" e "maior tempo"
    out_time = sorted(out, key=lambda z: z[0])
    return out, out_time  # (top2 por amp), (mesmos 2 ordenados por tempo)

def first_peak_after_time_within(t: np.ndarray, x: np.ndarray, t0: float, t1: float, fs: float):
    if t0 is None or t1 is None:
        return None, None
    if t0 > t1:
        t0, t1 = t1, t0
    tt, xx, _ = segment_by_time(t, x, t0, t1)
    if len(xx) < 10:
        return None, None
    dist = int(round(0.2 * fs))
    peaks, _ = signal.find_peaks(xx, distance=max(1, dist))
    if len(peaks) == 0:
        return None, None
    p = int(peaks[0])
    return float(tt[p]), float(xx[p])

def peaks_first_last_in_window(t: np.ndarray, x: np.ndarray, t0: float, t1: float, fs: float):
    if t0 is None or t1 is None:
        return (None, None), (None, None)
    if t0 > t1:
        t0, t1 = t1, t0
    tt, xx, _ = segment_by_time(t, x, t0, t1)
    if len(xx) < 10:
        return (None, None), (None, None)
    dist = int(round(0.2 * fs))
    peaks, _ = signal.find_peaks(xx, distance=max(1, dist))
    if len(peaks) == 0:
        return (None, None), (None, None)
    p_first = int(peaks[0])
    p_last = int(peaks[-1])
    return (float(tt[p_first]), float(xx[p_first])), (float(tt[p_last]), float(xx[p_last]))

def safe_diff(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)

# -------------------------
# Uploads
# -------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("1) Cinemática")
    kin_file = st.file_uploader("Carregue o arquivo de cinemática (.csv/.txt)", type=["csv", "txt"], key="kin")
with colB:
    st.subheader("2) Giroscópio")
    gyro_file = st.file_uploader("Carregue o arquivo do giroscópio (.csv/.txt)", type=["csv", "txt"], key="gyro")

kin_ready = False
gyro_ready = False

# -------------------------
# Cinemática
# -------------------------
if kin_file is not None:
    try:
        df_kin = read_table(kin_file)
        if df_kin.shape[1] < 3:
            raise ValueError("O arquivo de cinemática deve ter pelo menos 3 colunas: X, Y, Z.")
        st.success(f"Cinemática: {df_kin.shape[0]} linhas × {df_kin.shape[1]} colunas")

        col_x = df_kin.columns[0]  # ignora
        col_y = df_kin.columns[1]  # Y (AP)
        col_z = df_kin.columns[2]  # Z (vertical)

        y_kin_raw = safe_numeric(df_kin[col_y])
        z_kin_raw = safe_numeric(df_kin[col_z])

        y_kin, n_nan_y = fix_nans_1d(y_kin_raw)
        z_kin, n_nan_z = fix_nans_1d(z_kin_raw)

        if (n_nan_y + n_nan_z) > 0:
            st.warning(f"Cinemática: corrigidos NaNs/vazios → Y: {n_nan_y}, Z: {n_nan_z}")

        t_kin = time_vector(len(df_kin), FS_TARGET)
        st.caption(f"Mapeamento cinemática: X='{col_x}' (ignorado), Y='{col_y}', Z='{col_z}'")
        kin_ready = True
    except Exception as e:
        st.error(f"Erro ao processar cinemática: {e}")

# -------------------------
# Giroscópio
# -------------------------
if gyro_file is not None:
    try:
        df_g = read_table(gyro_file)
        cols_g = list(df_g.columns)
        if len(cols_g) < 4:
            raise ValueError("O arquivo do giroscópio deve ter pelo menos 4 colunas: tempo + 3 eixos.")
        st.success(f"Giroscópio: {df_g.shape[0]} linhas × {df_g.shape[1]} colunas")

        time_col = cols_g[0]
        t_g_in = safe_numeric(df_g[time_col])
        t_g_in = guess_time_unit_and_convert_to_seconds(t_g_in)

        g1, g2, g3 = st.columns(3)
        with g1:
            gx_col = st.selectbox("Coluna gyro X", cols_g[1:], index=0)
        with g2:
            gy_col = st.selectbox("Coluna gyro Y (vertical p/ salto)", cols_g[1:], index=min(1, len(cols_g[1:]) - 1))
        with g3:
            gz_col = st.selectbox("Coluna gyro Z", cols_g[1:], index=min(2, len(cols_g[1:]) - 1))

        gx = safe_numeric(df_g[gx_col])
        gy = safe_numeric(df_g[gy_col])
        gz = safe_numeric(df_g[gz_col])

        gx_dt = signal.detrend(gx, type="linear")
        gy_dt = signal.detrend(gy, type="linear")
        gz_dt = signal.detrend(gz, type="linear")

        t_g, gx_i = interpolate_to_fs(t_g_in, gx_dt, FS_TARGET)
        _,   gy_i = interpolate_to_fs(t_g_in, gy_dt, FS_TARGET)
        _,   gz_i = interpolate_to_fs(t_g_in, gz_dt, FS_TARGET)

        gx_f = butter_lowpass_filtfilt(gx_i, FS_TARGET, cutoff_hz=CUTOFF_HZ, order=4)
        gy_f = butter_lowpass_filtfilt(gy_i, FS_TARGET, cutoff_hz=CUTOFF_HZ, order=4)
        gz_f = butter_lowpass_filtfilt(gz_i, FS_TARGET, cutoff_hz=CUTOFF_HZ, order=4)

        g_norm = np.sqrt(gx_f**2 + gy_f**2 + gz_f**2)
        gyro_ready = True
    except Exception as e:
        st.error(f"Erro ao processar giroscópio: {e}")

# -------------------------
# Trigger + ROI + Markov + Métricas + Plots
# -------------------------
if kin_ready and gyro_ready:
    st.divider()
    st.subheader("Trigger (0–20 s) — escolha do pico e zeragem do tempo")

    tmax = float(TRIGGER_VIEW_SEC)
    mk = (t_kin >= 0) & (t_kin <= tmax)
    mg = (t_g >= 0) & (t_g <= tmax)

    i1, i2 = st.columns(2)
    with i1:
        t_peak_kin = st.number_input(
            "Tempo do pico — cinemática (s) [usar Z]",
            min_value=0.0, max_value=tmax, value=1.0, step=0.01, format="%.2f"
        )
    with i2:
        t_peak_gyro = st.number_input(
            "Tempo do pico — giroscópio (s) [usar Y filtrado]",
            min_value=0.0, max_value=tmax, value=1.0, step=0.01, format="%.2f"
        )

    cL, cR = st.columns(2)
    with cL:
        plot_with_lines(
            t_kin[mk], z_kin[mk], "Cinemática — Z (0–20 s)",
            vlines_black=[float(t_peak_kin)], labels_black=["pico"]
        )
    with cR:
        plot_with_lines(
            t_g[mg], gy_f[mg], "Giroscópio — Y filtrado (0–20 s)",
            vlines_black=[float(t_peak_gyro)], labels_black=["pico"]
        )

    t_kin_sync = t_kin - float(t_peak_kin)
    t_g_sync   = t_g   - float(t_peak_gyro)
    st.info("Tempo sincronizado: t_sync = t − t_pico. Pico ocorre em t=0 em ambos.")

    # ROI manual
    st.divider()
    st.subheader("Intervalo de interesse (ROI)")

    tmin_common = float(max(np.min(t_kin_sync), np.min(t_g_sync)))
    tmax_common = float(min(np.max(t_kin_sync), np.max(t_g_sync)))
    if tmax_common <= tmin_common:
        st.error("Não foi possível encontrar faixa temporal comum após sincronização.")
        st.stop()

    s1, s2 = st.columns(2)
    with s1:
        roi_start = st.number_input(
            "ROI início (s)",
            min_value=tmin_common, max_value=tmax_common,
            value=max(0.0, tmin_common), step=0.05, format="%.2f"
        )
    with s2:
        roi_end = st.number_input(
            "ROI fim (s)",
            min_value=tmin_common, max_value=tmax_common,
            value=min(10.0, tmax_common), step=0.05, format="%.2f"
        )
    if roi_start > roi_end:
        roi_start, roi_end = roi_end, roi_start

    st.divider()
    st.subheader("K-means/Markov por série no ROI — linhas vermelhas = início/fim Markov")

    if KMeans is None or StandardScaler is None:
        st.error("Para esta etapa é necessário scikit-learn (sklearn).")
        st.stop()

    if (roi_end - roi_start) < 2.0:
        st.warning("ROI muito curto. Recomendo >= 2s (baseline 1s + detecção).")
        st.stop()

    t_common = np.arange(roi_start, roi_end, 1.0 / FS_TARGET)
    if len(t_common) < 50:
        st.warning("Poucas amostras no ROI para segmentação estável.")
        st.stop()

    y_common = np.interp(t_common, t_kin_sync, y_kin)
    z_common = np.interp(t_common, t_kin_sync, z_kin)
    g_common = np.interp(t_common, t_g_sync, g_norm)

    seed = st.number_input("Seed do K-means (usado em todas as séries)", min_value=0, value=42, step=1)

    states_y = kmeans_states_1d(y_common, FS_TARGET, n_states=6, seed=int(seed))
    states_z = kmeans_states_1d(z_common, FS_TARGET, n_states=6, seed=int(seed))
    states_g = kmeans_states_1d(g_common, FS_TARGET, n_states=6, seed=int(seed))

    # Início: baseline do 1º segundo do ROI
    base_y_start = baseline_from_start(states_y, FS_TARGET, baseline_sec=1.0)
    base_z_start = baseline_from_start(states_z, FS_TARGET, baseline_sec=1.0)
    base_g_start = baseline_from_start(states_g, FS_TARGET, baseline_sec=1.0)

    y_start = detect_start_10_5(t_common, states_y, base_y_start, baseline_run=10, other_run=5)
    z_start = detect_start_10_5(t_common, states_z, base_z_start, baseline_run=10, other_run=5)
    g_start = detect_start_10_5(t_common, states_g, base_g_start, baseline_run=10, other_run=5)

    # Fim: ancorado em ROI fim
    base_y_end, y_end = detect_end_from_marker_10_5(t_common, states_y, FS_TARGET, end_time=roi_end,
                                                    end_baseline_sec=1.0, baseline_run=10, other_run=5)
    base_z_end, z_end = detect_end_from_marker_10_5(t_common, states_z, FS_TARGET, end_time=roi_end,
                                                    end_baseline_sec=1.0, baseline_run=10, other_run=5)
    base_g_end, g_end = detect_end_from_marker_10_5(t_common, states_g, FS_TARGET, end_time=roi_end,
                                                    end_baseline_sec=1.0, baseline_run=10, other_run=5)

    # -------------------------
    # Derivados CINEMÁTICA
    # -------------------------
    # mínimo Y entre y_start e y_end (no sinal completo sincronizado)
    y_min_t = None
    if y_start is not None and y_end is not None:
        t0, t1 = (y_start, y_end) if y_start <= y_end else (y_end, y_start)
        ty, yy, _ = segment_by_time(t_kin_sync, y_kin, t0, t1)
        if len(yy) >= 3:
            idx_min = int(np.nanargmin(yy))
            y_min_t = float(ty[idx_min])

    # picos Z: 1º após z_start e 1º antes z_end (no sinal completo sincronizado)
    (z_peak1_t, z_peak1_val), (z_peak2_t, z_peak2_val) = (None, None), (None, None)
    if z_start is not None and z_end is not None:
        t0z, t1z = (z_start, z_end) if z_start <= z_end else (z_end, z_start)
        (z_peak1_t, z_peak1_val), (z_peak2_t, z_peak2_val) = peaks_first_last_in_window(
            t_kin_sync, z_kin, t0z, t1z, FS_TARGET
        )

    # -------------------------
    # Derivados GIROSCÓPIO
    # -------------------------
    gyro_top2_by_amp = []
    gyro_top2_by_time = []
    g_first_peak_t = g_first_peak_val = None

    if g_start is not None and g_end is not None:
        (gyro_top2_by_amp, gyro_top2_by_time) = top2_peaks_in_window(
            t_g_sync, g_norm, g_start, g_end, fs=FS_TARGET
        )
        g_first_peak_t, g_first_peak_val = first_peak_after_time_within(
            t_g_sync, g_norm, g_start, g_end, fs=FS_TARGET
        )

    # “menor tempo” e “maior tempo” (entre os 2 maiores picos por amplitude)
    # Se só houver 1 pico, ambos viram o mesmo.
    gyro_early_t = gyro_early_val = None
    gyro_late_t = gyro_late_val = None
    if len(gyro_top2_by_time) >= 1:
        gyro_early_t, gyro_early_val = gyro_top2_by_time[0]
        gyro_late_t, gyro_late_val = gyro_top2_by_time[-1]

    # -------------------------
    # TABELAS (como você pediu)
    # -------------------------
    st.divider()
    st.subheader("Tabelas de parâmetros")

    # ---- Tabela cinemática
    kin_params = {
        "Início do sinal AP (Y)": y_start,
        "Final do sinal AP (Y)": y_end,
        "Início do sinal V (Z)": z_start,
        "Final do sinal V (Z)": z_end,
        "Alcance em 3 m AP (min Y)": y_min_t,
        "Pico sentado→pé em Z (1º pico após início)": z_peak1_t,
        "Pico pé→sentado em Z (pico antes do final)": z_peak2_t,
        "Duração total AP": safe_diff(y_end, y_start),
        "Duração total V": safe_diff(z_end, z_start),
        "Duração da ida": safe_diff(y_min_t, z_peak1_t),
        "Duração da volta": safe_diff(z_peak2_t, y_min_t),
        "Duração para ficar em pé": safe_diff(z_peak1_t, z_start),
        "Duração para sentar": safe_diff(z_end, z_peak2_t),
    }
    kin_table = pd.DataFrame(
        [{"Parâmetro": k, "Valor (s)": v} for k, v in kin_params.items()]
    )
    st.markdown("### Cinemática")
    st.dataframe(kin_table, use_container_width=True)

    # ---- Tabela giroscópio
    gyro_params = {
        "Início giro": g_start,
        "Final giro": g_end,
        "Ficar em sentado/pé (1º pico após início)": g_first_peak_t,
        "Alcance em 3 m (grande giro de menor tempo)": gyro_early_t,
        "Giro na cadeira (grande giro de maior tempo)": gyro_late_t,
        "Duração total giro": safe_diff(g_end, g_start),
        "Duração de sentado/pé giro": safe_diff(g_first_peak_t, g_start),
        "Duração da ida giro": safe_diff(gyro_early_t, g_first_peak_t),
        "Duração da volta giro": safe_diff(gyro_late_t, gyro_early_t),
        "Duração de pé/sentado giro": safe_diff(gyro_late_t, g_end),  # conforme seu texto
    }
    gyro_table = pd.DataFrame(
        [{"Parâmetro": k, "Valor (s)": v} for k, v in gyro_params.items()]
    )
    st.markdown("### Giroscópio")
    st.dataframe(gyro_table, use_container_width=True)

    # -------------------------
    # PLOTS (estrutura 3 colunas)
    # -------------------------
    st.divider()
    st.subheader("Gráficos (com ROI, Markov e marcadores)")

    r1, r2, r3 = st.columns(3)

    with r1:
        fig = plt.figure()
        plt.plot(t_kin_sync, y_kin)
        plt.axvline(roi_start, linestyle="--", label="ROI início")
        plt.axvline(roi_end, linestyle="--", label="ROI fim")
        for vl, lab in [(y_start, "Y início (Markov)"), (y_end, "Y fim (Markov)")]:
            if vl is not None:
                plt.axvline(vl, linestyle="--", color="red", label=lab)
        if y_min_t is not None:
            y_min_val = float(np.interp(y_min_t, t_kin_sync, y_kin))
            plt.axvline(y_min_t, linestyle="-", label="t do mínimo")
            plt.scatter([y_min_t], [y_min_val], s=40, label="mínimo Y")
        plt.title("Cinemática Y (AP)")
        plt.xlabel("Tempo sincronizado (s)")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with r2:
        fig = plt.figure()
        plt.plot(t_kin_sync, z_kin)
        plt.axvline(roi_start, linestyle="--", label="ROI início")
        plt.axvline(roi_end, linestyle="--", label="ROI fim")
        for vl, lab in [(z_start, "Z início (Markov)"), (z_end, "Z fim (Markov)")]:
            if vl is not None:
                plt.axvline(vl, linestyle="--", color="red", label=lab)
        if z_peak1_t is not None:
            plt.scatter([z_peak1_t], [z_peak1_val], s=45, label="1º pico após início")
        if z_peak2_t is not None:
            plt.scatter([z_peak2_t], [z_peak2_val], s=45, label="pico antes do fim")
        plt.title("Cinemática Z (vertical)")
        plt.xlabel("Tempo sincronizado (s)")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with r3:
        fig = plt.figure()
        plt.plot(t_g_sync, g_norm)
        plt.axvline(roi_start, linestyle="--", label="ROI início")
        plt.axvline(roi_end, linestyle="--", label="ROI fim")
        for vl, lab in [(g_start, "G início (Markov)"), (g_end, "G fim (Markov)")]:
            if vl is not None:
                plt.axvline(vl, linestyle="--", color="red", label=lab)

        # marca 2 maiores picos (por amplitude)
        for k, (tp, vp) in enumerate(gyro_top2_by_amp, start=1):
            plt.scatter([tp], [vp], s=55, label=f"Top {k} pico")

        # marca 1º pico após início
        if g_first_peak_t is not None:
            plt.scatter([g_first_peak_t], [g_first_peak_val], s=65, label="1º pico após início")

        plt.title("Giroscópio — norma (||gyro||)")
        plt.xlabel("Tempo sincronizado (s)")
        plt.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    # -------------------------
    # Segmentos recortados
    # -------------------------
    st.divider()
    st.subheader("Segmentos recortados (usando início/fim Markov)")

    def plot_segment(name, t_start, t_end, t_full, x_full):
        if t_start is None or t_end is None:
            st.caption(f"{name}: não foi possível detectar início/fim.")
            return
        t_seg2, x_seg2, _ = segment_by_time(t_full, x_full, t_start, t_end)
        plot_with_lines(t_seg2, x_seg2, f"{name} — segmento", xlabel="Tempo sincronizado (s)")

    a1, a2, a3 = st.columns(3)
    with a1:
        plot_segment("Cinemática Y", y_start, y_end, t_kin_sync, y_kin)
    with a2:
        plot_segment("Cinemática Z", z_start, z_end, t_kin_sync, z_kin)
    with a3:
        plot_segment("Giroscópio norma", g_start, g_end, t_g_sync, g_norm)

else:
    st.caption("Carregue os dois arquivos para habilitar trigger, sincronização, ROI e segmentação por série.")
