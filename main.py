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

st.set_page_config(layout="wide", page_title="Sync + Markov/KMeans por série (ROI) — segmento único")
st.title("Sincronização por salto + Segmentação automática por série (K-means/Markov no ROI) — 1 segmento por série")

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

def plot_with_lines(t, x, title, vlines=None, labels=None, xlabel="Tempo (s)"):
    fig = plt.figure()
    plt.plot(t, x)
    if vlines is not None:
        for i, vl in enumerate(vlines):
            lab = labels[i] if labels and i < len(labels) else None
            plt.axvline(vl, linestyle="--", label=lab)
        if labels:
            plt.legend(loc="upper left", fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

def segment_by_time(t: np.ndarray, x: np.ndarray, t0: float, t1: float):
    if t0 > t1:
        t0, t1 = t1, t0
    m = (t >= t0) & (t <= t1)
    return t[m], x[m], m

def mode_int(arr: np.ndarray) -> int:
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(cnts)])

def baseline_from_roi_start(states: np.ndarray, fs: float, baseline_sec: float = 1.0) -> int:
    n = int(round(baseline_sec * fs))
    n = max(1, min(n, len(states)))
    return mode_int(states[:n])

def detect_single_segment_from_states(
    t: np.ndarray,
    states: np.ndarray,
    fs: float,
    baseline_sec: float = 1.0,
    baseline_run_before: int = 10,
    other_run_after: int = 5,
):
    """
    Baseline: estado predominante no 1º segundo do ROI.
    Início: 10 amostras baseline seguidas de 5 amostras não-baseline (qualquer outro estado).
    Fim: primeiro retorno ao baseline após o início.
    Retorna: baseline_state, t_start, t_end (ou None se não encontrar).
    """
    if len(states) < (baseline_run_before + other_run_after + 2):
        return None, None, None

    baseline_state = baseline_from_roi_start(states, fs, baseline_sec=baseline_sec)

    def is_run(arr, idx, length, value=None, not_value=None):
        if idx < 0 or idx + length > len(arr):
            return False
        w = arr[idx: idx + length]
        if value is not None:
            return np.all(w == value)
        if not_value is not None:
            return np.all(w != not_value)
        return False

    start_idx = None
    for i in range(baseline_run_before, len(states) - other_run_after):
        if is_run(states, i - baseline_run_before, baseline_run_before, value=baseline_state) and \
           is_run(states, i, other_run_after, not_value=baseline_state):
            start_idx = i
            break

    if start_idx is None:
        return baseline_state, None, None

    end_idx = None
    for j in range(start_idx + other_run_after, len(states)):
        if states[j] == baseline_state:
            end_idx = j
            break

    if end_idx is None:
        end_idx = len(states) - 1

    return baseline_state, float(t[start_idx]), float(t[end_idx])

def segment_markov_kmeans_1d(
    t_common: np.ndarray,
    x_common: np.ndarray,
    fs: float,
    n_states: int,
    seed: int,
):
    """
    Aplica KMeans em UMA série (no ROI). Features: [x(t), dx/dt].
    Retorna states (labels KMeans) no eixo t_common.
    """
    if KMeans is None or StandardScaler is None:
        raise RuntimeError("sklearn não disponível.")

    x = np.asarray(x_common, dtype=float)
    dx = np.gradient(x) * fs

    Xfeat = np.column_stack([x, dx])
    Xs = StandardScaler().fit_transform(Xfeat)

    km = KMeans(n_clusters=n_states, random_state=int(seed), n_init=20)
    states = km.fit_predict(Xs).astype(int)
    return states

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
# Cinemática (fixa por posição: X ignora, Y plota, Z trigger+plota)
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
# Giroscópio (tempo = 1ª coluna) + detrend + interp 100 Hz + LP 1.5 Hz + norma
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
# Trigger manual + sincronização + ROI + Markov/KMeans por série no ROI (segmento único)
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
        plot_with_lines(t_kin[mk], z_kin[mk], "Cinemática — Z (0–20 s)",
                        vlines=[float(t_peak_kin)], labels=["pico"])
    with cR:
        plot_with_lines(t_g[mg], gy_f[mg], "Giroscópio — Y filtrado (0–20 s)",
                        vlines=[float(t_peak_gyro)], labels=["pico"])

    t_kin_sync = t_kin - float(t_peak_kin)
    t_g_sync   = t_g   - float(t_peak_gyro)

    st.info("Tempo sincronizado: t_sync = t − t_pico. Pico ocorre em t=0 em ambos.")

    st.divider()
    st.subheader("Sinais completos (tempo sincronizado)")

    p1, p2, p3 = st.columns(3)
    with p1:
        plot_with_lines(t_kin_sync, y_kin, "Cinemática — tempo_sync × Y (AP)",
                        vlines=[0.0], labels=["pico (t=0)"], xlabel="Tempo sincronizado (s)")
    with p2:
        plot_with_lines(t_kin_sync, z_kin, "Cinemática — tempo_sync × Z (vertical)",
                        vlines=[0.0], labels=["pico (t=0)"], xlabel="Tempo sincronizado (s)")
    with p3:
        plot_with_lines(t_g_sync, g_norm, "Giroscópio — tempo_sync × norma (||gyro||)",
                        vlines=[0.0], labels=["pico (t=0)"], xlabel="Tempo sincronizado (s)")

    # ROI manual
    st.divider()
    st.subheader("Intervalo de interesse (ROI) — Markov/KMeans será aplicado somente aqui")

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

    # ROI overlay
    p1, p2, p3 = st.columns(3)
    with p1:
        plot_with_lines(t_kin_sync, y_kin, "Cinemática Y com ROI",
                        vlines=[0.0, roi_start, roi_end], labels=["pico", "ROI início", "ROI fim"],
                        xlabel="Tempo sincronizado (s)")
    with p2:
        plot_with_lines(t_kin_sync, z_kin, "Cinemática Z com ROI",
                        vlines=[0.0, roi_start, roi_end], labels=["pico", "ROI início", "ROI fim"],
                        xlabel="Tempo sincronizado (s)")
    with p3:
        plot_with_lines(t_g_sync, g_norm, "Giroscópio norma com ROI",
                        vlines=[0.0, roi_start, roi_end], labels=["pico", "ROI início", "ROI fim"],
                        xlabel="Tempo sincronizado (s)")

    # Markov/KMeans por série só no ROI
    st.divider()
    st.subheader("Segmentação automática por série (K-means 6 estados + regra Markov) — 1 segmento por série")

    if KMeans is None or StandardScaler is None:
        st.error("Para esta etapa é necessário scikit-learn (sklearn).")
        st.stop()

    if (roi_end - roi_start) < 2.0:
        st.warning("ROI muito curto. Recomendo >= 2s para baseline=1s + detecção.")
        st.stop()

    t_common = np.arange(roi_start, roi_end, 1.0 / FS_TARGET)
    if len(t_common) < 50:
        st.warning("Poucas amostras no ROI para segmentação estável.")
        st.stop()

    y_common = np.interp(t_common, t_kin_sync, y_kin)
    z_common = np.interp(t_common, t_kin_sync, z_kin)
    g_common = np.interp(t_common, t_g_sync, g_norm)

    seed = st.number_input("Seed do K-means (usado em todas as séries)", min_value=0, value=42, step=1)

    # KMeans states por série (independente)
    states_y = segment_markov_kmeans_1d(t_common, y_common, FS_TARGET, n_states=6, seed=int(seed))
    states_z = segment_markov_kmeans_1d(t_common, z_common, FS_TARGET, n_states=6, seed=int(seed))
    states_g = segment_markov_kmeans_1d(t_common, g_common, FS_TARGET, n_states=6, seed=int(seed))

    # baseline = 1s após início do ROI; regra: 10 baseline -> 5 não-baseline; 1 segmento por série
    base_y, y_start, y_end = detect_single_segment_from_states(
        t_common, states_y, FS_TARGET, baseline_sec=1.0, baseline_run_before=10, other_run_after=5
    )
    base_z, z_start, z_end = detect_single_segment_from_states(
        t_common, states_z, FS_TARGET, baseline_sec=1.0, baseline_run_before=10, other_run_after=5
    )
    base_g, g_start, g_end = detect_single_segment_from_states(
        t_common, states_g, FS_TARGET, baseline_sec=1.0, baseline_run_before=10, other_run_after=5
    )

    st.markdown("### Resultado por série")
    st.write({
        "baseline_state_Y (1s após ROI início)": None if base_y is None else int(base_y),
        "inicio_Y_s": y_start, "fim_Y_s": y_end,
        "baseline_state_Z (1s após ROI início)": None if base_z is None else int(base_z),
        "inicio_Z_s": z_start, "fim_Z_s": z_end,
        "baseline_state_G_norm (1s após ROI início)": None if base_g is None else int(base_g),
        "inicio_G_s": g_start, "fim_G_s": g_end,
    })

    # Overlays: cada série mostra ROI + seu próprio início/fim detectado (se existir)
    def make_lines(roi_start, roi_end, t_start, t_end, prefix):
        v = [roi_start, roi_end]
        lab = ["ROI início", "ROI fim"]
        if (t_start is not None) and (t_end is not None):
            v += [t_start, t_end]
            lab += [f"{prefix} início", f"{prefix} fim"]
        return v, lab

    vy, ly = make_lines(roi_start, roi_end, y_start, y_end, "Y")
    vz, lz = make_lines(roi_start, roi_end, z_start, z_end, "Z")
    vg, lg = make_lines(roi_start, roi_end, g_start, g_end, "G")

    st.divider()
    st.subheader("Sinais com segmento detectado (1 por série)")

    r1, r2, r3 = st.columns(3)
    with r1:
        plot_with_lines(t_kin_sync, y_kin, "Cinemática Y (AP) — 1 segmento detectado em Y",
                        vlines=vy, labels=ly, xlabel="Tempo sincronizado (s)")
    with r2:
        plot_with_lines(t_kin_sync, z_kin, "Cinemática Z — 1 segmento detectado em Z",
                        vlines=vz, labels=lz, xlabel="Tempo sincronizado (s)")
    with r3:
        plot_with_lines(t_g_sync, g_norm, "Giroscópio norma — 1 segmento detectado na norma",
                        vlines=vg, labels=lg, xlabel="Tempo sincronizado (s)")

    # Segmentos recortados detectados (se existirem)
    st.divider()
    st.subheader("Registros segmentados (detectados automaticamente)")

    def plot_detected_segment(name, t_start, t_end, t_full, x_full):
        if t_start is None or t_end is None:
            st.caption(f"{name}: nenhum segmento detectado com os critérios atuais.")
            return
        t_seg, x_seg, _ = segment_by_time(t_full, x_full, t_start, t_end)
        plot_with_lines(t_seg, x_seg, f"{name} — segmento detectado", xlabel="Tempo sincronizado (s)")

    a1, a2, a3 = st.columns(3)
    with a1:
        plot_detected_segment("Cinemática Y", y_start, y_end, t_kin_sync, y_kin)
    with a2:
        plot_detected_segment("Cinemática Z", z_start, z_end, t_kin_sync, z_kin)
    with a3:
        plot_detected_segment("Giroscópio norma", g_start, g_end, t_g_sync, g_norm)

else:
    st.caption("Carregue os dois arquivos para habilitar trigger, sincronização, ROI e segmentação por série.")
