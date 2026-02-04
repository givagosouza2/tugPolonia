# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# KMeans + scaler
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    KMeans = None
    StandardScaler = None

st.set_page_config(layout="wide", page_title="Sync + Segmentação Markov/KMeans")
st.title("Sincronização por salto (Cinemática x Giroscópio) + Segmentação automática (K-means/Markov)")

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
            plt.legend()
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

def detect_segments_from_states(
    t: np.ndarray,
    states: np.ndarray,
    fs: float,
    baseline_seconds: float = 2.0,
    baseline_run_before: int = 10,
    other_run_after: int = 5,
    other_run_before_end: int = 10,
    baseline_run_after_end: int = 5,
):
    """
    Retorna:
      baseline_state, segments=[(t_start, t_end), ...]
    Regras:
      baseline_state = modo nos primeiros baseline_seconds
      start: 10 baseline seguidos de 5 não-baseline
      end:   10 não-baseline seguidos de 5 baseline  (simétrico)
    """
    n0 = int(round(baseline_seconds * fs))
    n0 = min(n0, len(states))
    if n0 < max(baseline_run_before, other_run_after) + 1:
        return None, []

    baseline_state = mode_int(states[:n0])

    segments = []
    i = baseline_run_before

    # helper para checar runs
    def is_run(arr, idx, length, value=None, not_value=None):
        if idx < 0 or idx + length > len(arr):
            return False
        window = arr[idx: idx + length]
        if value is not None:
            return np.all(window == value)
        if not_value is not None:
            return np.all(window != not_value)
        return False

    # detecta múltiplos segmentos: baseline -> nonbaseline -> baseline -> ...
    while i < len(states) - max(other_run_after, baseline_run_after_end) - 1:
        # procura START
        found_start = False
        while i < len(states) - other_run_after - 1:
            if is_run(states, i - baseline_run_before, baseline_run_before, value=baseline_state) and \
               is_run(states, i, other_run_after, not_value=baseline_state):
                start_idx = i
                found_start = True
                break
            i += 1

        if not found_start:
            break

        # a partir do start, procura END (retorno ao baseline)
        j = start_idx + other_run_before_end
        found_end = False
        while j < len(states) - baseline_run_after_end - 1:
            if is_run(states, j - other_run_before_end, other_run_before_end, not_value=baseline_state) and \
               is_run(states, j, baseline_run_after_end, value=baseline_state):
                end_idx = j  # primeiro baseline após o bloco de não-baseline
                found_end = True
                break
            j += 1

        if found_end:
            segments.append((float(t[start_idx]), float(t[end_idx])))
            i = end_idx + baseline_run_after_end  # pula pra frente
        else:
            # segmento aberto até o fim
            segments.append((float(t[start_idx]), float(t[-1])))
            break

    return baseline_state, segments

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
# Cinemática (fixo por posição: 1=X ignora, 2=Y plota, 3=Z trigger+plota)
# -------------------------
if kin_file is not None:
    try:
        df_kin = read_table(kin_file)
        st.success(f"Cinemática: {df_kin.shape[0]} linhas × {df_kin.shape[1]} colunas")
        if df_kin.shape[1] < 3:
            raise ValueError("O arquivo de cinemática deve ter pelo menos 3 colunas: X, Y, Z.")

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
        st.caption(f"Mapeamento cinemática (fixo): X='{col_x}' (ignorado), Y='{col_y}', Z='{col_z}'")
        kin_ready = True
    except Exception as e:
        st.error(f"Erro ao processar cinemática: {e}")

# -------------------------
# Giroscópio (tempo = 1ª coluna) + detrend + interp 100 Hz + LP 1.5 Hz + norma
# -------------------------
if gyro_file is not None:
    try:
        df_g = read_table(gyro_file)
        st.success(f"Giroscópio: {df_g.shape[0]} linhas × {df_g.shape[1]} colunas")

        cols_g = list(df_g.columns)
        if len(cols_g) < 4:
            raise ValueError("O arquivo do giroscópio deve ter pelo menos 4 colunas: tempo + 3 eixos.")

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
# Trigger manual (0–20 s) + zeragem do tempo
# -------------------------
if kin_ready and gyro_ready:
    st.divider()
    st.subheader("Ajuste do trigger (0–20 s) — escolha do pico e zeragem do tempo")

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
            t_kin[mk], z_kin[mk],
            "Cinemática — eixo Z (0–20 s)",
            vlines=[float(t_peak_kin)], labels=["pico"]
        )
    with cR:
        plot_with_lines(
            t_g[mg], gy_f[mg],
            "Giroscópio — eixo Y (filtrado) (0–20 s)",
            vlines=[float(t_peak_gyro)], labels=["pico"]
        )

    # zeragem do tempo
    t_kin_sync = t_kin - float(t_peak_kin)
    t_g_sync   = t_g   - float(t_peak_gyro)

    st.info("Aplicado: t_sync = t_original − t_pico. Agora o pico escolhido ocorre em t=0 em ambos.")

    # -------------------------
    # Segmentação automática por KMeans/Markov
    # -------------------------
    st.divider()
    st.subheader("Segmentação automática (K-means 6 estados + regra Markov)")

    if KMeans is None or StandardScaler is None:
        st.error("Falta scikit-learn no ambiente (sklearn). Instale para usar KMeans/StandardScaler.")
        st.stop()

    # parâmetros (fixos conforme seu critério)
    k_states = 6
    baseline_seconds = 2.0
    baseline_run_before = 10
    other_run_after = 5

    # Para fim do segmento (simétrico)
    other_run_before_end = 10
    baseline_run_after_end = 5

    # Construir eixo temporal comum (interseção)
    t0_common = max(np.min(t_kin_sync), np.min(t_g_sync))
    t1_common = min(np.max(t_kin_sync), np.max(t_g_sync))
    if t1_common <= t0_common:
        st.error("Não há faixa temporal comum entre cinemática e giroscópio após sincronização.")
        st.stop()

    t_common = np.arange(t0_common, t1_common, 1.0 / FS_TARGET)

    # Interpolar sinais no eixo comum
    y_common = np.interp(t_common, t_kin_sync, y_kin)
    z_common = np.interp(t_common, t_kin_sync, z_kin)
    g_common = np.interp(t_common, t_g_sync, g_norm)

    # Features para KMeans (você pode trocar aqui se quiser: só gyro, só z, etc.)
    Xfeat = np.column_stack([y_common, z_common, g_common])

    # padronização
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xfeat)

    # KMeans
    seed = st.number_input("Seed do K-means", min_value=0, value=42, step=1)
    km = KMeans(n_clusters=k_states, random_state=int(seed), n_init=20)
    states = km.fit_predict(Xs).astype(int)

    baseline_state, segments = detect_segments_from_states(
        t_common, states, FS_TARGET,
        baseline_seconds=baseline_seconds,
        baseline_run_before=baseline_run_before,
        other_run_after=other_run_after,
        other_run_before_end=other_run_before_end,
        baseline_run_after_end=baseline_run_after_end
    )

    if baseline_state is None:
        st.warning("Não foi possível determinar baseline/segmentos (registro muito curto ou dados insuficientes).")
        segments = []

    st.write({
        "baseline_state (modo nos 2s iniciais)": None if baseline_state is None else int(baseline_state),
        "n_segmentos_detectados": int(len(segments))
    })

    # Overlay dos segmentos nos 3 gráficos (tempo sincronizado)
    st.divider()
    st.subheader("Sinais completos (tempo sincronizado) com segmentos detectados")

    # linhas verticais: pico=0 + inícios/fins
    vlines = [0.0]
    labels = ["pico (t=0)"]
    for i, (ts, te) in enumerate(segments, start=1):
        vlines += [ts, te]
        labels += [f"início {i}", f"fim {i}"]

    p1, p2, p3 = st.columns(3)
    with p1:
        plot_with_lines(
            t_kin_sync, y_kin,
            "Cinemática — tempo_sync × eixo Y (AP)",
            vlines=vlines, labels=labels,
            xlabel="Tempo sincronizado (s)"
        )
    with p2:
        plot_with_lines(
            t_kin_sync, z_kin,
            "Cinemática — tempo_sync × eixo Z (vertical)",
            vlines=vlines, labels=labels,
            xlabel="Tempo sincronizado (s)"
        )
    with p3:
        plot_with_lines(
            t_g_sync, g_norm,
            "Giroscópio — tempo_sync × norma (||gyro||)",
            vlines=vlines, labels=labels,
            xlabel="Tempo sincronizado (s)"
        )

    # Mostrar segmentos recortados abaixo
    st.divider()
    st.subheader("Registros segmentados (automático)")

    if len(segments) == 0:
        st.caption("Nenhum segmento foi detectado com os critérios atuais.")
    else:
        # tabela
        seg_df = pd.DataFrame(
            [{"segmento": i+1, "inicio_s": s, "fim_s": e, "duracao_s": e - s} for i, (s, e) in enumerate(segments)]
        )
        st.dataframe(seg_df, use_container_width=True)

        # plot cada segmento (opção: escolher qual)
        seg_ids = [f"Segmento {i+1}" for i in range(len(segments))]
        choice = st.selectbox("Selecione um segmento para visualizar", seg_ids, index=0)
        idx = seg_ids.index(choice)
        seg_start, seg_end = segments[idx]

        tY_seg, y_seg, _ = segment_by_time(t_common, y_common, seg_start, seg_end)
        tZ_seg, z_seg, _ = segment_by_time(t_common, z_common, seg_start, seg_end)
        tG_seg, g_seg, _ = segment_by_time(t_common, g_common, seg_start, seg_end)

        q1, q2, q3 = st.columns(3)
        with q1:
            plot_with_lines(
                tY_seg, y_seg,
                f"{choice} — Cinemática Y (AP)",
                vlines=[seg_start, seg_end], labels=["início", "fim"],
                xlabel="Tempo sincronizado (s)"
            )
        with q2:
            plot_with_lines(
                tZ_seg, z_seg,
                f"{choice} — Cinemática Z (vertical)",
                vlines=[seg_start, seg_end], labels=["início", "fim"],
                xlabel="Tempo sincronizado (s)"
            )
        with q3:
            plot_with_lines(
                tG_seg, g_seg,
                f"{choice} — Giroscópio norma (||gyro||)",
                vlines=[seg_start, seg_end], labels=["início", "fim"],
                xlabel="Tempo sincronizado (s)"
            )

else:
    st.caption("Carregue os dois arquivos para habilitar trigger, sincronização e segmentação.")
