# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(layout="wide", page_title="Sincronização Cinemática x Giroscópio")

st.title("Sincronização manual por pico de salto (Cinemática x Giroscópio)")

# -------------------------
# Utilidades
# -------------------------
def read_table(uploaded_file) -> pd.DataFrame:
    """Lê CSV/TXT tentando inferir separador. Mantém apenas colunas numéricas quando possível."""
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    # tenta converter para numérico quando possível
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def pick_column(df: pd.DataFrame, label: str, preferred: list[str]):
    cols = list(df.columns)
    # tenta auto-selecionar por nome
    lower_map = {c: str(c).lower() for c in cols}
    for p in preferred:
        for c in cols:
            if p in lower_map[c]:
                return st.selectbox(label, cols, index=cols.index(c))
    return st.selectbox(label, cols, index=0)

def butter_lowpass_filtfilt(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    wn = cutoff_hz / nyq
    b, a = signal.butter(order, wn, btype="lowpass")
    return signal.filtfilt(b, a, x)

def safe_numeric(series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x)
    if mask.sum() < 5:
        raise ValueError("A coluna selecionada não parece numérica (muitos NaN/inf).")
    return x

def time_vector(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=float) / float(fs)

def interpolate_to_fs(t_in: np.ndarray, x_in: np.ndarray, fs_out: float):
    # garante monotonicidade
    order = np.argsort(t_in)
    t_in = t_in[order]
    x_in = x_in[order]

    # remove duplicatas de tempo
    dt = np.diff(t_in)
    keep = np.hstack(([True], dt > 0))
    t_in = t_in[keep]
    x_in = x_in[keep]

    t_out = np.arange(t_in[0], t_in[-1], 1.0 / fs_out)
    x_out = np.interp(t_out, t_in, x_in)
    return t_out, x_out

def find_peak_in_window(t: np.ndarray, x: np.ndarray, t0: float, t1: float):
    m = (t >= t0) & (t <= t1)
    if m.sum() < 3:
        return None, None
    idx_local = np.argmax(x[m])
    idx = np.where(m)[0][0] + idx_local
    return idx, t[idx]

# -------------------------
# Uploads
# -------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("1) Arquivo de Cinemática")
    kin_file = st.file_uploader("Carregue o arquivo de cinemática (.csv/.txt)", type=["csv", "txt"], key="kin")

with colB:
    st.subheader("2) Arquivo de Giroscópio")
    gyro_file = st.file_uploader("Carregue o arquivo de giroscópio (.csv/.txt)", type=["csv", "txt"], key="gyro")

fs_target = 100.0

# -------------------------
# Processa Cinemática
# -------------------------
kin_ready = False
if kin_file is not None:
    try:
        df_kin = read_table(kin_file)
        st.success(f"Cinemática carregada: {df_kin.shape[0]} linhas × {df_kin.shape[1]} colunas")

        c1, c2 = st.columns(2)
        with c1:
            kin_y_col = pick_column(df_kin, "Coluna Y (antero-posterior)", preferred=["y", "ap", "antero", "anterop", "anterior"])
        with c2:
            kin_z_col = pick_column(df_kin, "Coluna Z (vertical)", preferred=["z", "vert", "vertical"])

        y_kin = safe_numeric(df_kin[kin_y_col])
        z_kin = safe_numeric(df_kin[kin_z_col])

        t_kin = time_vector(len(df_kin), fs_target)

        kin_ready = True

    except Exception as e:
        st.error(f"Erro ao processar cinemática: {e}")

# -------------------------
# Processa Giroscópio
# -------------------------
gyro_ready = False
if gyro_file is not None:
    try:
        df_g = read_table(gyro_file)
        st.success(f"Giroscópio carregado: {df_g.shape[0]} linhas × {df_g.shape[1]} colunas")

        st.markdown("**Seleção de colunas do giroscópio**")
        g1, g2, g3, g4 = st.columns([1, 1, 1, 1])

        with g1:
            gx_col = pick_column(df_g, "Coluna gyro X", preferred=["gx", "gyro_x", "x"])
        with g2:
            gy_col = pick_column(df_g, "Coluna gyro Y (vertical p/ salto)", preferred=["gy", "gyro_y", "y"])
        with g3:
            gz_col = pick_column(df_g, "Coluna gyro Z", preferred=["gz", "gyro_z", "z"])
        with g4:
            has_time = st.checkbox("O arquivo do giroscópio tem coluna de tempo?", value=False)

        if has_time:
            tcol = pick_column(df_g, "Coluna de tempo do giroscópio", preferred=["time", "tempo", "t", "timestamp"])
            t_g_in = safe_numeric(df_g[tcol])

            # heurística simples: se parecer ms, converte para s
            if np.nanmedian(np.diff(np.sort(t_g_in))) > 1.0:  # provável ms
                t_g_in = t_g_in / 1000.0
        else:
            fs_g_in = st.number_input("Taxa de aquisição original do giroscópio (Hz)", min_value=1.0, value=100.0, step=1.0)
            t_g_in = time_vector(len(df_g), fs_g_in)

        gx = safe_numeric(df_g[gx_col])
        gy = safe_numeric(df_g[gy_col])
        gz = safe_numeric(df_g[gz_col])

        # detrend
        gx_dt = signal.detrend(gx, type="linear")
        gy_dt = signal.detrend(gy, type="linear")
        gz_dt = signal.detrend(gz, type="linear")

        # interp para 100 Hz
        t_g, gx_i = interpolate_to_fs(t_g_in, gx_dt, fs_target)
        _,   gy_i = interpolate_to_fs(t_g_in, gy_dt, fs_target)
        _,   gz_i = interpolate_to_fs(t_g_in, gz_dt, fs_target)

        # filtro passa-baixa 1.5 Hz
        cutoff = 1.5
        gx_f = butter_lowpass_filtfilt(gx_i, fs_target, cutoff_hz=cutoff, order=4)
        gy_f = butter_lowpass_filtfilt(gy_i, fs_target, cutoff_hz=cutoff, order=4)
        gz_f = butter_lowpass_filtfilt(gz_i, fs_target, cutoff_hz=cutoff, order=4)

        # norma
        g_norm = np.sqrt(gx_f**2 + gy_f**2 + gz_f**2)

        gyro_ready = True

    except Exception as e:
        st.error(f"Erro ao processar giroscópio: {e}")

# -------------------------
# Plots básicos
# -------------------------
if kin_ready or gyro_ready:
    st.divider()
    st.subheader("Visualização dos sinais")

    pcol1, pcol2 = st.columns(2)

    if kin_ready:
        with pcol1:
            st.markdown("**Cinemática (100 Hz) — eixos Y e Z**")
            fig = plt.figure()
            plt.plot(t_kin, y_kin, label="Y (AP)")
            plt.plot(t_kin, z_kin, label="Z (vertical)")
            plt.xlabel("Tempo (s)")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

    if gyro_ready:
        with pcol2:
            st.markdown("**Giroscópio — norma (detrend + interp 100 Hz + LP 1,5 Hz)**")
            fig = plt.figure()
            plt.plot(t_g, g_norm, label="||gyro||")
            plt.xlabel("Tempo (s)")
            plt.legend()
            plt.tight_layout()
            st.pyplot(fig, clear_figure=True)

# -------------------------
# Sincronização manual
# -------------------------
if kin_ready and gyro_ready:
    st.divider()
    st.subheader("Sincronização manual pelo pico do salto")

    st.markdown(
        "1) Escolha uma **janela inicial** para inspecionar o começo dos registros.\n"
        "2) Dentro dessa janela, ajuste o **slider do pico** em cada sinal.\n"
        "3) O app alinha os tempos usando a diferença entre os picos."
    )

    cwin1, cwin2, cwin3 = st.columns([1, 1, 1])
    with cwin1:
        win_sec = st.number_input("Janela a partir do início (s)", min_value=1.0, value=5.0, step=0.5)
    with cwin2:
        kin_use = st.selectbox("Usar qual eixo da cinemática para detectar salto?", ["Z (vertical)", "Y (AP)"])
    with cwin3:
        gyro_use = st.selectbox("Usar qual sinal do giroscópio para detectar salto?", ["Norma ||gyro||", "Eixo Y (filtrado)"])

    # define sinal de sincronização
    kin_sync = z_kin if kin_use.startswith("Z") else y_kin
    gyro_sync = g_norm if gyro_use.startswith("Norma") else gy_f

    # janela (início)
    t0, t1 = 0.0, float(win_sec)

    # mostra zoom da janela
    z1, z2 = st.columns(2)
    with z1:
        st.markdown("**Zoom — cinemática (janela inicial)**")
        fig = plt.figure()
        m = (t_kin >= t0) & (t_kin <= t1)
        plt.plot(t_kin[m], kin_sync[m])
        plt.xlabel("Tempo (s)")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with z2:
        st.markdown("**Zoom — giroscópio (janela inicial)**")
        fig = plt.figure()
        m = (t_g >= t0) & (t_g <= t1)
        plt.plot(t_g[m], gyro_sync[m])
        plt.xlabel("Tempo (s)")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Escolha manual do pico do salto")

    # sliders para escolher o pico (tempo)
    s1, s2 = st.columns(2)
    with s1:
        peak_t_kin = st.slider("Tempo do pico (cinemática)", min_value=0.0, max_value=float(win_sec), value=min(1.0, float(win_sec)))
    with s2:
        peak_t_gyro = st.slider("Tempo do pico (giroscópio)", min_value=0.0, max_value=float(win_sec), value=min(1.0, float(win_sec)))

    # pega índice mais próximo
    idx_kin = int(np.argmin(np.abs(t_kin - peak_t_kin)))
    idx_gyro = int(np.argmin(np.abs(t_g - peak_t_gyro)))

    # offset para alinhar: queremos que o pico do gyro caia no pico da cinemática
    offset = t_g[idx_gyro] - t_kin[idx_kin]
    t_g_aligned = t_g - offset

    st.info(f"Offset aplicado ao giroscópio: {offset:.4f} s (gyro deslocado por -offset).")

    # gráfico sincronizado
    st.markdown("### Sinais sincronizados (cinemática vs giroscópio)")
    fig = plt.figure()
    plt.plot(t_kin, kin_sync, label=f"Cinemática ({kin_use})")
    plt.plot(t_g_aligned, gyro_sync, label=f"Giroscópio ({gyro_use})", alpha=0.9)
    plt.axvline(t_kin[idx_kin], linestyle="--", label="Pico cinemática")
    plt.axvline(t_g_aligned[idx_gyro], linestyle="--", label="Pico giroscópio (alinhado)")
    plt.xlabel("Tempo sincronizado (s)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # opcional: recorte sincronizado a partir do pico
    st.markdown("### Recorte a partir do pico (opcional)")
    post_sec = st.number_input("Duração após o pico (s)", min_value=1.0, value=10.0, step=1.0)
    t_start = t_kin[idx_kin]
    t_end = t_start + float(post_sec)

    fig = plt.figure()
    mk = (t_kin >= t_start) & (t_kin <= t_end)
    mg = (t_g_aligned >= t_start) & (t_g_aligned <= t_end)
    plt.plot(t_kin[mk], kin_sync[mk], label="Cinemática (recorte)")
    plt.plot(t_g_aligned[mg], gyro_sync[mg], label="Giroscópio (recorte)", alpha=0.9)
    plt.xlabel("Tempo sincronizado (s)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

else:
    st.caption("Carregue os dois arquivos para habilitar a sincronização manual.")

