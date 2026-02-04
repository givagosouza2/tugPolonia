# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(layout="wide", page_title="Sync Cinemática x Giroscópio")
st.title("Sincronização por salto (Cinemática x Giroscópio)")

FS_TARGET = 100.0
TRIGGER_VIEW_SEC = 20.0

# -------------------------
# Helpers
# -------------------------
def read_table(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def safe_numeric(series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(x).sum() < 5:
        raise ValueError("A coluna selecionada não parece numérica (muitos NaN/inf).")
    return x

def time_vector(n: int, fs: float) -> np.ndarray:
    return np.arange(n, dtype=float) / float(fs)

def interpolate_to_fs(t_in: np.ndarray, x_in: np.ndarray, fs_out: float):
    order = np.argsort(t_in)
    t_in = t_in[order]
    x_in = x_in[order]

    # remove tempos repetidos
    dt = np.diff(t_in)
    keep = np.hstack(([True], dt > 0))
    t_in = t_in[keep]
    x_in = x_in[keep]

    # remove NaNs
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

def guess_time_unit_and_convert_to_seconds(t: np.ndarray) -> np.ndarray:
    t = t.astype(float)
    t_sorted = np.sort(t[np.isfinite(t)])
    if len(t_sorted) < 5:
        return t
    dt_med = np.nanmedian(np.diff(t_sorted))
    # se dt mediano > 1, provavelmente ms
    if dt_med > 1.0:
        return t / 1000.0
    return t

def nearest_index(t: np.ndarray, tt: float) -> int:
    return int(np.argmin(np.abs(t - tt)))

def plot_line(t, x, title, vline=None, vline_label=None):
    fig = plt.figure()
    plt.plot(t, x)
    if vline is not None:
        plt.axvline(vline, linestyle="--", label=vline_label if vline_label else "pico")
        if vline_label:
            plt.legend()
    plt.title(title)
    plt.xlabel("Tempo (s)")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

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
        st.success(f"Cinemática: {df_kin.shape[0]} linhas × {df_kin.shape[1]} colunas")

        c1, c2 = st.columns(2)
        with c1:
            kin_y_col = st.selectbox("Coluna Y (antero-posterior) — cinemática", list(df_kin.columns), index=0)
        with c2:
            kin_z_col = st.selectbox("Coluna Z (vertical) — cinemática", list(df_kin.columns), index=min(1, len(df_kin.columns)-1))

        y_kin = safe_numeric(df_kin[kin_y_col])
        z_kin = safe_numeric(df_kin[kin_z_col])
        t_kin = time_vector(len(df_kin), FS_TARGET)

        kin_ready = True
    except Exception as e:
        st.error(f"Erro ao processar cinemática: {e}")

# -------------------------
# Giroscópio (tempo = 1ª coluna)
# -------------------------
if gyro_file is not None:
    try:
        df_g = read_table(gyro_file)
        st.success(f"Giroscópio: {df_g.shape[0]} linhas × {df_g.shape[1]} colunas")

        cols_g = list(df_g.columns)
        if len(cols_g) < 4:
            raise ValueError("O arquivo do giroscópio deve ter pelo menos 4 colunas: tempo + 3 eixos.")

        time_col = cols_g[0]  # FIXO
        t_g_in = safe_numeric(df_g[time_col])
        t_g_in = guess_time_unit_and_convert_to_seconds(t_g_in)

        g1, g2, g3 = st.columns(3)
        with g1:
            gx_col = st.selectbox("Coluna gyro X", cols_g[1:], index=0)
        with g2:
            gy_col = st.selectbox("Coluna gyro Y (vertical p/ salto)", cols_g[1:], index=min(1, len(cols_g[1:])-1))
        with g3:
            gz_col = st.selectbox("Coluna gyro Z", cols_g[1:], index=min(2, len(cols_g[1:])-1))

        gx = safe_numeric(df_g[gx_col])
        gy = safe_numeric(df_g[gy_col])
        gz = safe_numeric(df_g[gz_col])

        # detrend
        gx_dt = signal.detrend(gx, type="linear")
        gy_dt = signal.detrend(gy, type="linear")
        gz_dt = signal.detrend(gz, type="linear")

        # interp p/ 100 Hz
        t_g, gx_i = interpolate_to_fs(t_g_in, gx_dt, FS_TARGET)
        _,   gy_i = interpolate_to_fs(t_g_in, gy_dt, FS_TARGET)
        _,   gz_i = interpolate_to_fs(t_g_in, gz_dt, FS_TARGET)

        # LP 1.5 Hz
        cutoff = 1.5
        gx_f = butter_lowpass_filtfilt(gx_i, FS_TARGET, cutoff_hz=cutoff, order=4)
        gy_f = butter_lowpass_filtfilt(gy_i, FS_TARGET, cutoff_hz=cutoff, order=4)
        gz_f = butter_lowpass_filtfilt(gz_i, FS_TARGET, cutoff_hz=cutoff, order=4)

        g_norm = np.sqrt(gx_f**2 + gy_f**2 + gz_f**2)

        gyro_ready = True
    except Exception as e:
        st.error(f"Erro ao processar giroscópio: {e}")

# -------------------------
# Trigger view + sliders + sincronização
# -------------------------
if kin_ready and gyro_ready:
    st.divider()
    st.subheader("Trigger (0–20 s) + ajuste manual do pico")

    # recortes 0–20 s
    tmax = float(TRIGGER_VIEW_SEC)
    mk = (t_kin >= 0) & (t_kin <= tmax)
    mg = (t_g >= 0) & (t_g <= tmax)

    # escolha do eixo cinemática para trigger (mantém padrão Z)
    trig_col1, trig_col2, trig_col3 = st.columns([1.2, 1.2, 1.6])
    with trig_col1:
        kin_trigger_axis = st.selectbox("Trigger na cinemática", ["Z (vertical)", "Y (AP)"], index=0)
    with trig_col2:
        gyro_trigger_signal = st.selectbox("Trigger no giroscópio", ["Y (filtrado)", "Norma ||gyro||"], index=0)
    with trig_col3:
        window_post = st.number_input("Janela após o pico para visualizar sincronizado (s)", min_value=1.0, value=10.0, step=1.0)

    kin_trig = z_kin if kin_trigger_axis.startswith("Z") else y_kin
    gyro_trig = gy_f if gyro_trigger_signal.startswith("Y") else g_norm

    # plots trigger (lado a lado)
    left, right = st.columns(2)
    with left:
        plot_line(t_kin[mk], kin_trig[mk], f"Cinemática — {kin_trigger_axis} (0–20 s)")
    with right:
        plot_line(t_g[mg], gyro_trig[mg], f"Giroscópio — {gyro_trigger_signal} (0–20 s)")

    # sliders de pico
    st.markdown("### Selecione o tempo do pico do salto em cada sinal")
    s1, s2 = st.columns(2)
    with s1:
        peak_t_kin = st.slider(
            "Tempo do pico (cinemática)",
            min_value=0.0, max_value=tmax,
            value=min(1.0, tmax), step=1.0/FS_TARGET
        )
    with s2:
        peak_t_gyro = st.slider(
            "Tempo do pico (giroscópio)",
            min_value=0.0, max_value=tmax,
            value=min(1.0, tmax), step=1.0/FS_TARGET
        )

    idx_kin = nearest_index(t_kin, peak_t_kin)
    idx_gyro = nearest_index(t_g, peak_t_gyro)

    # offset: alinhar pico do giroscópio no pico da cinemática
    offset = t_g[idx_gyro] - t_kin[idx_kin]
    t_g_aligned = t_g - offset

    st.info(f"Offset estimado: {offset:.4f} s  → tempo do giroscópio foi deslocado por {-offset:.4f} s para alinhar os picos.")

    # gráfico sincronizado (janela em torno do pico)
    t0 = t_kin[idx_kin]
    t1 = t0 + float(window_post)

    mks = (t_kin >= t0) & (t_kin <= t1)
    mgs = (t_g_aligned >= t0) & (t_g_aligned <= t1)

    st.markdown("### Visualização sincronizada (após o pico)")
    fig = plt.figure()
    plt.plot(t_kin[mks], kin_trig[mks], label=f"Cinemática ({kin_trigger_axis})")
    plt.plot(t_g_aligned[mgs], gyro_trig[mgs], label=f"Giroscópio ({gyro_trigger_signal})", alpha=0.9)
    plt.axvline(t0, linestyle="--", label="Pico (referência)")
    plt.xlabel("Tempo sincronizado (s)")
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    # opcional: exportar offset
    st.markdown("### Resultado")
    st.write(
        {
            "pico_cinematica_s": float(t_kin[idx_kin]),
            "pico_giroscopio_s": float(t_g[idx_gyro]),
            "offset_s (gyro - kin)": float(offset),
        }
    )

# -------------------------
# 3 colunas com gráficos completos (como você pediu)
# -------------------------
if kin_ready and gyro_ready:
    st.divider()
    st.subheader("Sinais completos (3 colunas)")

    p1, p2, p3 = st.columns(3)

    with p1:
        fig = plt.figure()
        plt.plot(t_kin, y_kin)
        plt.title("Cinemática — tempo × eixo Y (AP)")
        plt.xlabel("Tempo (s)")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with p2:
        fig = plt.figure()
        plt.plot(t_kin, z_kin)
        plt.title("Cinemática — tempo × eixo Z (vertical)")
        plt.xlabel("Tempo (s)")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with p3:
        fig = plt.figure()
        plt.plot(t_g, g_norm)
        plt.title("Giroscópio — tempo × norma (||gyro||)")
        plt.xlabel("Tempo (s)")
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

else:
    st.caption("Carregue os dois arquivos para habilitar o trigger, a sincronização e os 3 gráficos.")
