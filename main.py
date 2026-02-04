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
CUTOFF_HZ = 1.5

# -------------------------
# Helpers
# -------------------------
def read_table(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file, sep=None, engine="python")
    return df

def safe_numeric(series) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if np.isfinite(x).sum() < 5:
        raise ValueError("A coluna selecionada não parece numérica (muitos NaN/inf).")
    return x

def fix_nans_1d(x: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Resolve NaN/vazios:
    - interpola NaNs internos (linear)
    - preenche bordas (bfill/ffill)
    """
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

def plot_with_vline(t, x, title, vline=None, vlabel="pico"):
    fig = plt.figure()
    plt.plot(t, x)
    if vline is not None:
        plt.axvline(vline, linestyle="--", label=vlabel)
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
    kin_file = st.file_uploader(
        "Carregue o arquivo de cinemática (.csv/.txt)", type=["csv", "txt"], key="kin"
    )

with colB:
    st.subheader("2) Giroscópio")
    gyro_file = st.file_uploader(
        "Carregue o arquivo do giroscópio (.csv/.txt)", type=["csv", "txt"], key="gyro"
    )

kin_ready = False
gyro_ready = False

# -------------------------
# Processa Cinemática (fixo por posição: 1=X ignora, 2=Y plota, 3=Z trigger+plota)
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
# Processa Giroscópio (tempo = 1ª coluna) + detrend + interp 100 Hz + LP 1.5 Hz + norma
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

        # seleção dos eixos (colunas 2..)
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

        # detrend
        gx_dt = signal.detrend(gx, type="linear")
        gy_dt = signal.detrend(gy, type="linear")
        gz_dt = signal.detrend(gz, type="linear")

        # interp p/ 100 Hz
        t_g, gx_i = interpolate_to_fs(t_g_in, gx_dt, FS_TARGET)
        _,   gy_i = interpolate_to_fs(t_g_in, gy_dt, FS_TARGET)
        _,   gz_i = interpolate_to_fs(t_g_in, gz_dt, FS_TARGET)

        # LP 1.5 Hz
        gx_f = butter_lowpass_filtfilt(gx_i, FS_TARGET, cutoff_hz=CUTOFF_HZ, order=4)
        gy_f = butter_lowpass_filtfilt(gy_i, FS_TARGET, cutoff_hz=CUTOFF_HZ, order=4)
        gz_f = butter_lowpass_filtfilt(gz_i, FS_TARGET, cutoff_hz=CUTOFF_HZ, order=4)

        # norma
        g_norm = np.sqrt(gx_f**2 + gy_f**2 + gz_f**2)

        gyro_ready = True
    except Exception as e:
        st.error(f"Erro ao processar giroscópio: {e}")

# -------------------------
# Trigger (0–20 s): plota Z da cinemática e Y do giroscópio + inputs numéricos + zeragem do tempo
# -------------------------
if kin_ready and gyro_ready:
    st.divider()
    st.subheader("Ajuste do trigger (0–20 s) — escolha do pico e zeragem do tempo")

    tmax = float(TRIGGER_VIEW_SEC)
    mk = (t_kin >= 0) & (t_kin <= tmax)
    mg = (t_g >= 0) & (t_g <= tmax)

    # inputs numéricos
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

    # plots 0–20s com linha vertical
    cL, cR = st.columns(2)
    with cL:
        plot_with_vline(
            t_kin[mk], z_kin[mk],
            "Cinemática — eixo Z (0–20 s)",
            vline=float(t_peak_kin), vlabel="pico escolhido"
        )
    with cR:
        plot_with_vline(
            t_g[mg], gy_f[mg],
            "Giroscópio — eixo Y (filtrado) (0–20 s)",
            vline=float(t_peak_gyro), vlabel="pico escolhido"
        )

    # zeragem do tempo (pico -> t=0)
    t_kin_sync = t_kin - float(t_peak_kin)
    t_g_sync   = t_g   - float(t_peak_gyro)

    st.info("Aplicado: t_sync = t_original − t_pico. Agora o pico escolhido ocorre em t=0 em ambos.")

    # -------------------------
    # 3 colunas: cinemática tempo_sync×Y, cinemática tempo_sync×Z, giroscópio tempo_sync×norma
    # -------------------------
    st.divider()
    st.subheader("Sinais completos (tempo sincronizado)")

    p1, p2, p3 = st.columns(3)

    with p1:
        fig = plt.figure()
        plt.plot(t_kin_sync, y_kin)
        plt.axvline(0, linestyle="--", label="pico (t=0)")
        plt.title("Cinemática — tempo_sync × eixo Y (AP)")
        plt.xlabel("Tempo sincronizado (s)")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with p2:
        fig = plt.figure()
        plt.plot(t_kin_sync, z_kin)
        plt.axvline(0, linestyle="--", label="pico (t=0)")
        plt.title("Cinemática — tempo_sync × eixo Z (vertical)")
        plt.xlabel("Tempo sincronizado (s)")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    with p3:
        fig = plt.figure()
        plt.plot(t_g_sync, g_norm)
        plt.axvline(0, linestyle="--", label="pico (t=0)")
        plt.title("Giroscópio — tempo_sync × norma (||gyro||)")
        plt.xlabel("Tempo sincronizado (s)")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

    st.markdown("### Resultado do alinhamento")
    st.write({
        "t_pico_cinematica_s": float(t_peak_kin),
        "t_pico_giroscopio_s": float(t_peak_gyro),
        "delta_picos_s (gyro - kin)": float(t_peak_gyro - t_peak_kin),
    })

else:
    st.caption("Carregue os dois arquivos para habilitar o trigger e os gráficos sincronizados.")
