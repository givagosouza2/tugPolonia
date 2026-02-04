# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

st.set_page_config(layout="wide", page_title="Sync Cinemática x Giroscópio")
st.title("Sincronização por salto (Cinemática x Giroscópio) + Segmentação")

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

        col_x = df_kin.columns[0]
        col_y = df_kin.columns[1]
        col_z = df_kin.columns[2]

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
# Trigger (0–20 s) + zeragem do tempo
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

    # zeragem do tempo (pico -> t=0)
    t_kin_sync = t_kin - float(t_peak_kin)
    t_g_sync   = t_g   - float(t_peak_gyro)

    st.info("Aplicado: t_sync = t_original − t_pico. Agora o pico escolhido ocorre em t=0 em ambos.")

    # -------------------------
    # 3 gráficos completos (tempo sincronizado) com marcações de segmento
    # -------------------------
    st.divider()
    st.subheader("Sinais completos (tempo sincronizado) + seleção do segmento")

    # limites possíveis do segmento (interseção do tempo sincronizado)
    tmin_common = float(max(np.min(t_kin_sync), np.min(t_g_sync)))
    tmax_common = float(min(np.max(t_kin_sync), np.max(t_g_sync)))

    if tmax_common <= tmin_common:
        st.error("Não foi possível encontrar faixa temporal comum entre cinemática e giroscópio após a sincronização.")
        st.stop()

    s1, s2 = st.columns(2)
    with s1:
        seg_start = st.number_input(
            "Início do segmento (s) [tempo sincronizado]",
            min_value=tmin_common, max_value=tmax_common, value=max(0.0, tmin_common), step=0.05, format="%.2f"
        )
    with s2:
        seg_end = st.number_input(
            "Fim do segmento (s) [tempo sincronizado]",
            min_value=tmin_common, max_value=tmax_common, value=min(5.0, tmax_common), step=0.05, format="%.2f"
        )

    # garante ordem
    if seg_start > seg_end:
        seg_start, seg_end = seg_end, seg_start

    # plots completos com linhas do segmento + linha do pico (t=0)
    p1, p2, p3 = st.columns(3)

    with p1:
        plot_with_lines(
            t_kin_sync, y_kin,
            "Cinemática — tempo_sync × eixo Y (AP)",
            vlines=[0.0, seg_start, seg_end],
            labels=["pico (t=0)", "início", "fim"],
            xlabel="Tempo sincronizado (s)"
        )

    with p2:
        plot_with_lines(
            t_kin_sync, z_kin,
            "Cinemática — tempo_sync × eixo Z (vertical)",
            vlines=[0.0, seg_start, seg_end],
            labels=["pico (t=0)", "início", "fim"],
            xlabel="Tempo sincronizado (s)"
        )

    with p3:
        plot_with_lines(
            t_g_sync, g_norm,
            "Giroscópio — tempo_sync × norma (||gyro||)",
            vlines=[0.0, seg_start, seg_end],
            labels=["pico (t=0)", "início", "fim"],
            xlabel="Tempo sincronizado (s)"
        )

    # -------------------------
    # Segmentos (abaixo): recorta e plota novamente
    # -------------------------
    st.divider()
    st.subheader("Registros segmentados")

    tY_seg, y_seg, _ = segment_by_time(t_kin_sync, y_kin, seg_start, seg_end)
    tZ_seg, z_seg, _ = segment_by_time(t_kin_sync, z_kin, seg_start, seg_end)
    tG_seg, g_seg, _ = segment_by_time(t_g_sync,   g_norm, seg_start, seg_end)

    q1, q2, q3 = st.columns(3)

    with q1:
        plot_with_lines(
            tY_seg, y_seg,
            "Segmento — Cinemática Y (AP)",
            vlines=[seg_start, seg_end], labels=["início", "fim"],
            xlabel="Tempo sincronizado (s)"
        )

    with q2:
        plot_with_lines(
            tZ_seg, z_seg,
            "Segmento — Cinemática Z (vertical)",
            vlines=[seg_start, seg_end], labels=["início", "fim"],
            xlabel="Tempo sincronizado (s)"
        )

    with q3:
        plot_with_lines(
            tG_seg, g_seg,
            "Segmento — Giroscópio norma (||gyro||)",
            vlines=[seg_start, seg_end], labels=["início", "fim"],
            xlabel="Tempo sincronizado (s)"
        )

    # Resumo
    st.markdown("### Resumo do segmento")
    st.write({
        "inicio_s": float(seg_start),
        "fim_s": float(seg_end),
        "duracao_s": float(seg_end - seg_start),
        "amostras_cinematica": int(len(tY_seg)),
        "amostras_giroscopio": int(len(tG_seg)),
    })

else:
    st.caption("Carregue os dois arquivos para habilitar trigger, sincronização e segmentação.")
