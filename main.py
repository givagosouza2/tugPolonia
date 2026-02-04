# -------------------------
# Processa Cinemática (colunas fixas: 1=X ignora, 2=Y plota, 3=Z trigger+plota) + correção de NaNs
# -------------------------
if kin_file is not None:
    try:
        df_kin = read_table(kin_file)
        st.success(f"Cinemática: {df_kin.shape[0]} linhas × {df_kin.shape[1]} colunas")

        if df_kin.shape[1] < 3:
            raise ValueError("O arquivo de cinemática deve ter pelo menos 3 colunas: X, Y, Z.")

        # FIXO por posição
        col_x = df_kin.columns[0]  # X (não usar/plotar)
        col_y = df_kin.columns[1]  # Y (AP) -> plotar
        col_z = df_kin.columns[2]  # Z (vertical) -> trigger + plotar

        # lê Y e Z como numérico
        y_kin_raw = safe_numeric(df_kin[col_y])
        z_kin_raw = safe_numeric(df_kin[col_z])

        # corrige NaNs/vazios em Y e Z
        y_kin, n_nan_y = fix_nans_1d(y_kin_raw)
        z_kin, n_nan_z = fix_nans_1d(z_kin_raw)

        if (n_nan_y + n_nan_z) > 0:
            st.warning(f"Cinemática: corrigidos NaNs/vazios → Y: {n_nan_y}, Z: {n_nan_z}")

        # tempo a 100 Hz
        t_kin = time_vector(len(df_kin), FS_TARGET)

        # só para transparência (não plota X)
        st.caption(f"Mapeamento cinemática (fixo): X='{col_x}' (ignorado), Y='{col_y}', Z='{col_z}'")

        kin_ready = True

    except Exception as e:
        st.error(f"Erro ao processar cinemática: {e}")
