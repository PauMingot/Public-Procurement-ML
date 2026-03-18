# Licitaciones_Creacion_Historicos_con_Lags.py
# Versión final (Corregida + Lags de 3, 9, 15 meses)
import numpy as np
import pandas as pd
import sqlite3
from tqdm.auto import tqdm
import re
import warnings
import random

tqdm.pandas()
warnings.filterwarnings('ignore', category=UserWarning)

# -----------------------
# CONFIG
# -----------------------
RUTA_BASE_DATOS = r"C:\Users\User\Documents\InferIA\LicitacionesCorrecto_Procesada.sqlite"
NOMBRE_TABLA = "LicitacionesCorrecto_Procesada"
RUTA_GUARDADO = r"C:\Users\User\Documents\InferIA"

# --- Columnas Clave ---
GROUP_COL = 'lote_adjudicatario' # El ID de la empresa
DATE_COL = 'fecha_primera_publicacion'

# --- Columnas de Features ---
GEO_COL = 'com_aut_licitacion' 
PRESUPUESTO_COL = 'lote_presupuesto_base_sin_impuestos'
DESCUENTO_COL = 'descuento_promedio'
TIPO_CONTRATO_COL = 'tipo_contrato'

# --- ¡NUEVO! Configuración de Lags ---
WINDOWS_MESES = {
    '3m': '90D',
    '9m': '270D',
    '15m': '450D'
}
MIN_PERIODS = 2 # (Requiere al menos 2 licitaciones en la ventana)
# --- FIN NUEVO ---

# -----------------------
# Helpers
# -----------------------
def clean_col_name(name):
    """Limpia los nombres de columnas para que sean seguros."""
    name = str(name).lower()
    name = re.sub(r'[áäàãâ]', 'a', name)
    name = re.sub(r'[éëèê]', 'e', name)
    name = re.sub(r'[íïìî]', 'i', name)
    name = re.sub(r'[óöòõô]', 'o', name)
    name = re.sub(r'[úüùû]', 'u', name)
    name = re.sub(r'[ñ]', 'n', name)
    name = re.sub(r'[^a-z0-9_]+', '_', name)
    name = name.strip('_')
    return name

# -----------------------
# Load data
# -----------------------
print("Cargando datos desde sqlite...")
conn = sqlite3.connect(RUTA_BASE_DATOS)
df = pd.read_sql_query(f"SELECT * FROM {NOMBRE_TABLA}", conn)
conn.close()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors='coerce')
df.dropna(subset=[DATE_COL, GROUP_COL], inplace=True)
print(f"Filas cargadas y limpias (con fecha y adjudicatario): {len(df)}")

# Guardar las columnas originales para la limpieza final
original_columns = df.columns.tolist()

# -----------------------
# Temporal split
# -----------------------
df = df.sort_values(DATE_COL).reset_index(drop=True)
n_total = len(df)
train_end = int(0.6 * n_total)
val_end = int(0.8 * n_total)

train_df_orig = df.iloc[:train_end].copy().reset_index(drop=True)
val_df_orig = df.iloc[train_end:val_end].copy().reset_index(drop=True)
test_df_orig = df.iloc[val_end:].copy().reset_index(drop=True)

print(f"Train: {len(train_df_orig)}, Val: {len(val_df_orig)}, Test: {len(test_df_orig)}")


# --- ¡BLOQUE NUEVO! ---
# -----------------------------------------------------------------
# FUNCIÓN 0: CALCULAR FEATURES DE "MERCADO" (por Tipo de Contrato)
# -----------------------------------------------------------------
def calcular_features_mercado(df_hist, group_col, target_col, date_col, windows):
    """
    Calcula las estadísticas de rolling del 'mercado' (ej. por tipo_contrato).
    Usa .shift(1) para asegurar que solo se usa el pasado (sin fugas).
    """
    print(f"Calculando estadísticas de mercado para '{group_col}'...")
    df_hist = df_hist.sort_values(date_col)
    
    # 1. Agrupar por día y 'group_col' para tener el descuento medio de ese día
    df_daily_mean = df_hist.groupby([pd.Grouper(key=date_col, freq='D'), group_col])[target_col].mean()
    
    # 2. Resetear índice para tener 'date_col' y 'group_col' como columnas
    df_daily_mean = df_daily_mean.reset_index()

    # 3. Pivotar para tener 'tipo_contrato' como columnas y 'date' como índice
    df_pivot = df_daily_mean.pivot(index=date_col, columns=group_col, values=target_col)
    
    # 4. Rellenar huecos (forward-fill) para que el rolling sea más estable
    df_pivot_filled = df_pivot.fillna(method='ffill')

    # 5. Calcular las medias móviles (rolling) y hacer .shift(1) (ANTI-FUGAS)
    lookup_dfs = []
    for w_name, w_val in windows.items():
        col_prefix = f'mercado_{group_col}_{target_col}_{w_name}'
        # min_periods=2 para que sea un promedio de al menos 2 días
        df_rolled = df_pivot_filled.rolling(w_val, min_periods=MIN_PERIODS).mean().shift(1)
        
        # Renombrar columnas
        df_rolled = df_rolled.rename(columns=lambda x: f"{col_prefix}_{clean_col_name(x)}")
        lookup_dfs.append(df_rolled)
    
    # 6. Unir todas las ventanas (3m, 9m, 15m)
    df_final_lookup = pd.concat(lookup_dfs, axis=1)
    
    # 7. Rellenar NaNs iniciales con -1 (valor centinela)
    df_final_lookup = df_final_lookup.fillna(-1.0)
    
    return df_final_lookup.reset_index()
# --- FIN BLOQUE NUEVO ---


# -----------------------------------------------------------------
# FUNCI�N 1: CALC_HIST_TRAIN (Corregida)
# -----------------------------------------------------------------
def calc_hist_train(df_train, empresa_col, date_col, descuento_col, tipo_col, geo_col, pres_col):
    """
    Calcula features hist�ricas para 'train' usando .shift(1) para evitar fugas.
    """
    df = df_train.sort_values([empresa_col, date_col]).copy()
    g = df.groupby(empresa_col)

    # 1. n_licitaciones_hist (El contador)
    df['n_licitaciones_hist'] = g.cumcount()
    
    n_licit_safe = df['n_licitaciones_hist'].replace(0, 1)

    # 2. dias_desde_ultima_licitacion (Recencia)
    df['last_date'] = g[date_col].shift(1)
    df['dias_desde_ultima_licitacion'] = (df[date_col] - df['last_date']).dt.days
    df['dias_desde_ultima_licitacion'] = df['dias_desde_ultima_licitacion'].fillna(-1).astype(int)
    df = df.drop(columns=['last_date'])

    # 3. descuento_medio_hist
    if descuento_col in df.columns:
        df['descuento_medio_hist'] = g[descuento_col].transform(lambda s: s.expanding().mean().shift(1).fillna(-1.0))
    else:
        df['descuento_medio_hist'] = -1.0
        
    # 4. presupuesto_medio_hist (Tama�o)
    if pres_col in df.columns:
        df['presupuesto_medio_hist'] = g[pres_col].transform(lambda s: s.expanding().mean().shift(1).fillna(-1.0))
    else:
        df['presupuesto_medio_hist'] = -1.0

    # 5. pct_hist_{tipo_contrato} (Especializaci�n)
    if tipo_col in df.columns:
        tipos_list = list(df[tipo_col].dropna().unique())
        if len(tipos_list) > 0:
            for tc in tipos_list:
                df[f'cum_tc_{clean_col_name(tc)}'] = 0
            
            for emp, g_emp in df.groupby(empresa_col, sort=False):
                idx = g_emp.index
                for tc in tipos_list:
                    tc_clean = clean_col_name(tc)
                    mask_emp = (g_emp[tipo_col] == tc).astype(int)
                    cum = mask_emp.cumsum().shift(1).fillna(0).astype(int).values
                    df.loc[idx, f'cum_tc_{tc_clean}'] = cum
            
            for tc in tipos_list:
                tc_clean = clean_col_name(tc)
                pct_hist = df[f'cum_tc_{tc_clean}'] / n_licit_safe
                df[f'pct_hist_{tc_clean}'] = np.where(df['n_licitaciones_hist'] > 0, pct_hist, -1.0)
    
    # 6. pct_hist_com_aut_{geografia} (Localizaci�n)
    if geo_col in df.columns:
        geo_list = list(df[geo_col].dropna().unique())
        if len(geo_list) > 0:
            for geo in geo_list:
                df[f'cum_geo_{clean_col_name(geo)}'] = 0
                
            for emp, g_emp in df.groupby(empresa_col, sort=False):
                idx = g_emp.index
                for geo in geo_list:
                    geo_clean = clean_col_name(geo)
                    mask_emp = (g_emp[geo_col] == geo).astype(int)
                    cum = mask_emp.cumsum().shift(1).fillna(0).astype(int).values
                    df.loc[idx, f'cum_geo_{geo_clean}'] = cum
                    
            for geo in geo_list:
                geo_clean = clean_col_name(geo)
                pct_hist = df[f'cum_geo_{geo_clean}'] / n_licit_safe
                df[f'pct_hist_com_aut_{geo_clean}'] = np.where(df['n_licitaciones_hist'] > 0, pct_hist, -1.0)

    # --- �BLOQUE CORREGIDO! ---
    # 7. Lags de Empresa (Rolling Averages por Empresa)
    print("Calculando lags por empresa (90D, 270D, 450D)...")
    
    # Ponemos la fecha como �ndice para que .rolling('90D') funcione
    df_time = df.set_index(date_col) 
    
    for w_name, w_val in WINDOWS_MESES.items():
        col_name = f'desc_hist_emp_{w_name}'
        
        # 1. Calculamos el rolling (que tiene un �ndice Datetime duplicado)
        rolled_series = df_time.groupby(empresa_col)[descuento_col].rolling(w_val, min_periods=MIN_PERIODS).mean().shift(1)
        
        # 2. �LA CORRECCI�N!
        #    Reseteamos el �ndice de (empresa, fecha) a (fecha)
        #    y usamos .values para asignar por POSICI�N, ignorando el �ndice.
        df[col_name] = rolled_series.reset_index(level=0, drop=True).values
    
    # Rellenamos los NaNs (de las primeras filas de cada empresa) con -1
    for w_name in WINDOWS_MESES.keys():
        df[f'desc_hist_emp_{w_name}'] = df[f'desc_hist_emp_{w_name}'].fillna(-1.0)
    # --- FIN BLOQUE CORREGIDO ---

    return df

# -----------------------------------------------------------------
# FUNCI�N 2: BUILD_HIST_SUMMARY (Corregida)
# -----------------------------------------------------------------
def build_hist_summary(df_hist, empresa_col, date_col, descuento_col, tipo_col, geo_col, pres_col):
    """
    Construye la tabla resumen (SIN SHIFT) para usar con merge_asof.
    """
    df = df_hist.sort_values([empresa_col, date_col]).copy().reset_index(drop=True)
    g = df.groupby(empresa_col)
    
    df['hist_cum_n'] = g.cumcount() + 1
    
    for c in [descuento_col, pres_col]:
        if c in df.columns:
            df[f'hist_cum_sum_{c}'] = g[c].cumsum()

    # pct_hist_{tipo_contrato} (L�gica original, sin cambios)
    if tipo_col in df.columns:
        tipos_list = list(df[tipo_col].dropna().unique())
        for tc in tipos_list:
            df[f'hist_cum_tc_{clean_col_name(tc)}'] = 0
        for emp, g_emp in df.groupby(empresa_col, sort=False):
            idx = g_emp.index
            for tc in tipos_list:
                tc_clean = clean_col_name(tc)
                mask_emp = (g_emp[tipo_col] == tc).astype(int)
                cum_emp = mask_emp.cumsum().values
                df.loc[idx, f'hist_cum_tc_{tc_clean}'] = cum_emp
        
    # pct_hist_com_aut_{geografia} (L�gica original, sin cambios)
    if geo_col in df.columns:
        geo_list = list(df[geo_col].dropna().unique())
        for geo in geo_list:
            df[f'hist_cum_geo_{clean_col_name(geo)}'] = 0
        for emp, g_emp in df.groupby(empresa_col, sort=False):
            idx = g_emp.index
            for geo in geo_list:
                geo_clean = clean_col_name(geo)
                mask_emp = (g_emp[geo_col] == geo).astype(int)
                cum_emp = mask_emp.cumsum().values
                df.loc[idx, f'hist_cum_geo_{geo_clean}'] = cum_emp
    
    # --- �BLOQUE CORREGIDO! ---
    # 7. Lags de Empresa (Rolling Averages por Empresa) - VERSI�N RESUMEN
    print("Calculando lags por empresa (versi�n resumen)...")
    
    # Ponemos la fecha como �ndice para que .rolling('90D') funcione
    df_time = df.set_index(date_col)
    
    for w_name, w_val in WINDOWS_MESES.items():
        col_name = f'hist_roll_emp_{w_name}'
        
        # 1. Calculamos el rolling (sin .shift(1))
        rolled_series = df_time.groupby(empresa_col)[descuento_col].rolling(w_val, min_periods=MIN_PERIODS).mean()
        
        # 2. �LA CORRECCI�N!
        #    Asignamos los .values para ignorar el �ndice duplicado
        df[col_name] = rolled_series.reset_index(level=0, drop=True).values
    
    # Rellenamos NaNs con -1
    for w_name in WINDOWS_MESES.keys():
        df[f'hist_roll_emp_{w_name}'] = df[f'hist_roll_emp_{w_name}'].fillna(-1.0)
    # --- FIN BLOQUE CORREGIDO ---
                
    keep_cols = [empresa_col, date_col, 'hist_cum_n'] + \
                [c for c in df.columns if c.startswith('hist_cum_sum_') or \
                                          c.startswith('hist_cum_tc_') or \
                                          c.startswith('hist_cum_geo_') or \
                                          c.startswith('hist_roll_emp_')] 
                                          
    df_final = df[keep_cols].reset_index(drop=True)
    df_final = df_final.rename(columns={date_col: f"{date_col}_hist"})
                                          
    return df_final

# -----------------------------------------------------------------
# FUNCIÓN 3: APLICAR_HISTORICOS_ASOF (La lógica del merge_asof)
# -----------------------------------------------------------------
def aplicar_historicos_asof(df_para_aplicar, df_resumen, empresa_col, date_col, pres_col, descuento_col):
    """
    Aplica el resumen histórico a un df (val o test) usando merge_asof.
    """
    df_sorted = df_para_aplicar.sort_values([empresa_col, date_col]).reset_index(drop=True)
    hist_summary_sorted = df_resumen.sort_values([empresa_col, f"{date_col}_hist"]).reset_index(drop=True)

    merged_list = []
    
    all_cum_tc_cols = [c for c in hist_summary_sorted.columns if c.startswith('hist_cum_tc_')]
    all_cum_geo_cols = [c for c in hist_summary_sorted.columns if c.startswith('hist_cum_geo_')]
    
    # --- ¡NUEVO! ---
    all_roll_emp_cols = [c for c in hist_summary_sorted.columns if c.startswith('hist_roll_emp_')]
    # --- FIN NUEVO ---
    
    desc_sum_col = f'hist_cum_sum_{descuento_col}'
    pres_sum_col = f'hist_cum_sum_{pres_col}'

    for emp, g in tqdm(df_sorted.groupby(empresa_col, sort=False), desc="Aplicando merge_asof por empresa"):
        hist_sub = hist_summary_sorted[hist_summary_sorted[empresa_col] == emp]
        
        if hist_sub.empty:
            g_res = g.copy()
            g_res['n_licitaciones_hist'] = 0
            g_res['descuento_medio_hist'] = -1.0
            g_res['presupuesto_medio_hist'] = -1.0
            g_res['dias_desde_ultima_licitacion'] = -1
            
            for col in all_cum_tc_cols:
                g_res[f'pct_hist_{col.replace("hist_cum_tc_","")}'] = -1.0
            for col in all_cum_geo_cols:
                g_res[f'pct_hist_com_aut_{col.replace("hist_cum_geo_","")}'] = -1.0
            
            # --- ¡NUEVO! ---
            # Rellenar lags de empresa para empresas nuevas
            for col in all_roll_emp_cols:
                g_res[f'desc_hist_emp_{col.replace("hist_roll_emp_","")}'] = -1.0
            # --- FIN NUEVO ---
            
            merged_list.append(g_res)
            continue
            
        merged = pd.merge_asof(g.sort_values(date_col), hist_sub,
                               left_on=date_col, right_on=f"{date_col}_hist", by=empresa_col, 
                               direction='backward', suffixes=('', '_sufijo_basura'))
        
        merged_res = merged.copy()
        
        merged_res['n_licitaciones_hist'] = merged_res['hist_cum_n'].fillna(0).astype(int)
        n_licit_safe_merge = merged_res['n_licitaciones_hist'].replace(0, 1)
        
        merged_res['dias_desde_ultima_licitacion'] = (merged_res[date_col] - merged_res[f'{date_col}_hist']).dt.days.fillna(-1).astype(int)
            
        merged_res['descuento_medio_hist'] = np.where(merged_res['n_licitaciones_hist'] > 0,
                                                      merged_res[desc_sum_col].fillna(0) / n_licit_safe_merge, -1.0)
            
        merged_res['presupuesto_medio_hist'] = np.where(merged_res['n_licitaciones_hist'] > 0,
                                                        merged_res[pres_sum_col].fillna(0) / n_licit_safe_merge, -1.0)
                
        for col in all_cum_tc_cols:
            tc_clean = col.replace('hist_cum_tc_', '')
            pct_hist = merged_res[col].fillna(0) / n_licit_safe_merge
            merged_res[f'pct_hist_{tc_clean}'] = np.where(merged_res['n_licitaciones_hist'] > 0, pct_hist, -1.0)
                                                         
        for col in all_cum_geo_cols:
            geo_clean = col.replace('hist_cum_geo_', '')
            pct_hist = merged_res[col].fillna(0) / n_licit_safe_merge
            merged_res[f'pct_hist_com_aut_{geo_clean}'] = np.where(merged_res['n_licitaciones_hist'] > 0, pct_hist, -1.0)
        
        # --- ¡NUEVO! ---
        # Asignar los lags de empresa
        for col_name_hist in all_roll_emp_cols:
            col_name_final = f'desc_hist_emp_{col_name_hist.replace("hist_roll_emp_","")}'
            # El merge_asof ya nos ha dado el valor correcto (el último en el pasado)
            merged_res[col_name_final] = merged_res[col_name_hist].fillna(-1.0)
        # --- FIN NUEVO ---
            
        merged_list.append(merged_res)

    return pd.concat(merged_list, axis=0).reset_index(drop=True)

# -----------------------------------------------------------------
# --- SCRIPT PRINCIPAL DE EJECUCIÓN (¡MODIFICADO!) ---
# -----------------------------------------------------------------

# --- ¡NUEVO! PASO 0: CALCULAR LOOKUPS DE MERCADO (POR TIPO_CONTRATO) ---
print("\n--- PASO 0: Creando Lookups de Mercado (Anti-Fugas) ---")

# 0a. Crear lookup de Train (para Train y Val)
lookup_train = calcular_features_mercado(
    train_df_orig,
    group_col=TIPO_CONTRATO_COL,
    target_col=DESCUENTO_COL,
    date_col=DATE_COL,
    windows=WINDOWS_MESES
)

# 0b. Crear lookup de Train+Val (para Test)
df_train_val_orig = pd.concat([train_df_orig, val_df_orig], ignore_index=True)
lookup_train_val = calcular_features_mercado(
    df_train_val_orig,
    group_col=TIPO_CONTRATO_COL,
    target_col=DESCUENTO_COL,
    date_col=DATE_COL,
    windows=WINDOWS_MESES
)
# --- FIN NUEVO ---


# 1. Crear features de EMPRESA para TRAIN (usando .shift(1))
print("\n--- PASO 1: Calculando históricos de EMPRESA para TRAIN (lógica shift(1))...")
train_df = calc_hist_train(
    train_df_orig, 
    empresa_col=GROUP_COL, 
    date_col=DATE_COL,
    descuento_col=DESCUENTO_COL,
    tipo_col=TIPO_CONTRATO_COL,
    geo_col=GEO_COL,
    pres_col=PRESUPUESTO_COL
)

# --- ¡NUEVO! PASO 1b: Añadir features de MERCADO a TRAIN ---
print("Añadiendo features de MERCADO a TRAIN (merge_asof)...")
train_df = pd.merge_asof(
    train_df.sort_values(DATE_COL),
    lookup_train,
    on=DATE_COL,
    direction='backward' # (Encuentra la última estadística de mercado)
).fillna(-1.0) # Rellena los -1 de las primeras filas
# --- FIN NUEVO ---


# 2. Crear el resumen de EMPRESA de TRAIN (para consultar desde VAL)
print("\n--- PASO 2: Construyendo resumen histórico de EMPRESA de TRAIN...")
hist_train_summary = build_hist_summary(
    train_df_orig,
    empresa_col=GROUP_COL,
    date_col=DATE_COL,
    descuento_col=DESCUENTO_COL,
    tipo_col=TIPO_CONTRATO_COL,
    geo_col=GEO_COL,
    pres_col=PRESUPUESTO_COL
)

# 3. Aplicar resumen de EMPRESA de TRAIN a VAL
print("\n--- PASO 3: Aplicando históricos de EMPRESA a VAL (lógica merge_asof)...")
val_df = aplicar_historicos_asof(
    val_df_orig,
    hist_train_summary,
    empresa_col=GROUP_COL,
    date_col=DATE_COL,
    pres_col=PRESUPUESTO_COL,
    descuento_col=DESCUENTO_COL
)

# --- ¡NUEVO! PASO 3b: Añadir features de MERCADO a VAL ---
print("Añadiendo features de MERCADO a VAL (merge_asof)...")
val_df = pd.merge_asof(
    val_df.sort_values(DATE_COL),
    lookup_train, # <-- Usamos el lookup de TRAIN (¡Anti-Fugas!)
    on=DATE_COL,
    direction='backward'
).fillna(-1.0)
# --- FIN NUEVO ---


# 4. Crear el resumen de EMPRESA de TRAIN+VAL (para consultar desde TEST)
print("\n--- PASO 4: Construyendo resumen histórico de EMPRESA de TRAIN+VAL...")
hist_train_val_df = pd.concat([train_df_orig, val_df_orig], axis=0).reset_index(drop=True)
hist_train_val_summary = build_hist_summary(
    hist_train_val_df,
    empresa_col=GROUP_COL,
    date_col=DATE_COL,
    descuento_col=DESCUENTO_COL,
    tipo_col=TIPO_CONTRATO_COL,
    geo_col=GEO_COL,
    pres_col=PRESUPUESTO_COL
)

# 5. Aplicar resumen de EMPRESA de TRAIN+VAL a TEST
print("\n--- PASO 5: Aplicando históricos de EMPRESA a TEST (lógica merge_asof)...")
test_df = aplicar_historicos_asof(
    test_df_orig,
    hist_train_val_summary,
    empresa_col=GROUP_COL,
    date_col=DATE_COL,
    pres_col=PRESUPUESTO_COL,
    descuento_col=DESCUENTO_COL
)

# --- ¡NUEVO! PASO 5b: Añadir features de MERCADO a TEST ---
print("Añadiendo features de MERCADO a TEST (merge_asof)...")
test_df = pd.merge_asof(
    test_df.sort_values(DATE_COL),
    lookup_train_val, # <-- Usamos el lookup de TRAIN+VAL (¡Anti-Fugas!)
    on=DATE_COL,
    direction='backward'
).fillna(-1.0)
# --- FIN NUEVO ---


# -----------------------------------------------------------------
# --- PASO 6: LIMPIEZA FINAL DE COLUMNAS ---
# -----------------------------------------------------------------
print("\n--- PASO 6: Limpiando columnas de cálculo intermedias... ---")

final_features = [
    'n_licitaciones_hist',
    'dias_desde_ultima_licitacion',
    'descuento_medio_hist',
    'presupuesto_medio_hist',
]
# Añadir todas las 'pct_hist_', 'desc_hist_emp_' y 'mercado_'
final_features.extend([col for col in train_df.columns if col.startswith('pct_hist_')])
final_features.extend([col for col in train_df.columns if col.startswith('desc_hist_emp_')]) # <-- NUEVO
final_features.extend([col for col in train_df.columns if col.startswith('mercado_')]) # <-- NUEVO

final_columns_to_keep = original_columns + final_features

keep_cols_train = [col for col in train_df.columns if col in set(final_columns_to_keep)]
keep_cols_val = [col for col in val_df.columns if col in set(final_columns_to_keep)]
keep_cols_test = [col for col in test_df.columns if col in set(final_columns_to_keep)]

train_df = train_df[keep_cols_train]
val_df = val_df[keep_cols_val]
test_df = test_df[keep_cols_test]

print(f"Limpieza completada. Columnas finales en Train: {len(train_df.columns)}")

# 7. Guardar archivos Parquet
print("\n--- PASO 7: Guardando DataFrames procesados en formato Parquet... ---")
# (Guardamos con un nuevo sufijo 'v3_lags')
train_df.to_parquet(RUTA_GUARDADO + r"\train_procesado_v3_lags.parquet")
val_df.to_parquet(RUTA_GUARDADO + r"\val_procesado_v3_lags.parquet")
test_df.to_parquet(RUTA_GUARDADO + r"\test_procesado_v3_lags.parquet")
print("¡Guardado completado en '..._v3_lags.parquet'!")

print("\nScript de creación de históricos (con lags) completado.")