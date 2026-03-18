# Licitaciones_Creacion_Historicos_FINAL.py
# Versión final (Corregida: ZeroDivisionError + KeyError de fecha)
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
# ¡CRÍTICO! Eliminar filas sin fecha o sin empresa
df.dropna(subset=[DATE_COL, GROUP_COL], inplace=True)
print(f"Filas cargadas y limpias (con fecha y adjudicatario): {len(df)}")

# ==============================================================================
# 🧹 LIMPIEZA MAESTRA DE CALIDAD (NUEVO BLOQUE)
# ==============================================================================
# Eliminamos AHORA los datos basura para que no contaminen los cálculos históricos.
# Si una empresa tiene un Acuerdo Marco (100% baja), eso NO debe contar para su media.
# ==============================================================================
print("\n--- APLICANDO FILTRO DE CALIDAD PREVIO (ELIMINAR RUIDO) ---")

TARGET_COL_LIMPIEZA = 'lote_importe_adjudicacion_sin_impuestos' # Nombre estándar en tu BD

if TARGET_COL_LIMPIEZA in df.columns and PRESUPUESTO_COL in df.columns:
    # 1. Calcular baja temporal
    presu_safe = df[PRESUPUESTO_COL].replace(0, 0.01)
    df['baja_pct_temp'] = (df[PRESUPUESTO_COL] - df[TARGET_COL_LIMPIEZA]) / presu_safe
    
    # 2. Definir basura: Bajas > 90% (Acuerdos marco) O Adjudicaciones <= 0
    condicion_basura = (df['baja_pct_temp'] > 0.90) | (df[TARGET_COL_LIMPIEZA] <= 0)
    
    # 3. Filtrar
    n_antes = len(df)
    df = df[~condicion_basura].copy()
    n_despues = len(df)
    
    print(f"✅ Se han eliminado {n_antes - n_despues} filas ({((n_antes-n_despues)/n_antes):.2%}) de BASURA.")
    print("   (Acuerdos Marco, Errores de escala, Adjudicaciones 0€)")
    print("   Ahora los históricos (medias) serán reales y fiables.")
    
    df.drop(columns=['baja_pct_temp'], inplace=True)
else:
    print("⚠️ ADVERTENCIA: No se encontraron las columnas para limpiar. Revisa los nombres.")

# ==============================================================================

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

# -----------------------------------------------------------------
# FUNCIÓN 1: CALC_HIST_TRAIN (La lógica del .shift(1))
# -----------------------------------------------------------------
def calc_hist_train(df_train, empresa_col, date_col, descuento_col, tipo_col, geo_col, pres_col):
    """
    Calcula features históricas para 'train' usando .shift(1) para evitar fugas.
    """
    df = df_train.sort_values([empresa_col, date_col]).copy()
    g = df.groupby(empresa_col)

    # 1. n_licitaciones_hist (El contador)
    df['n_licitaciones_hist'] = g.cumcount()
    
    # Denominador "seguro" para evitar ZeroDivisionError
    n_licit_safe = df['n_licitaciones_hist'].replace(0, 1)

    # 2. dias_desde_ultima_licitacion (Recencia)
    df['last_date'] = g[date_col].shift(1)
    df['dias_desde_ultima_licitacion'] = (df[date_col] - df['last_date']).dt.days
    df['dias_desde_ultima_licitacion'] = df['dias_desde_ultima_licitacion'].fillna(-1).astype(int)
    df = df.drop(columns=['last_date'])

    # (Lógica de tasa_exito y tasa_sobrecoste ELIMINADA)

    # 3. descuento_medio_hist
    if descuento_col in df.columns:
        df['descuento_medio_hist'] = g[descuento_col].transform(lambda s: s.expanding().mean().shift(1).fillna(-1.0))
    else:
        df['descuento_medio_hist'] = -1.0
        
    # 4. presupuesto_medio_hist (Tamaño)
    if pres_col in df.columns:
        df['presupuesto_medio_hist'] = g[pres_col].transform(lambda s: s.expanding().mean().shift(1).fillna(-1.0))
    else:
        df['presupuesto_medio_hist'] = -1.0

    # 5. pct_hist_{tipo_contrato} (Especialización)
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
                pct_hist = df[f'cum_tc_{tc_clean}'] / n_licit_safe # División segura
                df[f'pct_hist_{tc_clean}'] = np.where(df['n_licitaciones_hist'] > 0, pct_hist, -1.0)
    
    # 6. pct_hist_com_aut_{geografia} (Localización)
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
                pct_hist = df[f'cum_geo_{geo_clean}'] / n_licit_safe # División segura
                df[f'pct_hist_com_aut_{geo_clean}'] = np.where(df['n_licitaciones_hist'] > 0, pct_hist, -1.0)
                
    return df

# -----------------------------------------------------------------
# FUNCIÓN 2: BUILD_HIST_SUMMARY (La tabla de consulta)
# -----------------------------------------------------------------
def build_hist_summary(df_hist, empresa_col, date_col, descuento_col, tipo_col, geo_col, pres_col):
    """
    Construye la tabla resumen (SIN SHIFT) para usar con merge_asof.
    """
    df = df_hist.sort_values([empresa_col, date_col]).copy().reset_index(drop=True)
    g = df.groupby(empresa_col)
    
    df['hist_cum_n'] = g.cumcount() + 1
    
    # Sumas para descuento y PRESUPUESTO
    for c in [descuento_col, pres_col]:
        if c in df.columns:
            df[f'hist_cum_sum_{c}'] = g[c].cumsum()

    # pct_hist_{tipo_contrato}
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
        
    # pct_hist_com_aut_{geografia}
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
                
    keep_cols = [empresa_col, date_col, 'hist_cum_n'] + \
                [c for c in df.columns if c.startswith('hist_cum_sum_') or \
                                          c.startswith('hist_cum_tc_') or \
                                          c.startswith('hist_cum_geo_')]
    
    df_final = df[keep_cols].reset_index(drop=True)
    
    # --- ¡CORRECCIÓN KeyError! ---
    # Renombramos la 'date_col' del historial para que no colisione
    df_final = df_final.rename(columns={date_col: f"{date_col}_hist"})
    # --- FIN CORRECCIÓN ---
                                          
    return df_final

# -----------------------------------------------------------------
# FUNCIÓN 3: APLICAR_HISTORICOS_ASOF (La lógica del merge_asof)
# -----------------------------------------------------------------
def aplicar_historicos_asof(df_para_aplicar, df_resumen, empresa_col, date_col, pres_col, descuento_col):
    """
    Aplica el resumen histórico a un df (val o test) usando merge_asof.
    """
    df_sorted = df_para_aplicar.sort_values([empresa_col, date_col]).reset_index(drop=True)
    # --- ¡CORRECCIÓN KeyError! ---
    # Ordenamos por la columna de fecha renombrada
    hist_summary_sorted = df_resumen.sort_values([empresa_col, f"{date_col}_hist"]).reset_index(drop=True)
    # --- FIN CORRECCIÓN ---

    merged_list = []
    
    all_cum_tc_cols = [c for c in hist_summary_sorted.columns if c.startswith('hist_cum_tc_')]
    all_cum_geo_cols = [c for c in hist_summary_sorted.columns if c.startswith('hist_cum_geo_')]
    
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
            
            merged_list.append(g_res)
            continue
            
        # --- ¡CORRECCIÓN KeyError! ---
        # El merge ahora usa 'date_col' (izquierda) y '..._hist' (derecha)
        merged = pd.merge_asof(g.sort_values(date_col), hist_sub,
                               left_on=date_col, right_on=f"{date_col}_hist", by=empresa_col, 
                               direction='backward', suffixes=('', '_sufijo_basura'))
        # --- FIN CORRECCIÓN ---
        
        merged_res = merged.copy()
        
        merged_res['n_licitaciones_hist'] = merged_res['hist_cum_n'].fillna(0).astype(int)
        
        # Denominador "seguro" para el merge
        n_licit_safe_merge = merged_res['n_licitaciones_hist'].replace(0, 1)
        
        # Esta línea AHORA FUNCIONARÁ
        merged_res['dias_desde_ultima_licitacion'] = (merged_res[date_col] - merged_res[f'{date_col}_hist']).dt.days.fillna(-1).astype(int)
            
        # descuento_medio_hist (con división segura)
        merged_res['descuento_medio_hist'] = np.where(merged_res['n_licitaciones_hist'] > 0,
                                                      merged_res[desc_sum_col].fillna(0) / n_licit_safe_merge, -1.0)
            
        # presupuesto_medio_hist (con división segura)
        merged_res['presupuesto_medio_hist'] = np.where(merged_res['n_licitaciones_hist'] > 0,
                                                        merged_res[pres_sum_col].fillna(0) / n_licit_safe_merge, -1.0)
                
        # pct hist per TIPO (con división segura)
        for col in all_cum_tc_cols:
            tc_clean = col.replace('hist_cum_tc_', '')
            pct_hist = merged_res[col].fillna(0) / n_licit_safe_merge
            merged_res[f'pct_hist_{tc_clean}'] = np.where(merged_res['n_licitaciones_hist'] > 0,
                                                         pct_hist, -1.0)
                                                         
        # pct hist per GEOGRAFÍA (con división segura)
        for col in all_cum_geo_cols:
            geo_clean = col.replace('hist_cum_geo_', '')
            pct_hist = merged_res[col].fillna(0) / n_licit_safe_merge
            merged_res[f'pct_hist_com_aut_{geo_clean}'] = np.where(merged_res['n_licitaciones_hist'] > 0,
                                                                   pct_hist, -1.0)
        
        merged_list.append(merged_res)

    return pd.concat(merged_list, axis=0).reset_index(drop=True)

# -----------------------------------------------------------------
# --- SCRIPT PRINCIPAL DE EJECUCIÓN ---
# -----------------------------------------------------------------

# (quick_sample_test eliminado)

# 1. Crear features para TRAIN (usando .shift(1))
print("Calculando históricos para TRAIN (lógica shift(1))...")
train_df = calc_hist_train(
    train_df_orig, 
    empresa_col=GROUP_COL, 
    date_col=DATE_COL,
    descuento_col=DESCUENTO_COL,
    tipo_col=TIPO_CONTRATO_COL,
    geo_col=GEO_COL,
    pres_col=PRESUPUESTO_COL
)

# 2. Crear el resumen de TRAIN (para consultar desde VAL)
print("Construyendo resumen histórico de TRAIN (para merge_asof)...")
hist_train_summary = build_hist_summary(
    train_df_orig,
    empresa_col=GROUP_COL,
    date_col=DATE_COL,
    descuento_col=DESCUENTO_COL,
    tipo_col=TIPO_CONTRATO_COL,
    geo_col=GEO_COL,
    pres_col=PRESUPUESTO_COL
)

# 3. Aplicar resumen de TRAIN a VAL
print("Aplicando históricos a VAL (lógica merge_asof)...")
val_df = aplicar_historicos_asof(
    val_df_orig,
    hist_train_summary,
    empresa_col=GROUP_COL,
    date_col=DATE_COL,
    pres_col=PRESUPUESTO_COL,
    descuento_col=DESCUENTO_COL
)

# 4. Crear el resumen de TRAIN+VAL (para consultar desde TEST)
print("Construyendo resumen histórico de TRAIN+VAL (para merge_asof)...")
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

# 5. Aplicar resumen de TRAIN+VAL a TEST
print("Aplicando históricos a TEST (lógica merge_asof)...")
test_df = aplicar_historicos_asof(
    test_df_orig,
    hist_train_val_summary,
    empresa_col=GROUP_COL,
    date_col=DATE_COL,
    pres_col=PRESUPUESTO_COL,
    descuento_col=DESCUENTO_COL
)

# -----------------------------------------------------------------
# --- ¡NUEVO! PASO 6: LIMPIEZA FINAL DE COLUMNAS ---
# -----------------------------------------------------------------
print("\nLimpiando columnas de cálculo intermedias...")

# 1. Definir las columnas finales que queremos MANTENER
#    (Todas las originales + las nuevas features de historial)
final_features = [
    'n_licitaciones_hist',
    'dias_desde_ultima_licitacion',
    'descuento_medio_hist',
    'presupuesto_medio_hist',
]
# Añadir todas las 'pct_hist_' (de tipo y geo)
final_features.extend([col for col in train_df.columns if col.startswith('pct_hist_')])

# 2. La lista final es: Columnas Originales + Nuevas Features
final_columns_to_keep = original_columns + final_features

# 3. Filtrar los DataFrames para quedarse SOLO con esas columnas
#    Usamos set() para eliminar duplicados si los hubiera y list() para mantener el orden
keep_cols_train = [col for col in train_df.columns if col in set(final_columns_to_keep)]
keep_cols_val = [col for col in val_df.columns if col in set(final_columns_to_keep)]
keep_cols_test = [col for col in test_df.columns if col in set(final_columns_to_keep)]

train_df = train_df[keep_cols_train]
val_df = val_df[keep_cols_val]
test_df = test_df[keep_cols_test]

print(f"Limpieza completada. Columnas finales en Train: {len(train_df.columns)}")

# 7. Guardar archivos Parquet
print("\nGuardando DataFrames procesados en formato Parquet ...")
train_df.to_parquet(RUTA_GUARDADO + r"\train_procesado_v2_limpio.parquet")
val_df.to_parquet(RUTA_GUARDADO + r"\val_procesado_v2_limpio.parquet")
test_df.to_parquet(RUTA_GUARDADO + r"\test_procesado_v2_limpio.parquet")
print("¡Guardado completado en '..._v2_limpio.parquet'!")

print("Script de creación de históricos completado.")
