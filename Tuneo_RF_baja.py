import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. CONFIGURACIÓN Y LIMPIEZA
# ------------------------------------------------------------
RUTA = Path(r"C:\Users\User\Documents\InferIA")
TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"

TARGET = 'pct_baja'  # <--- CAMBIADO AL PORCENTAJE DE BAJA
PRESUPUESTO_COL = 'lote_presupuesto_base_sin_impuestos'
IMPORTE_COL = 'lote_importe_adjudicacion_sin_impuestos'

OUTPUT_DIR = RUTA / "Tuning_RF_Baja"
OUTPUT_DIR.mkdir(exist_ok=True)

def limpiar_df_estricto(df):
    start_len = len(df)
    VALOR_ESTIMADO = 'valor_estimado_imputado'
    
    # 🚨 1. FÓRMULA EXPLÍCITA: Calculamos la baja (0 a 100)
    if IMPORTE_COL in df.columns and PRESUPUESTO_COL in df.columns:
        df[TARGET] = ((df[PRESUPUESTO_COL] - df[IMPORTE_COL]) / df[PRESUPUESTO_COL]) * 100
        
    if 'es_exito' in df.columns: df = df[df['es_exito'] == 1].copy()
    
    # Filtros de coherencia del presupuesto
    if PRESUPUESTO_COL in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[PRESUPUESTO_COL] <= df[VALOR_ESTIMADO]].copy()
        df = df[df[VALOR_ESTIMADO] <= (df[PRESUPUESTO_COL] * 10)].copy()
        
    # Filtros específicos para el porcentaje de baja (entre 0% y 100%)
    if TARGET in df.columns:
        df.dropna(subset=[TARGET], inplace=True) 
        df = df[(df[TARGET] >= 0.0) & (df[TARGET] <= 100.0)].copy() 
        
    return df

print("⏳ Cargando y limpiando datos (Solo Train y Val para Tuning)...")
train_df = limpiar_df_estricto(pd.read_parquet(TRAIN_PATH))
val_df   = limpiar_df_estricto(pd.read_parquet(VAL_PATH))

# 🚨 ANTES DE BORRAR COLUMNAS: Guardamos los euros de Validación para calcular el MAE luego
y_val_euros_real = val_df[IMPORTE_COL].values
presupuesto_val  = val_df[PRESUPUESTO_COL].values

# Columnas a borrar (AÑADIMOS EL IMPORTE PARA EVITAR DATA LEAKAGE)
DROP_COLS = [
    '_id', 'id', 'fecha_primera_publicacion', 'objeto', 'lote_objeto', 
    'organo_contratacion', 'lote_adjudicatario', 'lote_resultado', 
    'cpv_final_imputado', 'cif_normalizado', 'lote_importe_adjudicacion_con_impuestos',
    IMPORTE_COL, # <--- VITAL BORRARLO AQUÍ (El modelo no lo verá)
    'es_exito', 'es_sobrecoste', 'lote_numero_ofertas_recibidas', 
    'presupuesto_medio_hist', 'descuento_promedio', 
    'lote_precio_oferta_mas_alta', 'lote_precio_oferta_mas_baja', 'presupuesto_base_sin_impuestos'
]
train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns], inplace=True, errors='ignore')
val_df.drop(columns=[c for c in DROP_COLS if c in val_df.columns], inplace=True, errors='ignore')

# Variables (¡SIN LOGARITMOS EN LA Y!)
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].values

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET].values 

# Preprocesador (CART usa X originales, no logaritmos)
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='passthrough'
)

# Aplicamos preprocesador una sola vez
print("Transformando variables categóricas...")
X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_val)


# ------------------------------------------------------------
# 2. FASE 1: GRID SEARCH (Arquitectura del Árbol)
# ------------------------------------------------------------
print("\n" + "="*50)
print("--- FASE 1: BUSCANDO LA ESTRUCTURA ÓPTIMA (EVALUANDO EN EUROS) ---")

param_grid = {
    'max_depth': [10, 15, 20],       
    'min_samples_leaf': [2, 4, 8],    
    'max_features': [0.5, 0.8, 'sqrt'] 
}

best_mae_euros = float('inf')
best_params = {}

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Probando {len(combinations)} combinaciones (con n_estimators=100 fijo)...")

for idx, params in enumerate(combinations):
    rf = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1, 
        **params
    )
    # Entrenamiento directo (aprendiendo porcentajes)
    rf.fit(X_train_trans, y_train)
    
    # Predicción directa de la BAJA en %
    pred_val_baja = rf.predict(X_val_trans)
    
    # 🚨 DESTRANSFORMACIÓN MATEMÁTICA A EUROS
    # Importe_Predicho = Presupuesto - (Baja% / 100) * Presupuesto
    pred_val_euros = presupuesto_val - (pred_val_baja / 100.0) * presupuesto_val
    
    # Calculamos el error absoluto comparando euros contra euros
    mae_val_euros = mean_absolute_error(y_val_euros_real, pred_val_euros)
    
    print(f"[{idx+1}/{len(combinations)}] {params} --> MAE Val: {mae_val_euros:,.0f} €")
    
    if mae_val_euros < best_mae_euros:
        best_mae_euros = mae_val_euros
        best_params = params

print(f"\n🏆 MEJORES PARÁMETROS ENCONTRADOS: {best_params} (MAE: {best_mae_euros:,.0f} €)")


# ------------------------------------------------------------
# 3. FASE 2: CURVA DE CONVERGENCIA (n_estimators)
# ------------------------------------------------------------
print("\n" + "="*50)
print("--- FASE 2: BUSCANDO EL NÚMERO ÓPTIMO DE ÁRBOLES ---")

n_trees_list = list(range(25, 400, 25)) 
mae_scores_euros = []

for n in n_trees_list:
    rf_final = RandomForestRegressor(
        n_estimators=n,
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    rf_final.fit(X_train_trans, y_train)
    
    pred_val_baja = rf_final.predict(X_val_trans)
    pred_val_euros = presupuesto_val - (pred_val_baja / 100.0) * presupuesto_val
    
    mae_val_euros = mean_absolute_error(y_val_euros_real, pred_val_euros)
    mae_scores_euros.append(mae_val_euros)
    print(f"Árboles: {n} -> MAE Val: {mae_val_euros:,.0f} €")


# ------------------------------------------------------------
# 4. GRÁFICO DE CONVERGENCIA PARA LA MEMORIA
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(n_trees_list, mae_scores_euros, marker='o', linestyle='-', color='#E1812C', linewidth=2)
plt.title("Curva de Convergencia del Error (RF - Vía Porcentaje de Baja)\nDemostración Empírica del Límite de Breiman en Euros", fontsize=14, fontweight='bold')
plt.xlabel("Número de Árboles en el Ensamble (K)", fontsize=12)
plt.ylabel("Error Absoluto Medio (MAE) en Euros - Conjunto de Validación", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

best_idx = np.argmin(mae_scores_euros)
best_n = n_trees_list[best_idx]
best_score = mae_scores_euros[best_idx]
plt.axvline(x=best_n, color='red', linestyle='--', label=f'Óptimo / Estabilización (K={best_n})')

plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Curva_Convergencia_Breiman_Baja_Euros.png", dpi=300)
plt.close()

print(f"\n✅ Gráfico de convergencia guardado en {OUTPUT_DIR}")
print(f"\n💡 CONCLUSIÓN PARA TU TFG: Configura tu script RF_Baja.py con n_estimators={best_n} y {best_params}")