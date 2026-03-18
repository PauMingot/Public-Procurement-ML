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
TARGET = 'lote_importe_adjudicacion_sin_impuestos'
PRESUPUESTO_COL = 'lote_presupuesto_base_sin_impuestos'

OUTPUT_DIR = RUTA / "Tuning_RF_Importe"
OUTPUT_DIR.mkdir(exist_ok=True)

def limpiar_df_estricto(df):
    start_len = len(df)
    VALOR_ESTIMADO = 'valor_estimado_imputado'
    if 'es_exito' in df.columns: df = df[df['es_exito'] == 1].copy()
    if TARGET in df.columns and PRESUPUESTO_COL in df.columns:
        df = df[df[TARGET] <= df[PRESUPUESTO_COL]].copy()
    if PRESUPUESTO_COL in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[PRESUPUESTO_COL] <= df[VALOR_ESTIMADO]].copy()
    if TARGET in df.columns:
        df.dropna(subset=[TARGET], inplace=True) 
        df = df[df[TARGET] > 0].copy() 
        df = df[df[TARGET] >= 2.0].copy() 
    if PRESUPUESTO_COL in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[VALOR_ESTIMADO] <= (df[PRESUPUESTO_COL] * 10)].copy()
    return df

print("⏳ Cargando y limpiando datos (Solo Train y Val para Tuning)...")
train_df = limpiar_df_estricto(pd.read_parquet(TRAIN_PATH))
val_df   = limpiar_df_estricto(pd.read_parquet(VAL_PATH))

# Columnas a borrar
DROP_COLS = [
    '_id', 'id', 'fecha_primera_publicacion', 'objeto', 'lote_objeto', 
    'organo_contratacion', 'lote_adjudicatario', 'lote_resultado', 
    'cpv_final_imputado', 'cif_normalizado', 'lote_importe_adjudicacion_con_impuestos',
    'es_exito', 'es_sobrecoste', 'lote_numero_ofertas_recibidas', 
    'presupuesto_medio_hist', 'descuento_promedio', 
    'lote_precio_oferta_mas_alta', 'lote_precio_oferta_mas_baja', 'presupuesto_base_sin_impuestos'
]
train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns], inplace=True, errors='ignore')
val_df.drop(columns=[c for c in DROP_COLS if c in val_df.columns], inplace=True, errors='ignore')

# Variables
X_train = train_df.drop(columns=[TARGET])
y_train_log = np.log1p(train_df[TARGET])

X_val = val_df.drop(columns=[TARGET])
y_val_euros_real = val_df[TARGET].values # Para evaluar MAE real

# Preprocesador
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ], remainder='passthrough'
)

# Aplicamos preprocesador una sola vez para que el GridSearch sea MUCHO más rápido
print("Transformando variables categóricas...")
X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_val)

# ------------------------------------------------------------
# 2. FASE 1: GRID SEARCH (Arquitectura del Árbol)
# ------------------------------------------------------------
print("\n" + "="*50)
print("--- FASE 1: BUSCANDO LA ESTRUCTURA ÓPTIMA ---")

param_grid = {
    'max_depth': [10, 15, 20],       # Profundidad máxima
    'min_samples_leaf': [2, 4, 8],    # Mínimo de muestras en hojas (controla varianza)
    'max_features': [0.5, 0.8, 'sqrt'] # Selección aleatoria (mtry)
}

best_mae = float('inf')
best_params = {}

# Generar todas las combinaciones
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
    rf.fit(X_train_trans, y_train_log)
    
    # Predecir en Validación y destransformar
    pred_val_log = np.clip(rf.predict(X_val_trans), -np.inf, 30)
    pred_val_euros = np.expm1(pred_val_log)
    
    mae_val = mean_absolute_error(y_val_euros_real, pred_val_euros)
    
    print(f"[{idx+1}/{len(combinations)}] {params} --> MAE Val: {mae_val:,.0f} €")
    
    if mae_val < best_mae:
        best_mae = mae_val
        best_params = params

print(f"\n🏆 MEJORES PARÁMETROS ENCONTRADOS: {best_params} (MAE: {best_mae:,.0f} €)")

# ------------------------------------------------------------
# 3. FASE 2: CURVA DE CONVERGENCIA (n_estimators)
# ------------------------------------------------------------
print("\n" + "="*50)
print("--- FASE 2: BUSCANDO EL NÚMERO ÓPTIMO DE ÁRBOLES ---")

n_trees_list = list(range(25, 400, 25)) # De 10 a 250 árboles, saltando de 20 en 20
mae_scores = []

for n in n_trees_list:
    rf_final = RandomForestRegressor(
        n_estimators=n,
        random_state=42,
        n_jobs=-1,
        **best_params
    )
    rf_final.fit(X_train_trans, y_train_log)
    
    pred_val_log = np.clip(rf_final.predict(X_val_trans), -np.inf, 30)
    pred_val_euros = np.expm1(pred_val_log)
    
    mae_val = mean_absolute_error(y_val_euros_real, pred_val_euros)
    mae_scores.append(mae_val)
    print(f"Árboles: {n} -> MAE Val: {mae_val:,.0f} €")

# ------------------------------------------------------------
# 4. GRÁFICO DE CONVERGENCIA PARA LA MEMORIA
# ------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(n_trees_list, mae_scores, marker='o', linestyle='-', color='#2CA02C', linewidth=2)
plt.title("Curva de Convergencia del Error (Random Forest)\nDemostración Empírica del Límite de Breiman", fontsize=14, fontweight='bold')
plt.xlabel("Número de Árboles en el Ensamble (K)", fontsize=12)
plt.ylabel("Error Absoluto Medio (MAE) en Euros - Conjunto de Validación", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Anotar el mejor punto
best_idx = np.argmin(mae_scores)
best_n = n_trees_list[best_idx]
best_score = mae_scores[best_idx]
plt.axvline(x=best_n, color='red', linestyle='--', label=f'Óptimo / Estabilización (K={best_n})')

plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Curva_Convergencia_Breiman.png", dpi=300)
plt.close()

print(f"\n✅ Gráfico de convergencia guardado en {OUTPUT_DIR}")
print(f"\n💡 CONCLUSIÓN PARA TU TFG: Configura tu RF_ganador.py con n_estimators={best_n} y {best_params}")
