import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
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

OUTPUT_DIR = RUTA / "Tuning_XGB_Importe"
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

# Columnas a borrar por Data Leakage
DROP_COLS = [
    '_id', 'id', 'fecha_primera_publicacion', 'objeto', 'lote_objeto', 
    'organo_contratacion', 'lote_adjudicatario', 'lote_resultado', 
    'cpv_final_imputado', 'cif_normalizado', 'lote_importe_adjudicacion_con_impuestos',
    'es_exito', 'es_sobrecoste', 'lote_numero_ofertas_recibidas', 
    'presupuesto_medio_hist', 'descuento_promedio', 
    'lote_precio_oferta_mas_alta', 'lote_precio_oferta_mas_baja', 'presupuesto_base_sin_impuestos',
    'num_proceso_dias'
]

train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns], inplace=True, errors='ignore')
val_df.drop(columns=[c for c in DROP_COLS if c in val_df.columns], inplace=True, errors='ignore')

# Separación de variables
X_train = train_df.drop(columns=[TARGET])
y_train_log = np.log1p(train_df[TARGET])

X_val = val_df.drop(columns=[TARGET])
y_val_log = np.log1p(val_df[TARGET])
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

print("Transformando variables categóricas...")
X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_val)

# ------------------------------------------------------------
# 2. FASE 1: GRID SEARCH CON EARLY STOPPING
# ------------------------------------------------------------
print("\n" + "="*50)
print("--- BUSCANDO HIPERPARÁMETROS (Con Parada Temprana) ---")

# Cuadrícula teórica de XGBoost (Modificada para evitar sobreajuste extremo)
param_grid = {
    'learning_rate': [0.1, 0.15, 0.2],     
    'max_depth': [2,4,6],             
    'reg_lambda': [1],            
    'gamma': [0],                  
    'subsample': [0.8],               
    'colsample_bytree': [0.8]         
}

best_mae = float('inf')
best_params = {}
best_n_trees = 0

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Probando {len(combinations)} combinaciones...")

# Calculamos el máximo logaritmo del train una sola vez para usarlo de tope (PUNTO 3)
max_log_train = y_train_log.max()

for idx, params in enumerate(combinations):
    xgb = XGBRegressor(
        n_estimators=1500, 
        random_state=42, 
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="mae", 
        **params
    )
    
    # Entrenamos fijando MAE como métrica para decidir cuándo parar (PUNTO 2)
    xgb.fit(
        X_train_trans, y_train_log,
        eval_set=[(X_val_trans, y_val_log)],
        verbose=False
    )
    
    # PUNTO 3: 🚨 BLINDAJE MATEMÁTICO: No predecir por encima del máximo histórico
    pred_val_log = np.clip(xgb.predict(X_val_trans), 0, max_log_train)
    pred_val_euros = np.expm1(pred_val_log)
    
    mae_val = mean_absolute_error(y_val_euros_real, pred_val_euros)
    n_trees = xgb.best_iteration
    
    print(f"[{idx+1}/{len(combinations)}] Árboles: {n_trees} | MAE Val: {mae_val:,.0f} € | Params: {params}")
    
    if mae_val < best_mae:
        best_mae = mae_val
        best_params = params
        best_n_trees = n_trees

print("\n" + "="*50)
print(f"🏆 MEJORES PARÁMETROS: {best_params}")
print(f"🌲 Número óptimo de árboles: {best_n_trees}")
print(f"📉 Mejor MAE: {best_mae:,.0f} €")
print("💡 COPIA estos datos para tu script XGB_ganador_importe.py")