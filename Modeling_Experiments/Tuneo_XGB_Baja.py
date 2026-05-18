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

TARGET = 'pct_baja'  # <--- Target: Porcentaje de Baja
PRESUPUESTO_COL = 'lote_presupuesto_base_sin_impuestos'
IMPORTE_COL = 'lote_importe_adjudicacion_sin_impuestos'

OUTPUT_DIR = RUTA / "Tuning_XGB_Baja"
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
    'lote_precio_oferta_mas_alta', 'lote_precio_oferta_mas_baja', 'presupuesto_base_sin_impuestos',
    'num_proceso_dias'
]

train_df.drop(columns=[c for c in DROP_COLS if c in train_df.columns], inplace=True, errors='ignore')
val_df.drop(columns=[c for c in DROP_COLS if c in val_df.columns], inplace=True, errors='ignore')

# Variables (¡SIN LOGARITMOS EN LA Y! Predicimos el porcentaje directamente)
X_train = train_df.drop(columns=[TARGET])
y_train = train_df[TARGET].values

X_val = val_df.drop(columns=[TARGET])
y_val = val_df[TARGET].values 

# Preprocesador
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
# 2. FASE 1: GRID SEARCH CON EARLY STOPPING (EVALUANDO EN EUROS)
# ------------------------------------------------------------
print("\n" + "="*50)
print("--- BUSCANDO HIPERPARÁMETROS (Explorando la frontera superior) ---")

# Cuadrícula desplazada hacia arriba basándonos en los resultados anteriores
param_grid = {
    'learning_rate': [0.05, 0.1],     
    'max_depth': [15, 20],            # <--- Subimos el límite de profundidad  
    'reg_lambda': [10, 20, 40],       # <--- Subimos mucho la regularización L2      
    'gamma': [0, 1],                  
    'subsample': [0.8],               
    'colsample_bytree': [0.8]         
}

best_mae_euros = float('inf')
best_params = {}
best_n_trees = 0

keys, values = zip(*param_grid.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Probando {len(combinations)} combinaciones...")

for idx, params in enumerate(combinations):
    xgb = XGBRegressor(
        n_estimators=1500, 
        random_state=42, 
        n_jobs=-1,
        early_stopping_rounds=50, 
        eval_metric="mae",   # <--- AHORA SE PONE AQUÍ (En la definición del modelo)
        **params
    )
    
    # Entrenamos (XGBoost vigilará el MAE automáticamente)
    xgb.fit(
        X_train_trans, y_train,
        eval_set=[(X_val_trans, y_val)],
        verbose=False
    )
    
    # Predicción directa de la BAJA en %
    pred_val_baja = xgb.predict(X_val_trans)
    
    # DESTRANSFORMACIÓN MATEMÁTICA A EUROS
    pred_val_baja_segura = np.clip(pred_val_baja, 0.0, 100.0)
    pred_val_euros = presupuesto_val - (pred_val_baja_segura / 100.0) * presupuesto_val
    
    # Calculamos el error absoluto comparando euros contra euros
    mae_val_euros = mean_absolute_error(y_val_euros_real, pred_val_euros)
    n_trees = xgb.best_iteration
    
    print(f"[{idx+1}/{len(combinations)}] Árboles: {n_trees} | MAE Val: {mae_val_euros:,.0f} € | Params: {params}")
    
    if mae_val_euros < best_mae_euros:
        best_mae_euros = mae_val_euros
        best_params = params
        best_n_trees = n_trees

print("\n" + "="*50)
print(f"🏆 MEJORES PARÁMETROS: {best_params}")
print(f"🌲 Número óptimo de árboles: {best_n_trees}")
print(f"📉 Mejor MAE en Euros: {best_mae_euros:,.0f} €")
print("💡 COPIA estos datos para tu script XGB_ganador_baja.py")