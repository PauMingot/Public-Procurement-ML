import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import joblib
from pathlib import Path
import warnings
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------------------
RUTA = Path(r"C:\Users\User\Documents\InferIA")

TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"
TEST_PATH  = RUTA / "test_procesado_v2_limpio.parquet"

# Variables base
IMPORTE_ADJ = 'lote_importe_adjudicacion_sin_impuestos'
PRESUPUESTO = 'lote_presupuesto_base_sin_impuestos'
VALOR_ESTIMADO = 'valor_estimado_imputado'

# NUEVO TARGET
TARGET = 'pct_baja'

OUTPUT_DIR = RUTA / "salida_modelo_empresa_baja"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 2. CARGA DE DATOS
# ------------------------------------------------------------
print(f"⏳ Cargando {TRAIN_PATH.name}...")
train_df = pd.read_parquet(TRAIN_PATH)
print(f"⏳ Cargando {VAL_PATH.name}...")
val_df   = pd.read_parquet(VAL_PATH)
print(f"⏳ Cargando {TEST_PATH.name}...")
test_df  = pd.read_parquet(TEST_PATH)

print(f"Dimensiones iniciales -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ------------------------------------------------------------
# 3. LIMPIEZA, PREPARACIÓN Y CÁLCULO DEL NUEVO TARGET
# ------------------------------------------------------------
print("\n🧹 Iniciando limpieza estricta y cálculo de % de baja...")
for name, df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
    
    start_len = len(df)
    
    # 1. Solo adjudicadas
    if 'es_exito' in df.columns:
        df = df[df['es_exito'] == 1].copy()

    # 2. Eliminar sobrecostes (Adjudicación > Presupuesto)
    if IMPORTE_ADJ in df.columns and PRESUPUESTO in df.columns:
        df = df[df[IMPORTE_ADJ] <= df[PRESUPUESTO]].copy()

    # 3. Eliminar anomalías (Presupuesto > Valor Estimado)
    if PRESUPUESTO in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[PRESUPUESTO] <= df[VALOR_ESTIMADO]].copy()
        
    # 4. Eliminar NaNs y Ceros en el Importe
    if IMPORTE_ADJ in df.columns:
        df.dropna(subset=[IMPORTE_ADJ], inplace=True) 
        df = df[df[IMPORTE_ADJ] > 0].copy() 
    
    # 5. Filtro de Precios Simbólicos (>= 2€)
    if IMPORTE_ADJ in df.columns:
        df = df[df[IMPORTE_ADJ] >= 2.0].copy()

    # 6. Filtro de Coherencia Económica (Precios Unitarios y Acuerdos Marco)
    if PRESUPUESTO in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[VALOR_ESTIMADO] <= (df[PRESUPUESTO] * 10)].copy()

    # 🎯 7. CÁLCULO DEL NUEVO TARGET: Porcentaje de Baja (0.0 a 1.0)
    df[TARGET] = (df[PRESUPUESTO] - df[IMPORTE_ADJ]) / df[PRESUPUESTO]

    # Actualizar referencia del DataFrame
    if name == 'train': train_df = df
    elif name == 'val': val_df = df
    elif name == 'test': test_df = df
    
    print(f"  [{name.upper()}]: {start_len} filas -> {len(df)} tras limpieza.")

# ------------------------------------------------------------
# 4. PREPARACIÓN DE VARIABLES (DROPS Y TIPOS)
# ------------------------------------------------------------
DROP_COLS = [
    '_id', 'id', 'fecha_primera_publicacion',
    'objeto', 'lote_objeto', 'organo_contratacion',
    'lote_adjudicatario', 'lote_resultado',
    # ¡CRUCIAL! Eliminamos las variables de importe para no hacer trampa
    IMPORTE_ADJ, 'lote_importe_adjudicacion_con_impuestos',
    'es_exito', 'es_sobrecoste',
    'cpv_final_imputado', 'cif_normalizado',
    # Features que causan fuga de datos
    'descuento_promedio', 'lote_precio_oferta_mas_alta',
    'lote_precio_oferta_mas_baja', 'lote_numero_ofertas_recibidas',
    'presupuesto_medio_hist'
]

for df in [train_df, val_df, test_df]:
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors='ignore')

# Preparar columnas categóricas
cat_cols = train_df.select_dtypes(include=['object', 'category']).columns.tolist()

for df in [train_df, val_df, test_df]:
    for c in cat_cols:
        df[c] = df[c].astype('category')

# Alinear columnas
print("\nAlineando columnas entre Train, Val, y Test...")
train_cols = train_df.columns.tolist()
val_df   = val_df.reindex(columns=train_cols, fill_value=0)
test_df  = test_df.reindex(columns=train_cols, fill_value=0)

numeric_cols = train_df.select_dtypes(include=np.number).columns.drop(TARGET, errors='ignore').tolist()
print(f"Total features para el modelo: {len(numeric_cols)} numéricas, {len(cat_cols)} categóricas.")

# ------------------------------------------------------------
# 5. CONSTRUIR X/y Y PREPROCESADOR
# ------------------------------------------------------------
X_train = train_df.drop(columns=[TARGET], errors='ignore')
y_train = train_df[TARGET].copy() # Ya no usamos logaritmos, es un %

X_val   = val_df.drop(columns=[TARGET], errors='ignore')
y_val   = val_df[TARGET].copy()

X_test  = test_df.drop(columns=[TARGET], errors='ignore')
y_test  = test_df[TARGET].copy()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

# ------------------------------------------------------------
# 6. MODELO: ÁRBOL DE DECISIÓN (CART)
# ------------------------------------------------------------
print("\n🚀 Entrenando Modelo CART para predecir el % DE BAJA...")

pipeline_cart = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(
        random_state=42,
        max_depth=10,
        min_samples_leaf=50
    ))
])

pipeline_cart.fit(X_train, y_train)
print("✅ Modelo CART entrenado.")

# ------------------------------------------------------------
# 7. EVALUACIÓN Y OVERFITTING
# ------------------------------------------------------------
print("\n--- 📊 RESULTADOS DEL MODELO (TEST) ---")
# La predicción ya sale en formato porcentaje (0.0 a 1.0), no hace falta expm1
y_pred_cart = pipeline_cart.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred_cart))
mae = mean_absolute_error(y_test, y_pred_cart)
r2 = r2_score(y_test, y_pred_cart)

# Multiplicamos por 100 para mostrarlo en formato porcentaje (%) más legible
print(f"RMSE : {rmse * 100:.2f} %")
print(f"MAE  : {mae * 100:.2f} %")
print(f"R²   : {r2:.4f}")

pd.DataFrame([{'model': 'CART_baja_v2', 'rmse_pct': rmse*100, 'mae_pct': mae*100, 'r2': r2}]) \
  .to_csv(OUTPUT_DIR / "metricas_test_CART_baja_v2.csv", index=False)

joblib.dump(pipeline_cart, RUTA / "cart_pred_baja_empresa_v2.pkl")

print("\n--- ⚖️ COMPARATIVA TRAIN vs. TEST (OVERFITTING CHECK) ---")
try:
    y_pred_train_cart = pipeline_cart.predict(X_train)
    
    r2_train_cart = r2_score(y_train, y_pred_train_cart)
    mae_train_cart = mean_absolute_error(y_train, y_pred_train_cart)

    data_cart = {
        'Set': ['Train', 'Test'],
        'R² (Poder Predictivo)': [r2_train_cart, r2],
        'MAE (%)': [mae_train_cart * 100, mae * 100]
    }
    
    print(pd.DataFrame(data_cart).to_markdown(index=False, floatfmt=",.4f"))
    
    if (r2_train_cart - r2) > 0.15:
        print("\n⚠️ Diagnóstico: ¡ALERTA DE SOBREAJUSTE (Overfitting)!")
    else:
        print("\n✅ Diagnóstico: ¡Buen ajuste! (Las métricas son estables)")

except Exception as e:
    print(f"Error al calcular métricas de train: {e}")
    
# ------------------------------------------------------------
# 7. EVALUACIÓN Y OVERFITTING
# ------------------------------------------------------------
print("\n--- 📊 RESULTADOS DEL MODELO (TEST) ---")
y_pred_cart = pipeline_cart.predict(X_test)

rmse = sqrt(mean_squared_error(y_test, y_pred_cart))
mae = mean_absolute_error(y_test, y_pred_cart)
r2 = r2_score(y_test, y_pred_cart)

print(f"RMSE (%): {rmse * 100:.2f} %")
print(f"MAE  (%): {mae * 100:.2f} %")
print(f"R²      : {r2:.4f}")

# =====================================================================
# 💶 NUEVO: CONVERSIÓN A EUROS PARA COMPARATIVA JUSTA (APPLES TO APPLES)
# =====================================================================
print("\n--- 💶 TRADUCCIÓN A EUROS (Para comparar con el Modelo Base) ---")
# 1. Recuperamos los presupuestos del set de Test
presupuesto_test = X_test['lote_presupuesto_base_sin_impuestos'].values

# 2. Reconstruimos los importes en Euros: Adjudicación = Presupuesto * (1 - %baja)
# y_test tiene el % de baja real, y_pred_cart tiene el % de baja predicho
y_test_importe_real = presupuesto_test * (1 - y_test.values)
y_pred_importe_estimado = presupuesto_test * (1 - y_pred_cart)

# 3. Calculamos las métricas en Euros
rmse_convertido = sqrt(mean_squared_error(y_test_importe_real, y_pred_importe_estimado))
mae_convertido = mean_absolute_error(y_test_importe_real, y_pred_importe_estimado)

print(f"RMSE (Traducido a Euros): {rmse_convertido:,.2f} €")
print(f"MAE  (Traducido a Euros): {mae_convertido:,.2f} €")
# =====================================================================

# Guardar métricas... (el resto de tu código sigue igual)

# ------------------------------------------------------------
# 8. INTERPRETACIÓN (IMPORTANCIA DE VARIABLES Y GRÁFICO)
# ------------------------------------------------------------
print("\n--- 🔍 INTERPRETACIÓN DEL MODELO ---")
try:
    feature_names = pipeline_cart.named_steps['preprocessor'].get_feature_names_out()
except Exception:
    feature_names = numeric_cols + list(pipeline_cart.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(cat_cols))

model_cart = pipeline_cart.named_steps['model']
importances = model_cart.feature_importances_

df_importance_raw = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values('importance', ascending=False)
df_importance_raw = df_importance_raw[df_importance_raw['importance'] > 0]

print("\nTOP 10 FEATURES MÁS IMPORTANTES:")
print(df_importance_raw.head(10).to_markdown(index=False, floatfmt=".4f"))
df_importance_raw.to_csv(OUTPUT_DIR / "feature_importance_CART_baja_raw_v2.csv", index=False)

print("\n🎨 Generando gráfico del árbol...")
try:
    plt.figure(figsize=(20, 10))
    plot_tree(
        model_cart,
        feature_names=list(feature_names),
        filled=True, rounded=True, max_depth=3, fontsize=8
    )
    plt.title("Árbol de Decisión (CART) para % Baja - max_depth=3")
    plt.savefig(OUTPUT_DIR / 'cart_baja_plot_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico guardado en {OUTPUT_DIR / 'cart_baja_plot_v2.png'}")
except Exception as e:
    print(f"Error al generar gráfico: {e}")

print("\n🎉 ¡Script de modelado completado con éxito!")