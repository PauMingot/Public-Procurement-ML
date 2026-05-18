import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import joblib
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit

warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------------------------------------------
# 1. CONFIGURACIÓN Y FUNCIÓN MdAPE
# ------------------------------------------------------------
RUTA = Path(r"C:\Users\User\Documents\InferIA") 

TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"
TEST_PATH  = RUTA / "test_procesado_v2_limpio.parquet"

TARGET = 'lote_importe_adjudicacion_sin_impuestos'
PRESUPUESTO = 'lote_presupuesto_base_sin_impuestos' 

OUTPUT_DIR = RUTA / "salida_modelo_Regresion_Importe"
OUTPUT_DIR.mkdir(exist_ok=True)

def mdape(y_true, y_pred):
    """Calcula el Error Porcentual Absoluto Mediano"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if np.sum(mask) == 0: return 0.0
    ape = (np.abs(y_true[mask] - y_pred[mask])) / y_true[mask]
    return np.median(ape) * 100

# ------------------------------------------------------------
# 2. CARGA Y LIMPIEZA (ESTÁNDAR TFG)
# ------------------------------------------------------------
print("⏳ Cargando datos...")
train_df = pd.read_parquet(TRAIN_PATH)
val_df   = pd.read_parquet(VAL_PATH)
test_df  = pd.read_parquet(TEST_PATH)

def limpiar_df_estricto(df):
    """Aplica la limpieza estricta (Estándar TFG)"""
    start_len = len(df)
    VALOR_ESTIMADO = 'valor_estimado_imputado'
    
    if 'es_exito' in df.columns: df = df[df['es_exito'] == 1].copy()
    if TARGET in df.columns and PRESUPUESTO in df.columns:
        df = df[df[TARGET] <= df[PRESUPUESTO]].copy()
    if PRESUPUESTO in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[PRESUPUESTO] <= df[VALOR_ESTIMADO]].copy()
    
    if TARGET in df.columns:
        df.dropna(subset=[TARGET], inplace=True) 
        df = df[df[TARGET] > 0].copy() 
        df = df[df[TARGET] >= 2.0].copy() 
        
    if PRESUPUESTO in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[VALOR_ESTIMADO] <= (df[PRESUPUESTO] * 10)].copy()
        
    print(f"  Limpieza: {start_len} -> {len(df)} filas.")
    return df

print("\n🧹 Aplicando limpieza estricta...")
print("Train:")
train_df = limpiar_df_estricto(train_df)
print("Val:")
val_df = limpiar_df_estricto(val_df)
print("Test:")
test_df = limpiar_df_estricto(test_df)

# Concatenamos para el GridSearch, pero guardamos los índices de quién es quién
full_train_df = pd.concat([train_df, val_df], ignore_index=True)

# ------------------------------------------------------------
# 2.5. CONFIGURACIÓN DEL PREDEFINED SPLIT (EVITAR DATA LEAKAGE)
# ------------------------------------------------------------
# -1 indica que pertenece a TRAIN, 0 indica que pertenece a VALIDATION
test_fold = np.concatenate([
    np.full(len(train_df), -1),
    np.zeros(len(val_df))
])
ps = PredefinedSplit(test_fold)

# ------------------------------------------------------------
# 3. PREPARACIÓN DE TARGETS Y HELPER PARA METRICAS
# ------------------------------------------------------------
y_train_log = np.log1p(full_train_df[TARGET])
y_test_log = np.log1p(test_df[TARGET])

y_test_euros_real = test_df[TARGET].values
presupuesto_test = test_df[PRESUPUESTO].values

def evaluar_en_euros(modelo_nombre, y_pred_log):
    y_pred_log_segura = np.clip(y_pred_log, -np.inf, 30)
    y_pred_euros = np.expm1(y_pred_log_segura)
    y_pred_euros[y_pred_euros < 0] = 0
    
    return {
        'Modelo': modelo_nombre,
        'R²': r2_score(y_test_euros_real, y_pred_euros),
        'MAE (€)': mean_absolute_error(y_test_euros_real, y_pred_euros),
        'MdAPE (%)': mdape(y_test_euros_real, y_pred_euros)
    }

# ------------------------------------------------------------
# 4. MODELO 0: BASELINE ABSOLUTO (Copiar Presupuesto)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 0: BASELINE ABSOLUTO (Copiar Presupuesto) ---")

y_pred_baseline_euros = test_df[PRESUPUESTO].copy()
metrics_baseline = {
    'Modelo': 'Baseline Absoluto (Test)',
    'R²': r2_score(y_test_euros_real, y_pred_baseline_euros),
    'MAE (€)': mean_absolute_error(y_test_euros_real, y_pred_baseline_euros),
    'MdAPE (%)': mdape(y_test_euros_real, y_pred_baseline_euros)
}

# ------------------------------------------------------------
# 5. MODELO 1: REGRESIÓN LINEAL SIMPLE (Log-Log)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 1: Regresión Lineal Simple ---")

X_train_log = np.log1p(full_train_df[PRESUPUESTO]).values.reshape(-1, 1)
X_test_log_1 = np.log1p(test_df[PRESUPUESTO]).values.reshape(-1, 1)

model_simple = LinearRegression()
model_simple.fit(X_train_log, y_train_log)

y_pred_test_1_log = model_simple.predict(X_test_log_1)
metrics_mod1 = evaluar_en_euros('Regresión Simple (Test)', y_pred_test_1_log)

# ------------------------------------------------------------
# 6. MODELO 2: REGRESIÓN RIDGE "TOP 3" (Omitido GridSearch por simplicidad)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 2: Regresión Ridge Top 3 (Baseline Multivariante) ---")

FEATURES_NUM_2 = [PRESUPUESTO, 'descuento_medio_hist']
FEATURES_CAT_2 = ['tipo_procedimiento']

X_train_m2 = full_train_df[FEATURES_NUM_2 + FEATURES_CAT_2].copy()
X_test_m2 = test_df[FEATURES_NUM_2 + FEATURES_CAT_2].copy()

for col in FEATURES_NUM_2:
    X_train_m2.loc[:, f'log_{col}'] = np.log1p(X_train_m2[col].clip(0))
    X_test_m2.loc[:, f'log_{col}'] = np.log1p(X_test_m2[col].clip(0))
    
X_train_m2.drop(columns=FEATURES_NUM_2, inplace=True)
X_test_m2.drop(columns=FEATURES_NUM_2, inplace=True)

numeric_cols_2 = [c for c in X_train_m2.columns if c.startswith('log_')]

preprocessor_ridge_2 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols_2),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), FEATURES_CAT_2)
    ], remainder='passthrough'
)

pipeline_ridge_2 = Pipeline(steps=[
    ('preprocessor', preprocessor_ridge_2),
    ('model', Ridge(random_state=42, alpha=1.0))
])

pipeline_ridge_2.fit(X_train_m2, y_train_log)
y_pred_test_2_log = pipeline_ridge_2.predict(X_test_m2)
metrics_mod2 = evaluar_en_euros('Ridge Top 3 (Test)', y_pred_test_2_log)

# ------------------------------------------------------------
# 7. MODELOS OPTIMIZADOS: RIDGE, LASSO Y ELASTIC NET "TOP 10 REAL"
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- INICIANDO BÚSQUEDA DE HIPERPARÁMETROS (GRID SEARCH TEMPORAL) ---")

FEATURES_NUM_10 = [
    'lote_presupuesto_base_sin_impuestos', 
    'descuento_medio_hist', 
    'duracion_proceso_dias', 
    'dias_desde_ultima_licitacion', 
    'valor_estimado_imputado', 
    'n_licitaciones_hist'
]
FEATURES_CAT_10 = [
    'sistema_contratacion', 
    'tipo_procedimiento', 
    'com_aut_adjudicador'
]

FEATURES_NUM_10 = [c for c in FEATURES_NUM_10 if c in full_train_df.columns]
FEATURES_CAT_10 = [c for c in FEATURES_CAT_10 if c in full_train_df.columns]

X_train_m3 = full_train_df[FEATURES_NUM_10 + FEATURES_CAT_10].copy()
X_test_m3 = test_df[FEATURES_NUM_10 + FEATURES_CAT_10].copy()

for col in FEATURES_NUM_10:
    X_train_m3.loc[:, f'log_{col}'] = np.log1p(X_train_m3[col].clip(0))
    X_test_m3.loc[:, f'log_{col}'] = np.log1p(X_test_m3[col].clip(0))
    
X_train_m3.drop(columns=FEATURES_NUM_10, inplace=True)
X_test_m3.drop(columns=FEATURES_NUM_10, inplace=True)

numeric_cols_10 = [c for c in X_train_m3.columns if c.startswith('log_')]

preprocessor_10 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols_10),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), FEATURES_CAT_10)
    ], remainder='passthrough'
)

# Pipelines base
pipe_ridge = Pipeline(steps=[('preprocessor', preprocessor_10), ('model', Ridge(random_state=42))])
pipe_lasso = Pipeline(steps=[('preprocessor', preprocessor_10), ('model', Lasso(random_state=42, max_iter=10000))])
pipe_elastic = Pipeline(steps=[('preprocessor', preprocessor_10), ('model', ElasticNet(random_state=42, max_iter=10000))])

# Mallas de parámetros
param_grid_ridge = {'model__alpha': [0.1, 1.0, 10.0, 100.0, 500.0]}
param_grid_lasso = {'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}
param_grid_elastic = {
    'model__alpha': [0.0001, 0.001, 0.01, 0.1],
    'model__l1_ratio': [0.2, 0.5, 0.8] 
}

print("⏳ Optimizando Ridge...")
grid_ridge = GridSearchCV(pipe_ridge, param_grid_ridge, cv=ps, scoring='neg_mean_absolute_error', n_jobs=-1, refit=True)
grid_ridge.fit(X_train_m3, y_train_log)

print("⏳ Optimizando Lasso...")
grid_lasso = GridSearchCV(pipe_lasso, param_grid_lasso, cv=ps, scoring='neg_mean_absolute_error', n_jobs=-1, refit=True)
grid_lasso.fit(X_train_m3, y_train_log)

print("⏳ Optimizando Elastic Net...")
grid_elastic = GridSearchCV(pipe_elastic, param_grid_elastic, cv=ps, scoring='neg_mean_absolute_error', n_jobs=-1, refit=True)
grid_elastic.fit(X_train_m3, y_train_log)

best_ridge = grid_ridge.best_estimator_
best_lasso = grid_lasso.best_estimator_
best_elastic = grid_elastic.best_estimator_

print("\n🎯 Mejores hiperparámetros encontrados (Importe):")
print(f"  - Ridge: {grid_ridge.best_params_}")
print(f"  - Lasso: {grid_lasso.best_params_}")
print(f"  - Elastic Net: {grid_elastic.best_params_}")

metrics_ridge = evaluar_en_euros('Ridge Optimizado (Test)', best_ridge.predict(X_test_m3))
metrics_lasso = evaluar_en_euros('Lasso Optimizado (Test)', best_lasso.predict(X_test_m3))
metrics_elastic = evaluar_en_euros('Elastic Net Optimizado (Test)', best_elastic.predict(X_test_m3))

# ------------------------------------------------------------
# 7.5 DIAGNÓSTICO DE OVERFITTING (SÓLO LASSO - ESCALA REAL EUROS)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- DIAGNÓSTICO DE OVERFITTING (LASSO) ---")

# 1. Recuperamos los euros reales de Train
y_train_euros_real = full_train_df[TARGET].values

# 2. Predecimos en Train con Lasso (el modelo escupe logaritmos)
y_pred_train_log = best_lasso.predict(X_train_m3)

# 3. Destransformamos a Euros Reales
y_pred_train_log_segura = np.clip(y_pred_train_log, -np.inf, 30)
y_pred_train_euros = np.expm1(y_pred_train_log_segura)
y_pred_train_euros[y_pred_train_euros < 0] = 0

# 4. Calculamos métricas en Train
r2_train_lasso = r2_score(y_train_euros_real, y_pred_train_euros)
mae_train_lasso = mean_absolute_error(y_train_euros_real, y_pred_train_euros)

# 5. Imprimimos comparativa Train vs Test
print(f"R²  Train: {r2_train_lasso:.4f}  |  R²  Test: {metrics_lasso['R²']:.4f}")
print(f"MAE Train: {mae_train_lasso:,.2f} € |  MAE Test: {metrics_lasso['MAE (€)']:,.2f} €")

if mae_train_lasso > 0:
    diferencia_mae = ((metrics_lasso['MAE (€)'] - mae_train_lasso) / mae_train_lasso) * 100
    print(f"-> Degradación del MAE en Test: {diferencia_mae:+.1f}%")

# ------------------------------------------------------------
# 8. TABLA COMPARATIVA FINAL 
# ------------------------------------------------------------
print("\n" + "="*60)
print("🏆 RESUMEN FINAL ACTUALIZADO (Importe Directo) 🏆")
df_final = pd.DataFrame([metrics_baseline, metrics_mod1, metrics_mod2, metrics_ridge, metrics_lasso, metrics_elastic])
print(df_final[['Modelo', 'R²', 'MAE (€)', 'MdAPE (%)']].to_markdown(index=False, floatfmt=",.4f"))

# ------------------------------------------------------------
# 9. EXPLICABILIDAD (COEFICIENTES DEL MEJOR MODELO)
# (Asumimos Lasso Optimizado como ganador por defecto para el gráfico, puedes cambiar a best_ridge o best_elastic si ganan)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- INTERPRETACIÓN DE COEFICIENTES (Lasso Optimizado) ---")
try:
    feature_names = best_lasso.named_steps['preprocessor'].get_feature_names_out()
    coefs = best_lasso.named_steps['model'].coef_
    
    df_coef = pd.DataFrame({'Variable': feature_names, 'Impacto_Coeficiente_Log': coefs})
    df_coef['Impacto_Absoluto'] = df_coef['Impacto_Coeficiente_Log'].abs()
    
    vars_eliminadas = len(df_coef[df_coef['Impacto_Coeficiente_Log'] == 0])
    print(f"Lasso ha eliminado {vars_eliminadas} variables (asignándoles coeficiente 0.0).")
    
    df_coef = df_coef.sort_values('Impacto_Absoluto', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df_coef.head(15), 
        x='Impacto_Coeficiente_Log', 
        y='Variable', 
        hue='Variable',      
        legend=False,        
        palette="vlag"
    )
    plt.title("Impacto de las Variables en el Importe (Lasso Optimizado)")
    plt.xlabel("Coeficiente Log (+ implica encarecimiento, - implica abaratamiento)")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importancia_coeficientes_lasso_importe_real.png", dpi=300)
    plt.close()
    
except Exception as e:
    print(f"Error al generar la interpretabilidad: {e}")

joblib.dump(best_lasso, OUTPUT_DIR / "regresion_lasso_top10_importe_real_optimizado.pkl")

# ------------------------------------------------------------
# 10. ANÁLISIS GRÁFICO (ERRORES Y RESIDUOS)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- GENERANDO GRÁFICOS DE RENDIMIENTO ---")

try:
    y_pred_log_segura = np.clip(best_lasso.predict(X_test_m3), -np.inf, 30)
    y_pred_lasso_euros = np.expm1(y_pred_log_segura)
    y_pred_lasso_euros[y_pred_lasso_euros < 0] = 0

    df_results = pd.DataFrame({
        'Real': y_test_euros_real,
        'Prediccion_Lasso': y_pred_lasso_euros,
        'Prediccion_Baseline': y_pred_baseline_euros
    })

    df_results['APE_Lasso'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_Lasso'])) / df_results['Real']
    df_results['APE_Baseline'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_Baseline'])) / df_results['Real']

    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df_results[['APE_Baseline', 'APE_Lasso']],
        orient='h',
        showfliers=False,
        palette=['#4C72B0', '#55A868'] 
    )
    plt.title('Distribución del Error Porcentual Absoluto (APE)\nBaseline vs. Lasso Optimizado (Importe)', fontsize=13, fontweight='bold')
    plt.xlabel('Error Porcentual Absoluto (APE %) - Escala Logarítmica')
    plt.gca().set_xscale('log')
    plt.xlim(0.1, 300) 
    plt.axvline(x=df_results['APE_Baseline'].median(), color='blue', linestyle='--', label=f"MdAPE Baseline ({df_results['APE_Baseline'].median():.2f}%)")
    plt.axvline(x=df_results['APE_Lasso'].median(), color='darkgreen', linestyle='--', label=f"MdAPE Lasso ({df_results['APE_Lasso'].median():.2f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_comparativa_error_boxplot_Lasso.png', dpi=300)
    plt.close()

    min_val_zoom = max(10.0, df_results['Real'].quantile(0.005)) 
    max_val_zoom = df_results['Real'].quantile(0.995)

    plt.figure(figsize=(8, 8))
    plt.scatter(df_results['Real'], df_results['Prediccion_Lasso'], alpha=0.15, s=10, color='#55A868')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([min_val_zoom, max_val_zoom], [min_val_zoom, max_val_zoom], color='red', linestyle='--', linewidth=2, label='Predicción Perfecta (y=x)')
    plt.title("Modelo Lasso Optimizado\n(Predicción Directa Log-Log)", fontsize=14, fontweight='bold')
    plt.xlabel("Valor Real Adjudicado (€) [Escala Log]", fontsize=12)
    plt.ylabel("Valor Predicho (€) [Escala Log]", fontsize=12)
    plt.xlim(min_val_zoom, max_val_zoom)
    plt.ylim(min_val_zoom, max_val_zoom)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_scatter_Lasso_Zoom.png', dpi=300)
    plt.close()

except Exception as e:
    print(f"Error generando gráficos: {e}")

print("\n🎉 ¡Script Importe completado!")