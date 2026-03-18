import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import joblib
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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

full_train_df = pd.concat([train_df, val_df], ignore_index=True)

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
print("Train+Val:")
full_train_df = limpiar_df_estricto(full_train_df)
print("Test:")
test_df = limpiar_df_estricto(test_df)

# ------------------------------------------------------------
# 3. PREPARACIÓN DE TARGETS Y HELPER PARA METRICAS
# ------------------------------------------------------------
# En Importe, predecimos el Logaritmo para estabilizar
y_train_log = np.log1p(full_train_df[TARGET])
y_test_log = np.log1p(test_df[TARGET])

y_test_euros_real = test_df[TARGET].values
presupuesto_test = test_df[PRESUPUESTO].values

def evaluar_en_euros(modelo_nombre, y_pred_log):
    # Destransformamos (expm1) para evaluar en euros reales
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
# 6. MODELO 2: REGRESIÓN RIDGE "TOP 3"
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 2: Regresión Ridge (Presupuesto + Historial + Proc.) ---")

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
# 7. MODELO 3: REGRESIÓN RIDGE "TOP 10 REAL (CART IMPORTE)"
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 3: Regresión Ridge (Top 10 Variables REALES del Importe) ---")

# 🚨 ACTUALIZACIÓN: TOP 10 extraído exclusivamente del Árbol de Importe
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

preprocessor_ridge_10 = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols_10),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), FEATURES_CAT_10)
    ], remainder='passthrough'
)

pipeline_ridge_10 = Pipeline(steps=[
    ('preprocessor', preprocessor_ridge_10),
    ('model', Ridge(random_state=42, alpha=1.0))
])

pipeline_ridge_10.fit(X_train_m3, y_train_log)
y_pred_test_3_log = pipeline_ridge_10.predict(X_test_m3)
metrics_mod3 = evaluar_en_euros('Ridge Top 10 Real (Test)', y_pred_test_3_log)

# ------------------------------------------------------------
# 8. MODELO 4: REGRESIÓN LASSO "TOP 10 REAL"
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 4: Regresión Lasso (Top 10 Variables REALES) ---")

pipeline_lasso_10 = Pipeline(steps=[
    ('preprocessor', preprocessor_ridge_10),
    ('model', Lasso(random_state=42, alpha=0.001, max_iter=10000))
])

pipeline_lasso_10.fit(X_train_m3, y_train_log)
y_pred_test_4_log = pipeline_lasso_10.predict(X_test_m3)
metrics_mod4 = evaluar_en_euros('Lasso Top 10 Real (Test)', y_pred_test_4_log)

# ------------------------------------------------------------
# 9. TABLA COMPARATIVA FINAL (CON MdAPE)
# ------------------------------------------------------------
print("\n" + "="*60)
print("🏆 RESUMEN FINAL ACTUALIZADO (Importe Directo) 🏆")
df_final = pd.DataFrame([metrics_baseline, metrics_mod1, metrics_mod2, metrics_mod3, metrics_mod4])
print(df_final[['Modelo', 'R²', 'MAE (€)', 'MdAPE (%)']].to_markdown(index=False, floatfmt=",.4f"))

# ------------------------------------------------------------
# 10. EXPLICABILIDAD (COEFICIENTES LASSO)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- INTERPRETACIÓN DE COEFICIENTES (Lasso Top 10 Real) ---")
try:
    feature_names = pipeline_lasso_10.named_steps['preprocessor'].get_feature_names_out()
    coefs = pipeline_lasso_10.named_steps['model'].coef_
    
    df_coef = pd.DataFrame({'Variable': feature_names, 'Impacto_Coeficiente_Log': coefs})
    df_coef['Impacto_Absoluto'] = df_coef['Impacto_Coeficiente_Log'].abs()
    
    # ¡NUEVO! Comprobamos cuántas variables ha eliminado Lasso
    vars_eliminadas = len(df_coef[df_coef['Impacto_Coeficiente_Log'] == 0])
    print(f"Lasso ha eliminado {vars_eliminadas} variables (asignándoles coeficiente 0.0).")
    
    df_coef = df_coef.sort_values('Impacto_Absoluto', ascending=False)
    
    print("\nTop 15 variables con más impacto en predecir EL IMPORTE (Escala Log):")
    print(df_coef[['Variable', 'Impacto_Coeficiente_Log']].head(15).to_markdown(index=False, floatfmt=",.4f"))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df_coef.head(15), 
        x='Impacto_Coeficiente_Log', 
        y='Variable', 
        hue='Variable',      
        legend=False,        
        palette="vlag"
    )
    plt.title("Impacto de las Variables en el Importe (Lasso Top 10 Real)")
    plt.xlabel("Coeficiente Log (+ implica encarecimiento, - implica abaratamiento)")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importancia_coeficientes_lasso_importe_real.png", dpi=300)
    plt.close()
    
    print(f"\n✅ Gráfico de coeficientes guardado en {OUTPUT_DIR / 'importancia_coeficientes_lasso_importe_real.png'}")
    
except Exception as e:
    print(f"Error al generar la interpretabilidad: {e}")

joblib.dump(pipeline_lasso_10, OUTPUT_DIR / "regresion_lasso_top10_importe_real.pkl")


# ------------------------------------------------------------
# 11. ANÁLISIS GRÁFICO (ERRORES Y RESIDUOS SEPARADOS CON ZOOM)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- GENERANDO GRÁFICOS DE RENDIMIENTO (Separados y con Zoom) ---")

try:
    # Reconstruimos la predicción del modelo ganador (Lasso Top 10) en euros
    # Usamos y_pred_test_4_log porque es la variable donde se guardó la predicción de Lasso
    y_pred_log_segura = np.clip(y_pred_test_4_log, -np.inf, 30)
    y_pred_lasso_euros = np.expm1(y_pred_log_segura)
    y_pred_lasso_euros[y_pred_lasso_euros < 0] = 0

    df_results = pd.DataFrame({
        'Real': y_test_euros_real,
        'Prediccion_Lasso': y_pred_lasso_euros,
        'Prediccion_Baseline': y_pred_baseline_euros
    })

    df_results['APE_Lasso'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_Lasso'])) / df_results['Real']
    df_results['APE_Baseline'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_Baseline'])) / df_results['Real']

    # --- 1. Gráfico Boxplot APE ---
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df_results[['APE_Baseline', 'APE_Lasso']],
        orient='h',
        showfliers=False,
        palette=['#4C72B0', '#55A868'] # Verde para diferenciar Lasso del naranja de Ridge
    )
    plt.title('Distribución del Error Porcentual Absoluto (APE)\nBaseline vs. Lasso Top 10 Real (Importe)', fontsize=13, fontweight='bold')
    plt.xlabel('Error Porcentual Absoluto (APE %) - Escala Logarítmica')
    
    plt.gca().set_xscale('log')
    plt.xlim(0.1, 300) # Fijamos escala para comparar limpiamente
    
    plt.axvline(x=df_results['APE_Baseline'].median(), color='blue', linestyle='--', label=f"MdAPE Baseline ({df_results['APE_Baseline'].median():.2f}%)")
    plt.axvline(x=df_results['APE_Lasso'].median(), color='darkgreen', linestyle='--', label=f"MdAPE Lasso ({df_results['APE_Lasso'].median():.2f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_comparativa_error_boxplot_Lasso.png', dpi=300)
    plt.close()

    # --- CÁLCULO DE LÍMITES PARA EL ZOOM (Recorte del 0.5% extremo) ---
    min_val_zoom = max(10.0, df_results['Real'].quantile(0.005)) 
    max_val_zoom = df_results['Real'].quantile(0.995)

    # --- 2. Gráfico Scatter: BASELINE ---
    plt.figure(figsize=(8, 8))
    plt.scatter(df_results['Real'], df_results['Prediccion_Baseline'], alpha=0.15, s=10, color='#4C72B0')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([min_val_zoom, max_val_zoom], [min_val_zoom, max_val_zoom], color='red', linestyle='--', linewidth=2, label='Predicción Perfecta (y=x)')
    plt.title("Baseline Absoluto\n(Asume Adjudicación = Presupuesto)", fontsize=14, fontweight='bold')
    plt.xlabel("Valor Real Adjudicado (€) [Escala Log]", fontsize=12)
    plt.ylabel("Valor Predicho (€) [Escala Log]", fontsize=12)
    plt.xlim(min_val_zoom, max_val_zoom)
    plt.ylim(min_val_zoom, max_val_zoom)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_scatter_Baseline_Zoom.png', dpi=300)
    plt.close()

    # --- 3. Gráfico Scatter: LASSO TOP 10 REAL ---
    plt.figure(figsize=(8, 8))
    plt.scatter(df_results['Real'], df_results['Prediccion_Lasso'], alpha=0.15, s=10, color='#55A868')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([min_val_zoom, max_val_zoom], [min_val_zoom, max_val_zoom], color='red', linestyle='--', linewidth=2, label='Predicción Perfecta (y=x)')
    plt.title("Modelo Lasso (Top 10 Real Importe)\n(Predicción Directa Log-Log)", fontsize=14, fontweight='bold')
    plt.xlabel("Valor Real Adjudicado (€) [Escala Log]", fontsize=12)
    plt.ylabel("Valor Predicho (€) [Escala Log]", fontsize=12)
    plt.xlim(min_val_zoom, max_val_zoom)
    plt.ylim(min_val_zoom, max_val_zoom)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_scatter_Lasso_Zoom.png', dpi=300)
    plt.close()

    print("✅ Gráficos separados y con zoom generados correctamente en la carpeta de salida.")

except Exception as e:
    print(f"Error generando gráficos: {e}")

print("\n🎉 ¡Script completado!")