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

IMPORTE_ADJ = 'lote_importe_adjudicacion_sin_impuestos'
PRESUPUESTO = 'lote_presupuesto_base_sin_impuestos' 
TARGET = 'pct_baja'

OUTPUT_DIR = RUTA / "salida_modelo_Regresion_Baja"
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

def limpiar_df_y_crear_baja(df):
    """Aplica la limpieza estricta y calcula el % de baja"""
    start_len = len(df)
    VALOR_ESTIMADO = 'valor_estimado_imputado'
    
    if 'es_exito' in df.columns: df = df[df['es_exito'] == 1].copy()
    if IMPORTE_ADJ in df.columns and PRESUPUESTO in df.columns:
        df = df[df[IMPORTE_ADJ] <= df[PRESUPUESTO]].copy()
    if PRESUPUESTO in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[PRESUPUESTO] <= df[VALOR_ESTIMADO]].copy()
    
    # Filtros de importe
    if IMPORTE_ADJ in df.columns:
        df.dropna(subset=[IMPORTE_ADJ], inplace=True) 
        df = df[df[IMPORTE_ADJ] > 0].copy() 
        df = df[df[IMPORTE_ADJ] >= 2.0].copy()
        
    if PRESUPUESTO in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[VALOR_ESTIMADO] <= (df[PRESUPUESTO] * 10)].copy()
        
    # 🎯 CREAR TARGET DE BAJA (0.0 a 1.0)
    df[TARGET] = (df[PRESUPUESTO] - df[IMPORTE_ADJ]) / df[PRESUPUESTO]
        
    print(f"  Limpieza: {start_len} -> {len(df)} filas.")
    return df

print("\n🧹 Aplicando limpieza estricta y calculando bajas...")
print("Train+Val:")
full_train_df = limpiar_df_y_crear_baja(full_train_df)
print("Test:")
test_df = limpiar_df_y_crear_baja(test_df)

# ------------------------------------------------------------
# 3. PREPARACIÓN DE TARGETS Y HELPER PARA METRICAS
# ------------------------------------------------------------
y_train_baja = full_train_df[TARGET]
y_test_baja = test_df[TARGET]

presupuesto_test = test_df[PRESUPUESTO].values
y_test_euros_real = test_df[IMPORTE_ADJ].values

def evaluar_en_euros(modelo_nombre, y_pred_baja):
    # Clipiamos la baja entre 0% y 99% para traducir a euros
    y_pred_baja_segura = np.clip(y_pred_baja, 0.0, 0.99)
    y_pred_euros = presupuesto_test * (1 - y_pred_baja_segura)
    
    return {
        'Modelo': modelo_nombre,
        'R² (Euros)': r2_score(y_test_euros_real, y_pred_euros),
        'MAE (€)': mean_absolute_error(y_test_euros_real, y_pred_euros),
        'MdAPE (%)': mdape(y_test_euros_real, y_pred_euros) # <-- Añadido MdAPE
    }

# ------------------------------------------------------------
# 4. MODELO 0: BASELINE ABSOLUTO (0% de Baja)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 0: BASELINE ABSOLUTO (0% de Baja) ---")
y_pred_baseline_baja = np.zeros(len(test_df))
metrics_baseline = evaluar_en_euros('Baseline Absoluto (Test)', y_pred_baseline_baja)

# ------------------------------------------------------------
# 5. MODELO 1: REGRESIÓN LINEAL SIMPLE
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 1: Regresión Lineal Simple ---")

X_train_log = np.log1p(full_train_df[PRESUPUESTO]).values.reshape(-1, 1)
X_test_log = np.log1p(test_df[PRESUPUESTO]).values.reshape(-1, 1)

model_simple = LinearRegression()
model_simple.fit(X_train_log, y_train_baja)

y_pred_test_1_baja = model_simple.predict(X_test_log)
metrics_mod1 = evaluar_en_euros('Regresión Simple (Test)', y_pred_test_1_baja)

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

pipeline_ridge_2.fit(X_train_m2, y_train_baja)
y_pred_test_2_baja = pipeline_ridge_2.predict(X_test_m2)
metrics_mod2 = evaluar_en_euros('Ridge Top 3 (Test)', y_pred_test_2_baja)

# ------------------------------------------------------------
# 7. MODELO 3: REGRESIÓN RIDGE "TOP 10 REAL"
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 3: Regresión Ridge (Top 10 Variables REALES) ---")

FEATURES_NUM_10 = [
    'descuento_medio_hist', 'duracion_proceso_dias', 'dias_desde_ultima_licitacion', 
    'anio', 'valor_estimado_imputado', 'lote_presupuesto_base_sin_impuestos', 'pct_hist_obras'
]
FEATURES_CAT_10 = ['tipo_procedimiento', 'sistema_contratacion', 'tipo_contrato']

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

pipeline_ridge_10.fit(X_train_m3, y_train_baja)
y_pred_test_3_baja = pipeline_ridge_10.predict(X_test_m3)
metrics_mod3 = evaluar_en_euros('Ridge Top 10 Real (Test)', y_pred_test_3_baja)

# ------------------------------------------------------------
# 8. MODELO 4: REGRESIÓN LASSO "TOP 10 REAL"
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 4: Regresión Lasso (Top 10 Variables REALES) ---")

pipeline_lasso_10 = Pipeline(steps=[
    ('preprocessor', preprocessor_ridge_10),
    ('model', Lasso(random_state=42, alpha=0.001, max_iter=10000))
])

pipeline_lasso_10.fit(X_train_m3, y_train_baja)
y_pred_test_4_baja = pipeline_lasso_10.predict(X_test_m3)
metrics_mod4 = evaluar_en_euros('Lasso Top 10 Real (Test)', y_pred_test_4_baja)

# ------------------------------------------------------------
# 9. TABLA COMPARATIVA FINAL (CON MdAPE)
# ------------------------------------------------------------
print("\n" + "="*60)
print("🏆 RESUMEN FINAL ACTUALIZADO (Predicción de Bajas Traducidas a EUROS) 🏆")
df_final = pd.DataFrame([metrics_baseline, metrics_mod1, metrics_mod2, metrics_mod3, metrics_mod4])
print(df_final[['Modelo', 'R² (Euros)', 'MAE (€)', 'MdAPE (%)']].to_markdown(index=False, floatfmt=",.4f"))


# ------------------------------------------------------------
# 10. EXPLICABILIDAD (COEFICIENTES)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- INTERPRETACIÓN DE COEFICIENTES (Ridge Top 10 Real) ---")
try:
    feature_names = pipeline_ridge_10.named_steps['preprocessor'].get_feature_names_out()
    coefs = pipeline_ridge_10.named_steps['model'].coef_
    
    df_coef = pd.DataFrame({'Variable': feature_names, 'Impacto_Coeficiente_Baja': coefs})
    df_coef['Impacto_Absoluto'] = df_coef['Impacto_Coeficiente_Baja'].abs()
    df_coef = df_coef.sort_values('Impacto_Absoluto', ascending=False)
    
    df_coef['Impacto_en_Puntos_Porcentuales'] = df_coef['Impacto_Coeficiente_Baja'] * 100
    print("\nTop 15 variables con más impacto en predecir LA BAJA (%):")
    print(df_coef[['Variable', 'Impacto_en_Puntos_Porcentuales']].head(15).to_markdown(index=False, floatfmt=",.4f"))
    
except Exception as e:
    print(f"Error al generar la interpretabilidad: {e}")

joblib.dump(pipeline_ridge_10, OUTPUT_DIR / "regresion_ridge_top10_baja_real.pkl")


# ------------------------------------------------------------
# 11. ANÁLISIS GRÁFICO (ERRORES Y RESIDUOS SEPARADOS CON ZOOM)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- GENERANDO GRÁFICOS DE RENDIMIENTO (Separados y con Zoom) ---")

try:
    y_pred_baseline_euros = presupuesto_test
    y_pred_ridge_euros = presupuesto_test * (1 - np.clip(y_pred_test_3_baja, 0.0, 0.99))

    df_results = pd.DataFrame({
        'Real': y_test_euros_real,
        'Prediccion_Ridge': y_pred_ridge_euros,
        'Prediccion_Baseline': y_pred_baseline_euros
    })

    df_results['APE_Ridge'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_Ridge'])) / df_results['Real']
    df_results['APE_Baseline'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_Baseline'])) / df_results['Real']

    # --- 1. Gráfico Boxplot APE ---
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df_results[['APE_Baseline', 'APE_Ridge']],
        orient='h',
        showfliers=False,
        palette=['#4C72B0', '#DD8452']
    )
    plt.title('Distribución del Error Porcentual Absoluto (APE)\nBaseline vs. Ridge Top 10 Real', fontsize=13, fontweight='bold')
    plt.xlabel('Error Porcentual Absoluto (APE %) - Escala Logarítmica')
    plt.gca().set_xscale('log')
    plt.xlim(0.1, 300)
    plt.axvline(x=df_results['APE_Baseline'].median(), color='blue', linestyle='--', label=f"MdAPE Baseline ({df_results['APE_Baseline'].median():.2f}%)")
    plt.axvline(x=df_results['APE_Ridge'].median(), color='darkred', linestyle='--', label=f"MdAPE Ridge ({df_results['APE_Ridge'].median():.2f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_comparativa_error_boxplot_Ridge.png', dpi=300)
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

    # --- 3. Gráfico Scatter: RIDGE TOP 10 REAL ---
    plt.figure(figsize=(8, 8))
    plt.scatter(df_results['Real'], df_results['Prediccion_Ridge'], alpha=0.15, s=10, color='#DD8452')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([min_val_zoom, max_val_zoom], [min_val_zoom, max_val_zoom], color='red', linestyle='--', linewidth=2, label='Predicción Perfecta (y=x)')
    plt.title("Modelo Ridge (Top 10 Real)\n(Corrige prediciendo el % de baja)", fontsize=14, fontweight='bold')
    plt.xlabel("Valor Real Adjudicado (€) [Escala Log]", fontsize=12)
    plt.ylabel("Valor Predicho (€) [Escala Log]", fontsize=12)
    plt.xlim(min_val_zoom, max_val_zoom)
    plt.ylim(min_val_zoom, max_val_zoom)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_scatter_Ridge_Zoom.png', dpi=300)
    plt.close()

    print("✅ Gráficos separados y con zoom generados correctamente en la carpeta de salida.")

except Exception as e:
    print(f"Error generando gráficos: {e}")

print("\n🎉 ¡Script completado!")