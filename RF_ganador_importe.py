import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from pathlib import Path
import warnings
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. CONFIGURACIÓN Y FUNCIÓN MdAPE
# ------------------------------------------------------------
RUTA = Path(r"C:\Users\User\Documents\InferIA") 

# Usamos los datos limpios para evaluar exactamente igual que Regresión Lineal
TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"
TEST_PATH  = RUTA / "test_procesado_v2_limpio.parquet"

TARGET = 'lote_importe_adjudicacion_sin_impuestos'
PRESUPUESTO = 'lote_presupuesto_base_sin_impuestos' 

OUTPUT_DIR = RUTA / "salida_modelo_RF_Importe"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- CONFIGURACIÓN DEL GANADOR (M1) ---
config_m1 = {
    "n_estimators": 300,      
    "max_depth": 15,          
    "max_features": 0.8,      
    "min_samples_leaf": 2     
}

def mdape(y_true, y_pred):
    """Calcula el Error Porcentual Absoluto Mediano"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 0
    if np.sum(mask) == 0: return 0.0
    ape = (np.abs(y_true[mask] - y_pred[mask])) / y_true[mask]
    return np.median(ape) * 100

# ------------------------------------------------------------
# 2. CARGA Y LIMPIEZA ESTRICTA
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

# Eliminar variables que causan data leakage
DROP_COLS = [
    '_id', 'id', 'fecha_primera_publicacion', 'objeto', 'lote_objeto', 
    'organo_contratacion', 'lote_adjudicatario', 'lote_resultado', 
    'cpv_final_imputado', 'cif_normalizado', 'lote_importe_adjudicacion_con_impuestos',
    'es_exito', 'es_sobrecoste', 'lote_numero_ofertas_recibidas', 
    'presupuesto_medio_hist', 'descuento_promedio', 
    'lote_precio_oferta_mas_alta', 'lote_precio_oferta_mas_baja', 'presupuesto_base_sin_impuestos'
]
full_train_df.drop(columns=[c for c in DROP_COLS if c in full_train_df.columns], inplace=True, errors='ignore')
test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns], inplace=True, errors='ignore')

# ------------------------------------------------------------
# 3. PREPARACIÓN DE TARGETS Y HELPER PARA METRICAS
# ------------------------------------------------------------
X_train = full_train_df.drop(columns=[TARGET], errors='ignore')
y_train = full_train_df[TARGET].copy()
y_train_log = np.log1p(y_train)

X_test = test_df.drop(columns=[TARGET], errors='ignore')
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

# Preprocesador (CART usa las numéricas originales, no logaritmo)
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    remainder='passthrough'
)

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
# 5. MODELO 1: RANDOM FOREST GANADOR
# ------------------------------------------------------------
print("\n" + "="*60)
print(f"--- MODELO 1: Random Forest Ganador ---")

pipeline_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(
        n_estimators=config_m1['n_estimators'],
        max_depth=config_m1['max_depth'],
        max_features=config_m1['max_features'],
        min_samples_leaf=config_m1['min_samples_leaf'],
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_rf.fit(X_train, y_train_log)
y_pred_test_rf_log = pipeline_rf.predict(X_test)
metrics_rf = evaluar_en_euros('Random Forest (Test)', y_pred_test_rf_log)

# Guardar el modelo
joblib.dump(pipeline_rf, OUTPUT_DIR / "rf_ganador_importe.pkl")

# ------------------------------------------------------------
# 6. TABLA COMPARATIVA FINAL (CON MdAPE)
# ------------------------------------------------------------
print("\n" + "="*60)
print("🏆 RESUMEN FINAL (Importe Directo) 🏆")
df_final = pd.DataFrame([metrics_baseline, metrics_rf])
print(df_final[['Modelo', 'R²', 'MAE (€)', 'MdAPE (%)']].to_markdown(index=False, floatfmt=",.4f"))
df_final.to_csv(OUTPUT_DIR / "tabla_resultados_final_RF_importe.csv", index=False)

# ------------------------------------------------------------
# 7. EXPLICABILIDAD (IMPORTANCIA DE VARIABLES)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- INTERPRETACIÓN DE VARIABLES (Random Forest) ---")
try:
    feature_names_rf = pipeline_rf.named_steps['preprocessor'].get_feature_names_out()
    model_rf = pipeline_rf.named_steps['model']
    importances_rf = model_rf.feature_importances_

    df_importance_rf_raw = pd.DataFrame({
        'feature': feature_names_rf,
        'importance': importances_rf
    }).sort_values('importance', ascending=False)
    
    def get_original_feature(col_name, cat_cols_list):
        if col_name.startswith('num__'): return col_name.replace('num__', '')
        if col_name.startswith('cat__'):
            for cat_col in cat_cols_list:
                if col_name.startswith(f'cat__{cat_col}_'): return cat_col
        return col_name

    df_importance_rf_raw['original_feature'] = df_importance_rf_raw['feature'].apply(
        lambda x: get_original_feature(x, cat_cols)
    )
    df_importance_rf_agg = df_importance_rf_raw.groupby('original_feature')['importance'].sum()
    df_importance_rf_agg = df_importance_rf_agg.sort_values(ascending=False).reset_index()

    print("\nTop 15 variables con más impacto en predecir EL IMPORTE:")
    print(df_importance_rf_agg.head(15).to_markdown(index=False, floatfmt=",.4f"))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df_importance_rf_agg.head(15), 
        x='importance', 
        y='original_feature', 
        hue='original_feature',      
        legend=False,        
        palette="viridis"
    )
    plt.title("Importancia de las Variables en el Importe (Random Forest)")
    plt.xlabel("Importancia Relativa (Impureza de Gini)")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importancia_variables_rf_importe.png", dpi=300)
    plt.close()
    
    print(f"\n✅ Gráfico de importancia guardado en {OUTPUT_DIR / 'importancia_variables_rf_importe.png'}")
    
except Exception as e:
    print(f"Error al generar la interpretabilidad: {e}")

# ------------------------------------------------------------
# 8. ANÁLISIS GRÁFICO (ERRORES Y RESIDUOS SEPARADOS CON ZOOM)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- GENERANDO GRÁFICOS DE RENDIMIENTO (Separados y con Zoom) ---")

try:
    # Destransformar para gráficos
    y_pred_rf_euros = np.expm1(np.clip(y_pred_test_rf_log, -np.inf, 30))
    y_pred_rf_euros[y_pred_rf_euros < 0] = 0

    df_results = pd.DataFrame({
        'Real': y_test_euros_real,
        'Prediccion_RF': y_pred_rf_euros,
        'Prediccion_Baseline': y_pred_baseline_euros
    })

    df_results['APE_RF'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_RF'])) / df_results['Real']
    df_results['APE_Baseline'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_Baseline'])) / df_results['Real']

    # --- 1. Gráfico Boxplot APE ---
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df_results[['APE_Baseline', 'APE_RF']],
        orient='h',
        showfliers=False,
        palette=['#4C72B0', '#E1812C'] 
    )
    plt.title('Distribución del Error Porcentual Absoluto (APE)\nBaseline vs. Random Forest (Importe)', fontsize=13, fontweight='bold')
    plt.xlabel('Error Porcentual Absoluto (APE %) - Escala Logarítmica')
    
    plt.gca().set_xscale('log')
    plt.xlim(0.1, 300) 
    
    plt.axvline(x=df_results['APE_Baseline'].median(), color='blue', linestyle='--', label=f"MdAPE Baseline ({df_results['APE_Baseline'].median():.2f}%)")
    plt.axvline(x=df_results['APE_RF'].median(), color='darkorange', linestyle='--', label=f"MdAPE RF ({df_results['APE_RF'].median():.2f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_comparativa_error_boxplot_RF.png', dpi=300)
    plt.close()

    # --- CÁLCULO DE LÍMITES PARA EL ZOOM ---
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

    # --- 3. Gráfico Scatter: RANDOM FOREST ---
    plt.figure(figsize=(8, 8))
    plt.scatter(df_results['Real'], df_results['Prediccion_RF'], alpha=0.15, s=10, color='#E1812C')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([min_val_zoom, max_val_zoom], [min_val_zoom, max_val_zoom], color='red', linestyle='--', linewidth=2, label='Predicción Perfecta (y=x)')
    plt.title("Modelo Random Forest\n(Predicción Directa Log-Log)", fontsize=14, fontweight='bold')
    plt.xlabel("Valor Real Adjudicado (€) [Escala Log]", fontsize=12)
    plt.ylabel("Valor Predicho (€) [Escala Log]", fontsize=12)
    plt.xlim(min_val_zoom, max_val_zoom)
    plt.ylim(min_val_zoom, max_val_zoom)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_scatter_RF_Zoom.png', dpi=300)
    plt.close()

    print("✅ Gráficos de rendimiento generados correctamente.")

except Exception as e:
    print(f"Error generando gráficos: {e}")

# ------------------------------------------------------------
# 9. VALORES SHAP
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- Iniciando cálculo de valores SHAP ---")
try:
    preprocessor_obj = pipeline_rf.named_steps['preprocessor']
    X_test_transformed = preprocessor_obj.transform(X_test)
    X_test_transformed_df = pd.DataFrame(
        X_test_transformed, 
        columns=feature_names_rf,
        index=X_test.index
    )

    explainer = shap.TreeExplainer(model_rf)
    print("Calculando SHAP (Muestra 2000)...")
    X_test_sample = X_test_transformed_df.sample(min(2000, len(X_test_transformed_df)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)

    # Summary Plot
    shap.summary_plot(shap_values, X_test_sample, plot_type="dot", max_display=20, show=False)
    plt.title("Impacto de Features en la Predicción (SHAP - Random Forest)")
    plt.savefig(OUTPUT_DIR / 'shap_summary_plot_RF.png', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Gráfico SHAP Summary guardado.")

except Exception as e:
    print(f"Error al generar SHAP: {e}")

print("\n🎉 ¡Script de Random Forest completado!")