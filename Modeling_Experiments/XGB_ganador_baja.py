import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
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

TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"
TEST_PATH  = RUTA / "test_procesado_v2_limpio.parquet"

TARGET = 'pct_baja'
PRESUPUESTO = 'lote_presupuesto_base_sin_impuestos' 
IMPORTE_COL = 'lote_importe_adjudicacion_sin_impuestos'

OUTPUT_DIR = RUTA / "salida_modelo_XGB_Baja"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- RELLENA CON LO QUE TE SALGA DEL TUNEO (M2 - BAJA) ---
config_m2 = {
    "n_estimators": 235,       # Reemplazar con best_n_trees
    "learning_rate": 0.05,     
    "max_depth": 15,          
    "reg_lambda": 40,
    "gamma": 0,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

def mdape(y_true, y_pred):
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
    start_len = len(df)
    VALOR_ESTIMADO = 'valor_estimado_imputado'
    
    # 🚨 1. FÓRMULA EXPLÍCITA: Calculamos la baja (0 a 100)
    if IMPORTE_COL in df.columns and PRESUPUESTO in df.columns:
        df[TARGET] = ((df[PRESUPUESTO] - df[IMPORTE_COL]) / df[PRESUPUESTO]) * 100
        
    if 'es_exito' in df.columns: df = df[df['es_exito'] == 1].copy()
    
    if PRESUPUESTO in df.columns and VALOR_ESTIMADO in df.columns:
        df = df[df[PRESUPUESTO] <= df[VALOR_ESTIMADO]].copy()
        df = df[df[VALOR_ESTIMADO] <= (df[PRESUPUESTO] * 10)].copy()
        
    if TARGET in df.columns:
        df.dropna(subset=[TARGET], inplace=True) 
        df = df[(df[TARGET] >= 0.0) & (df[TARGET] <= 100.0)].copy() 
        
    print(f"  Limpieza: {start_len} -> {len(df)} filas.")
    return df

print("\n🧹 Aplicando limpieza estricta y calculando bajas...")
full_train_df = limpiar_df_estricto(full_train_df)
test_df = limpiar_df_estricto(test_df)

# ------------------------------------------------------------
# 3. PREPARACIÓN DE TARGETS Y HELPER PARA METRICAS
# ------------------------------------------------------------
y_test_euros_real = test_df[IMPORTE_COL].values
presupuesto_test  = test_df[PRESUPUESTO].values

DROP_COLS = [
    '_id', 'id', 'fecha_primera_publicacion', 'objeto', 'lote_objeto', 
    'organo_contratacion', 'lote_adjudicatario', 'lote_resultado', 
    'cpv_final_imputado', 'cif_normalizado', 'lote_importe_adjudicacion_con_impuestos',
    IMPORTE_COL, # <--- BLINDAJE ANTI DATA-LEAKAGE
    'es_exito', 'es_sobrecoste', 'lote_numero_ofertas_recibidas', 
    'presupuesto_medio_hist', 'descuento_promedio', 
    'lote_precio_oferta_mas_alta', 'lote_precio_oferta_mas_baja', 'presupuesto_base_sin_impuestos',
    'duracion_proceso_dias', 'es_anomalia_temporal'
]

full_train_df.drop(columns=[c for c in DROP_COLS if c in full_train_df.columns], inplace=True, errors='ignore')
test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns], inplace=True, errors='ignore')

# Separar X e Y (SIN LOGARITMOS)
X_train = full_train_df.drop(columns=[TARGET], errors='ignore')
y_train = full_train_df[TARGET].copy()

X_test = test_df.drop(columns=[TARGET], errors='ignore')
y_test_baja = test_df[TARGET].copy()

def evaluar_en_euros(modelo_nombre, y_pred_baja):
    y_pred_baja_segura = np.clip(y_pred_baja, 0.0, 100.0)
    y_pred_euros = presupuesto_test - (y_pred_baja_segura / 100.0) * presupuesto_test
    
    return {
        'Modelo': modelo_nombre,
        'R²': r2_score(y_test_euros_real, y_pred_euros),
        'MAE (€)': mean_absolute_error(y_test_euros_real, y_pred_euros),
        'MdAPE (%)': mdape(y_test_euros_real, y_pred_euros)
    }

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
# 4. MODELO 0: BASELINE ABSOLUTO (0% Baja)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- MODELO 0: BASELINE ABSOLUTO (Asume 0% Baja) ---")

y_pred_baseline_baja = np.zeros(len(test_df))
metrics_baseline = evaluar_en_euros('Baseline Absoluto (Test)', y_pred_baseline_baja)


# ------------------------------------------------------------
# 4.5 EXTRA: CURVA DE APRENDIZAJE (EN EUROS REALES)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- GENERANDO GRÁFICO DE VALIDACIÓN EN EUROS REALES ---")
try:
    import sys, os
    
    # 1. Recuperamos train y val y SILENCIAMOS la consola
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    train_curva = limpiar_df_estricto(pd.read_parquet(TRAIN_PATH))
    val_curva = limpiar_df_estricto(pd.read_parquet(VAL_PATH))
    sys.stdout = old_stdout 

    # 🚨 EXTRAEMOS LOS EUROS REALES Y PRESUPUESTOS ANTES DE BORRAR COLUMNAS
    presupuesto_val_curva = val_curva[PRESUPUESTO].values
    y_val_euros_real_curva = val_curva[IMPORTE_COL].values

    X_t = train_curva.drop(columns=[TARGET] + [c for c in DROP_COLS if c in train_curva.columns], errors='ignore')
    y_t = train_curva[TARGET]
    X_v = val_curva.drop(columns=[TARGET] + [c for c in DROP_COLS if c in val_curva.columns], errors='ignore')
    y_v = val_curva[TARGET]

    preprocessor.fit(X_t)
    X_t_trans = preprocessor.transform(X_t)
    X_v_trans = preprocessor.transform(X_v)

    # 3. Entrenamos el modelo hasta 550 árboles
    print("Entrenando modelo para la curva...")
    modelo_curva = XGBRegressor(
        **{k: v for k, v in config_m2.items() if k != 'n_estimators'},
        n_estimators=550,
        random_state=42,
        n_jobs=-1
    )
    modelo_curva.fit(X_t_trans, y_t, verbose=False)

# 4. BUCLE MÁGICO: Calculamos el MAE en EUROS árbol por árbol
    print("Calculando MAE en Euros iteración por iteración...")
    
    rango_arboles = range(50, 551)
    mae_euros_history = []

    for i in rango_arboles:
        pred_pct = modelo_curva.predict(X_v_trans, iteration_range=(0, i))
        pred_pct_segura = np.clip(pred_pct, 0.0, 100.0)
        pred_euros = presupuesto_val_curva - (pred_pct_segura / 100.0) * presupuesto_val_curva
        
        mae_e = mean_absolute_error(y_val_euros_real_curva, pred_euros)
        mae_euros_history.append(mae_e)

    # --- ENCONTRAR EL MÍNIMO REAL AUTOMÁTICAMENTE ---
    mejor_mae_euros = min(mae_euros_history)
    mejor_iteracion_euros = rango_arboles[mae_euros_history.index(mejor_mae_euros)]
    print(f"¡ATENCIÓN! El verdadero óptimo económico está en el árbol: {mejor_iteracion_euros}")
    print(f"MAE en ese punto: {mejor_mae_euros:,.2f} €")

    # 5. GENERAMOS EL GRÁFICO FINAL
    plt.figure(figsize=(10, 6))
    
    plt.plot(list(rango_arboles), mae_euros_history, color='#D62728', linewidth=2.5, label='MAE de Validación (€)')
    plt.axvline(x=mejor_iteracion_euros, color='black', linestyle='--', label=f'Punto Óptimo Económico ({mejor_iteracion_euros})')
    
    plt.title('Evolución del Error Económico en XGBoost (Detalle iteraciones 50-550)', fontsize=14, fontweight='bold')
    plt.xlabel('Número de Iteraciones (Árboles)', fontsize=12)
    plt.ylabel('Error Absoluto Medio (MAE en Euros)', fontsize=12)
    plt.legend()
    
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_DIR / 'Curva_Early_Stopping_XGB_Baja.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("  -> Gráfico guardado con el mínimo correcto.")

except Exception as e:
    print(f"Error generando la curva: {e}")
    
# ------------------------------------------------------------
# 5. MODELO 1: XGBOOST GANADOR (PORCENTAJE BAJA)
# ------------------------------------------------------------
print("\n" + "="*60)
print(f"--- MODELO 1: XGBoost Ganador (Baja) ---")

pipeline_xgb = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(
        **config_m2,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline_xgb.fit(X_train, y_train)
y_pred_test_xgb_baja = pipeline_xgb.predict(X_test)
metrics_xgb = evaluar_en_euros('XGBoost Baja -> Euros (Test)', y_pred_test_xgb_baja)

joblib.dump(pipeline_xgb, OUTPUT_DIR / "xgb_ganador_baja.pkl")


# ------------------------------------------------------------
# 5.5 DIAGNÓSTICO DE OVERFITTING EN ESCALA NATIVA (% DE BAJA)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- DIAGNÓSTICO DE OVERFITTING (XGBOOST - ESCALA NATIVA) ---")

# 1. Predecir sobre los datos de entrenamiento
y_pred_train_xgb_baja = pipeline_xgb.predict(X_train)
# La predicción de test (y_pred_test_xgb_baja) ya está calculada en el paso 5

# 2. Calculamos el MAE directamente sobre la variable objetivo real (%)
mae_train_pct = mean_absolute_error(y_train, y_pred_train_xgb_baja)
mae_test_pct = mean_absolute_error(y_test_baja, y_pred_test_xgb_baja)

print(f"MAE Train: {mae_train_pct:.2f}% |  MAE Test: {mae_test_pct:.2f}%")

if mae_train_pct > 0:
    diferencia_mae_pct = ((mae_test_pct - mae_train_pct) / mae_train_pct) * 100
    print(f"-> Degradación del MAE en Test: {diferencia_mae_pct:+.1f}%")

# ------------------------------------------------------------
# 6. TABLA COMPARATIVA FINAL
# ------------------------------------------------------------
print("\n" + "="*60)
print("🏆 RESUMEN FINAL XGBOOST (Predicción de Bajas Traducidas a EUROS) 🏆")
df_final = pd.DataFrame([metrics_baseline, metrics_xgb])
print(df_final[['Modelo', 'R²', 'MAE (€)', 'MdAPE (%)']].to_markdown(index=False, floatfmt=",.4f"))
df_final.to_csv(OUTPUT_DIR / "tabla_resultados_final_XGB_baja.csv", index=False)

# ------------------------------------------------------------
# 7. EXPLICABILIDAD (IMPORTANCIA DE VARIABLES)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- INTERPRETACIÓN DE VARIABLES (XGBoost) ---")
try:
    feature_names_xgb = pipeline_xgb.named_steps['preprocessor'].get_feature_names_out()
    model_xgb = pipeline_xgb.named_steps['model']
    importances_xgb = model_xgb.feature_importances_

    df_importance_xgb_raw = pd.DataFrame({
        'feature': feature_names_xgb,
        'importance': importances_xgb
    }).sort_values('importance', ascending=False)
    
    def get_original_feature(col_name, cat_cols_list):
        if col_name.startswith('num__'): return col_name.replace('num__', '')
        if col_name.startswith('cat__'):
            for cat_col in cat_cols_list:
                if col_name.startswith(f'cat__{cat_col}_'): return cat_col
        return col_name

    df_importance_xgb_raw['original_feature'] = df_importance_xgb_raw['feature'].apply(
        lambda x: get_original_feature(x, cat_cols)
    )
    df_importance_xgb_agg = df_importance_xgb_raw.groupby('original_feature')['importance'].sum()
    df_importance_xgb_agg = df_importance_xgb_agg.sort_values(ascending=False).reset_index()

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=df_importance_xgb_agg.head(15), 
        x='importance', 
        y='original_feature', 
        hue='original_feature',      
        legend=False,        
        palette="magma" # Magma para diferenciarlo del target Importe
    )
    plt.title("Importancia de las Variables en el % de Baja (XGBoost)")
    plt.xlabel("Importancia Relativa (Gain)")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "importancia_variables_xgb_baja.png", dpi=300)
    plt.close()
    
except Exception as e:
    print(f"Error al generar la interpretabilidad: {e}")

# ------------------------------------------------------------
# 8. ANÁLISIS GRÁFICO (ERRORES Y RESIDUOS)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- GENERANDO GRÁFICOS DE RENDIMIENTO ---")
try:
    # Destransformar para gráficos
    y_pred_xgb_euros = presupuesto_test - (np.clip(y_pred_test_xgb_baja, 0.0, 100.0) / 100.0) * presupuesto_test
    y_pred_baseline_euros = presupuesto_test 

    df_results = pd.DataFrame({
        'Real': y_test_euros_real,
        'Prediccion_XGB': y_pred_xgb_euros,
        'Prediccion_Baseline': y_pred_baseline_euros
    })

    df_results['APE_XGB'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_XGB'])) / df_results['Real']
    df_results['APE_Baseline'] = 100 * (np.abs(df_results['Real'] - df_results['Prediccion_Baseline'])) / df_results['Real']

    # --- 1. Gráfico Boxplot APE ---
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df_results[['APE_Baseline', 'APE_XGB']],
        orient='h',
        showfliers=False,
        palette=['#4C72B0', '#D62728'] 
    )
    plt.title('Distribución del Error Porcentual Absoluto (APE)\nBaseline vs. XGBoost Baja', fontsize=13, fontweight='bold')
    plt.xlabel('Error Porcentual Absoluto (APE %) - Escala Logarítmica')
    plt.gca().set_xscale('log')
    plt.xlim(0.1, 300) 
    
    plt.axvline(x=df_results['APE_Baseline'].median(), color='blue', linestyle='--', label=f"MdAPE Baseline ({df_results['APE_Baseline'].median():.2f}%)")
    plt.axvline(x=df_results['APE_XGB'].median(), color='red', linestyle='--', label=f"MdAPE XGB ({df_results['APE_XGB'].median():.2f}%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '1_comparativa_error_boxplot_XGB_Baja.png', dpi=300)
    plt.close()

    min_val_zoom = max(10.0, df_results['Real'].quantile(0.005)) 
    max_val_zoom = df_results['Real'].quantile(0.995)

    # --- 3. Gráfico Scatter: XGBOOST ---
    plt.figure(figsize=(8, 8))
    plt.scatter(df_results['Real'], df_results['Prediccion_XGB'], alpha=0.15, s=10, color='#D62728')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([min_val_zoom, max_val_zoom], [min_val_zoom, max_val_zoom], color='black', linestyle='--', linewidth=2, label='Predicción Perfecta (y=x)')
    plt.title("Modelo XGBoost\n(Baja predicha destransformada a Euros)", fontsize=14, fontweight='bold')
    plt.xlabel("Valor Real Adjudicado (€) [Escala Log]", fontsize=12)
    plt.ylabel("Valor Predicho (€) [Escala Log]", fontsize=12)
    plt.xlim(min_val_zoom, max_val_zoom)
    plt.ylim(min_val_zoom, max_val_zoom)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3_scatter_XGB_Baja_Zoom.png', dpi=300)
    plt.close()

except Exception as e:
    print(f"Error generando gráficos: {e}")

# ------------------------------------------------------------
# 10. LA PRUEBA DEL ALGODÓN: XGBOOST (% PREDICHO VS % REAL)
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- GENERANDO SCATTER PLOT 'DESNUDO' PARA XGBOOST ---")

try:
    # Nos aseguramos de que no hay valores locos fuera de rango (0-100)
    y_pred_xgb_pct_limpio = np.clip(y_pred_test_xgb_baja, 0.0, 100.0)
    
    # Los valores reales en test
    y_real_pct = y_test_baja.values if hasattr(y_test_baja, 'values') else y_test_baja

    # Dibujamos el gráfico
    plt.figure(figsize=(8, 8))
    
    # Usamos el color rojo intenso que le dimos a XGBoost
    plt.scatter(y_real_pct, y_pred_xgb_pct_limpio, alpha=0.15, s=10, color='#D62728')
    
    # La diagonal perfecta
    plt.plot([-5, 60], [-5, 60], color='black', linestyle='--', linewidth=2, label='Predicción Perfecta (y=x)')
    
    # Ajustes estéticos (idénticos a los del TabM "malo")
    plt.title("La Prueba del Algodón: XGBoost\n(Baja PREDICHA % vs Baja REAL %)", fontsize=14, fontweight='bold')
    plt.xlabel("Baja REAL (%)", fontsize=12)
    plt.ylabel("Baja PREDICHA (%)", fontsize=12)
    
    # Fijamos los límites para que sea exactamente comparable a la imagen de TabM
    plt.xlim(-5, 60)
    plt.ylim(-5, 60)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    
    # Guardamos la imagen
    plt.savefig(OUTPUT_DIR / '10_Prueba_Algodon_XGB_Baja_Pct.png', dpi=300)
    plt.close() # Importante usar close en scripts locales
    print("  -> Gráfico 'Prueba del Algodón' guardado correctamente.")

except Exception as e:
    print(f"Error generando el gráfico de la prueba del algodón: {e}")

# ------------------------------------------------------------
# 9. VALORES SHAP
# ------------------------------------------------------------
print("\n" + "="*60)
print("--- Iniciando cálculo de valores SHAP ---")
try:
    preprocessor_obj = pipeline_xgb.named_steps['preprocessor']
    X_test_transformed = preprocessor_obj.transform(X_test)
    X_test_transformed_df = pd.DataFrame(
        X_test_transformed, 
        columns=feature_names_xgb,
        index=X_test.index
    )

    explainer = shap.TreeExplainer(model_xgb)
    print("Calculando SHAP (Muestra 2000)...")
    X_test_sample = X_test_transformed_df.sample(min(2000, len(X_test_transformed_df)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)

    shap.summary_plot(shap_values, X_test_sample, plot_type="dot", max_display=20, show=False)
    plt.title("Impacto de Features en la Predicción (SHAP - XGBoost Baja)")
    plt.savefig(OUTPUT_DIR / 'shap_summary_plot_XGB_Baja.png', bbox_inches='tight', dpi=300)
    plt.close()
    
except Exception as e:
    print(f"Error al generar SHAP: {e}")

print("\n🎉 ¡Script de XGBoost (Porcentaje de Baja) completado!")