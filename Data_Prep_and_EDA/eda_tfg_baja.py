import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=UserWarning)

# ------------------------------------------------------------
# 1. CONFIGURACIÓN Y RUTAS
# ------------------------------------------------------------
# Estilo de gráficos profesional para el TFG
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

RUTA = Path(r"C:\Users\User\Documents\InferIA")

# Archivos procesados estándar
TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"
TEST_PATH  = RUTA / "test_procesado_v2_limpio.parquet"

IMPORTE_ADJ = 'lote_importe_adjudicacion_sin_impuestos'
PRESUPUESTO = 'lote_presupuesto_base_sin_impuestos'
VALOR_ESTIMADO = 'valor_estimado_imputado'

# Carpeta de salida para no mezclar con las del importe
OUTPUT_DIR = RUTA / "graficas_eda_baja_tfg"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 2. CARGA DE DATOS Y UNIÓN (Solo para EDA)
# ------------------------------------------------------------
print("Cargando datasets...")
train_df = pd.read_parquet(TRAIN_PATH)
val_df   = pd.read_parquet(VAL_PATH)
test_df  = pd.read_parquet(TEST_PATH)

df_eda = pd.concat([train_df, val_df, test_df], ignore_index=True)
start_len = len(df_eda)
print(f"Total registros iniciales: {start_len}")

# ------------------------------------------------------------
# 3. LIMPIEZA BASE (¡EXACTAMENTE IGUAL QUE EN EL IMPORTE!)
# ------------------------------------------------------------
print("Aplicando filtros de coherencia estricta...")

if 'es_exito' in df_eda.columns:
    df_eda = df_eda[df_eda['es_exito'] == 1]

if IMPORTE_ADJ in df_eda.columns and PRESUPUESTO in df_eda.columns:
    df_eda = df_eda[df_eda[IMPORTE_ADJ] <= df_eda[PRESUPUESTO]]

if PRESUPUESTO in df_eda.columns and VALOR_ESTIMADO in df_eda.columns:
    df_eda = df_eda[df_eda[PRESUPUESTO] <= df_eda[VALOR_ESTIMADO]]
    
df_eda.dropna(subset=[IMPORTE_ADJ], inplace=True) 
df_eda = df_eda[df_eda[IMPORTE_ADJ] > 0].copy() 

# Filtro de Precios Simbólicos
UMBRAL_MINIMO_EUROS = 2.0 
df_eda = df_eda[df_eda[IMPORTE_ADJ] >= UMBRAL_MINIMO_EUROS]

# Filtro de Coherencia Económica (Precios Unitarios y Acuerdos Marco)
df_eda = df_eda[df_eda[VALOR_ESTIMADO] <= (df_eda[PRESUPUESTO] * 10)]

print(f"Registros finales tras filtros de negocio: {len(df_eda)}")

# ------------------------------------------------------------
# 3.5. CREACIÓN DEL TARGET: PORCENTAJE DE BAJA
# ------------------------------------------------------------
# Calculamos la baja (0 a 1) y la pasamos a porcentaje (0 a 100)
df_eda['pct_baja_100'] = ((df_eda[PRESUPUESTO] - df_eda[IMPORTE_ADJ]) / df_eda[PRESUPUESTO]) * 100

# ------------------------------------------------------------
# 4. GENERACIÓN DE GRÁFICAS (EDA PARA LA BAJA)
# ------------------------------------------------------------
print("\nGenerando gráficas de Análisis Exploratorio para % de Baja...")

# --- Gráfica 1: Distribución del % de Baja (Histograma) ---
print("  1. Histograma del % de Baja...")
plt.figure(figsize=(10, 6))

media_baja = df_eda['pct_baja_100'].mean()
mediana_baja = df_eda['pct_baja_100'].median()

# Para el % de baja, no usamos escala logarítmica en el eje X porque está acotado de 0 a 100
sns.histplot(df_eda['pct_baja_100'], bins=50, color='#4C72B0', edgecolor="white", alpha=0.8)

plt.axvline(media_baja, color='#C44E52', linestyle='--', linewidth=2, label=f'Media: {media_baja:.2f} %')
plt.axvline(mediana_baja, color='#55A868', linestyle='-', linewidth=2, label=f'Mediana: {mediana_baja:.2f} %')

plt.title('Distribución del Porcentaje de Baja (%)', fontsize=14, fontweight='bold')
plt.xlabel('Porcentaje de Baja (%)')
plt.ylabel('Frecuencia (Nº de Licitaciones)')
plt.xlim(-2, 100) # Acotamos visualmente de 0 a 100
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_distribucion_pct_baja.png', dpi=300)
plt.close()

# --- Gráfica 2: Matriz de Correlación (Con variables históricas) ---
print("  2. Matriz de Correlación (Baja vs Variables Históricas)...")

cols_corr_baja = [
    'pct_baja_100', 
    'descuento_medio_hist', 
    'duracion_proceso_dias',
    PRESUPUESTO, 
    'n_licitaciones_hist',
    'dias_desde_ultima_licitacion'
]

cols_corr_baja = [c for c in cols_corr_baja if c in df_eda.columns]

if len(cols_corr_baja) > 1:
    nombres_limpios_baja = {
        'pct_baja_100': '% Baja (Target)', 
        'descuento_medio_hist': 'Baja Histórica Media', 
        'duracion_proceso_dias': 'Duración Proceso',
        PRESUPUESTO: 'Presupuesto Base', 
        'n_licitaciones_hist': 'Nº Licitaciones Hist.',
        'dias_desde_ultima_licitacion': 'Días Última Licit.'
    }
    df_corr_baja = df_eda[cols_corr_baja].rename(columns=nombres_limpios_baja)
    
    plt.figure(figsize=(9, 8))
    # Spearman es mejor aquí porque las relaciones no suelen ser perfectamente lineales
    sns.heatmap(df_corr_baja.corr(method='spearman'), annot=True, cmap='vlag', fmt=".3f", 
                vmin=-1, vmax=1, square=True, annot_kws={"size": 11})
    plt.title("Correlación de Spearman (Target % Baja vs Negocio)", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_correlacion_spearman_baja.png', dpi=300)
    plt.close()


# --- Gráfica 3: Boxplots Categóricos (% Baja) ---
print("  3. Generando Boxplots Categóricos de la Baja...")
variables_categoricas = ['tipo_contrato', 'tipo_procedimiento']

for var in variables_categoricas:
    if var in df_eda.columns:
        plt.figure(figsize=(12, 7))
        
        # Ordenar categorías por la mediana de la baja para que tenga sentido visual
        orden = df_eda.groupby(var)['pct_baja_100'].median().sort_values(ascending=False).index
        
        sns.boxplot(
            data=df_eda, x='pct_baja_100', y=var, order=orden, palette='Set2',
            showfliers=False # Quitamos los outliers visualmente para ver mejor el cuerpo de las cajas
        )
        
        plt.title(f'Distribución del Porcentaje de Baja por {var.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.xlabel('Porcentaje de Baja (%)')
        plt.ylabel('') 
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'3_boxplot_baja_{var}.png', dpi=300)
        plt.close()

print(f"\n¡EDA del Porcentaje de Baja finalizado con éxito!\nGráficas guardadas en: {OUTPUT_DIR}")
