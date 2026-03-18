# ------------------------------------------------------------
# analisis_eda_empresas_v2.py (ACTUALIZADO CON FILTROS TFG)
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ------------------------------------------------------------
# 1. CONFIGURACIÓN
# ------------------------------------------------------------
RUTA = Path(r"C:\Users\User\Documents\InferIA")

# ACTUALIZACIÓN: Cargamos directamente los históricos limpios
TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"
TEST_PATH  = RUTA / "test_procesado_v2_limpio.parquet"

OUTPUT_DIR = RUTA / "graficas_analisis_empresas"
OUTPUT_DIR.mkdir(exist_ok=True)

# Variables clave para la limpieza del TFG
TARGET = 'lote_importe_adjudicacion_sin_impuestos'
PRESUPUESTO = 'lote_presupuesto_base_sin_impuestos'
VALOR_ESTIMADO = 'valor_estimado_imputado'

# ------------------------------------------------------------
# 2. CARGA DE DATOS
# ------------------------------------------------------------
print("⏳ Cargando datos limpios...")
full_df = pd.concat([
    pd.read_parquet(TRAIN_PATH),
    pd.read_parquet(VAL_PATH),
    pd.read_parquet(TEST_PATH)
], ignore_index=True)

print(f"Total registros iniciales: {len(full_df)}")

# ------------------------------------------------------------
# 3. LIMPIEZA COHERENTE (FILTROS EDA TFG)
# ------------------------------------------------------------
print("🧹 Aplicando filtros de coherencia del TFG...")

# 3.1 Solo adjudicadas (éxito)
if 'es_exito' in full_df.columns: 
    full_df = full_df[full_df['es_exito'] == 1]

# 3.2 Eliminar sobrecostes ilógicos (Adjudicación > Presupuesto)
if TARGET in full_df.columns and PRESUPUESTO in full_df.columns:
    full_df = full_df[full_df[TARGET] <= full_df[PRESUPUESTO]]

# 3.3 Eliminar anomalías (Presupuesto > Valor Estimado)
if PRESUPUESTO in full_df.columns and VALOR_ESTIMADO in full_df.columns:
    full_df = full_df[full_df[PRESUPUESTO] <= full_df[VALOR_ESTIMADO]]

# 3.4 Eliminar NaNs y Ceros en el Target
if TARGET in full_df.columns:
    full_df.dropna(subset=[TARGET], inplace=True) 
    full_df = full_df[full_df[TARGET] > 0].copy()

# 3.5 Filtro de Precios Simbólicos (>= 2€)
if TARGET in full_df.columns:
    full_df = full_df[full_df[TARGET] >= 2.0]

# 3.6 Filtro de Coherencia Económica (Precios Unitarios y Acuerdos Marco)
if PRESUPUESTO in full_df.columns and VALOR_ESTIMADO in full_df.columns:
    full_df = full_df[full_df[VALOR_ESTIMADO] <= (full_df[PRESUPUESTO] * 10)]

print(f"Registros tras filtros de licitaciones: {len(full_df)}")

# ------------------------------------------------------------
# 4. EXTRACCIÓN DE PERFILES DE EMPRESA
# ------------------------------------------------------------
print("🏢 Extrayendo última foto (perfil) de cada empresa...")
# Ordenar y quedarnos con la última foto de cada empresa
full_df['fecha_primera_publicacion'] = pd.to_datetime(full_df['fecha_primera_publicacion'])
full_df = full_df.sort_values('fecha_primera_publicacion')
df_perfiles = full_df.drop_duplicates(subset=['lote_adjudicatario'], keep='last').copy()

print(f"Total empresas detectadas: {len(df_perfiles)}")

# ------------------------------------------------------------
# 5. FILTRO DE LIMPIEZA CLAVE (HISTÓRICOS)
# ------------------------------------------------------------
# Eliminamos descuentos negativos (Sobrecostes o Fillna(-1) de los historiales)
if 'descuento_medio_hist' in df_perfiles.columns:
    df_clean = df_perfiles[df_perfiles['descuento_medio_hist'] >= 0.0].copy()
else:
    df_clean = df_perfiles.copy()

print(f"Empresas válidas tras limpiar errores de historial (-100%): {len(df_clean)}")
print(f"Se han eliminado {len(df_perfiles) - len(df_clean)} empresas con datos 'sucios' en su historial.")

# A partir de aquí iría el bloque "3. GENERACIÓN DE GRÁFICAS (CON DATOS LIMPIOS)"...

# 3. GENERACIÓN DE GRÁFICAS (CON DATOS LIMPIOS)
sns.set_theme(style="whitegrid")

# --- GRÁFICA 1: Histograma Descuentos (CON MEDIA Y MEDIANA) ---
plt.figure(figsize=(10, 6))
data_desc = df_clean['descuento_medio_hist'] * 100 
sns.histplot(data_desc, bins=40, kde=True, color="teal")

# Añadimos la Mediana (robusta a extremos)
plt.axvline(data_desc.median(), color='red', linestyle='--', linewidth=2, label=f'Mediana: {data_desc.median():.2f}%')

# Añadimos la Media (sensible a extremos)
plt.axvline(data_desc.mean(), color='orange', linestyle='-', linewidth=2, label=f'Media: {data_desc.mean():.2f}%')

plt.title("Distribución de Descuentos con historial", fontsize=14, fontweight='bold')
plt.xlabel("Descuento Medio (%)", fontsize=12)
plt.ylabel("Número de Empresas", fontsize=12)

# Colocamos la leyenda
plt.legend(title="Estadísticas", frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "1_distribucion_descuentos_clean.png", dpi=150, bbox_inches='tight')
plt.close()

# --- GRÁFICA 2: Victorias (Log-Log Plot) ---
print("📊 Generando Gráfica 2 corregida...")
plt.figure(figsize=(10, 6))

# CAMBIO CLAVE: log_scale=True aplica logaritmo al eje X automáticamente.
# Esto permite ver bien tanto a los que tienen 1 victoria como a los que tienen 1000.
sns.histplot(df_clean['n_licitaciones_hist'], log_scale=True, color="orange")

plt.title("Concentración del Mercado", fontsize=14)
plt.xlabel("Nº Victorias (Escala Logarítmica: 1, 10, 100, 1000)")
plt.ylabel("Número de Empresas")

# Añadimos líneas para que se entienda la escala
plt.axvline(1, color='grey', linestyle=':', alpha=0.5)
plt.axvline(10, color='grey', linestyle=':', alpha=0.5)
plt.axvline(100, color='grey', linestyle=':', alpha=0.5)

plt.savefig(OUTPUT_DIR / "2_distribucion_victorias_loglog.png", dpi=150, bbox_inches='tight')
plt.close()

# --- GRÁFICA 3: Scatter Plot ---
plt.figure(figsize=(10, 6))
# Filtramos también presupuestos absurdamente bajos (< 100€) para limpiar el gráfico
df_scatter = df_clean[df_clean['presupuesto_medio_hist'] > 100]

sns.scatterplot(
    data=df_scatter, 
    x='presupuesto_medio_hist', 
    y='descuento_medio_hist', 
    alpha=0.2, s=20, color='purple'
)
plt.xscale('log')
#plt.ylim(0, 0.8) # Hacemos zoom entre 0% y 60% de descuento (donde está lo interesante)
plt.title("Tamaño vs Descuento (Zoom en zona operativa)", fontsize=14)
plt.xlabel("Presupuesto Medio (€) [Log]")
plt.ylabel("Descuento Medio")
plt.savefig(OUTPUT_DIR / "3_scatter_clean.png", dpi=150, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------
# NUEVAS GRÁFICAS AVANZADAS (GEO, ESPECIALIZACIÓN, EXPERIENCIA)
# ------------------------------------------------------------
print("📊 Generando Gráficas Avanzadas de Estrategia...")

# 1. PREPARACIÓN DE DATOS (Feature Engineering Express)
# Detectamos columnas dinámicamente
cols_geo = [c for c in df_clean.columns if 'pct_hist_com_aut_' in c]
cols_tipo = [c for c in df_clean.columns if 'pct_hist_' in c and 'com_aut' not in c]

# Calculamos métricas nuevas
# Cuántas CCAA distintas tienen valor > 0
df_clean['num_ccaa_operativas'] = (df_clean[cols_geo] > 0).sum(axis=1)
# Cuál es el porcentaje más alto en un solo tipo (1.0 = 100% especialista)
df_clean['grado_especializacion'] = df_clean[cols_tipo].max(axis=1) * 100


# --- GRÁFICA 4: EXPANSIÓN GEOGRÁFICA ---
plt.figure(figsize=(10, 6))
# Contamos cuántas empresas operan en 1, 2, 3... CCAA
sns.countplot(data=df_clean, x='num_ccaa_operativas', color="royalblue")

plt.title("¿Locales o Nacionales?", fontsize=14)
plt.xlabel("Número de Comunidades Autónomas donde han ganado", fontsize=12)
plt.ylabel("Número de Empresas", fontsize=12)
plt.yscale('log') # Escala log porque habrá muchísimas con 1 y pocas con 17
plt.savefig(OUTPUT_DIR / "4_expansion_geografica.png", dpi=150, bbox_inches='tight')
plt.close()


# --- GRÁFICA 5: NIVEL DE ESPECIALIZACIÓN ---
plt.figure(figsize=(10, 6))
sns.histplot(df_clean['grado_especializacion'], bins=20, color="green", kde=True)

plt.title("¿Generalistas o Especialistas?", fontsize=14)
plt.xlabel("Grado de Especialización Máxima (%)", fontsize=12)
plt.ylabel("Número de Empresas", fontsize=12)
plt.axvline(df_clean['grado_especializacion'].median(), color='red', linestyle='--', label='Mediana')
plt.legend()
plt.savefig(OUTPUT_DIR / "5_especializacion.png", dpi=150, bbox_inches='tight')
plt.close()


# --- GRÁFICA 6: EXPERIENCIA VS AGRESIVIDAD (CON LEYENDA CORREGIDA) ---
plt.figure(figsize=(10, 6))

# 1. Nube de puntos (Empresas individuales)
sns.scatterplot(
    data=df_clean, 
    x='n_licitaciones_hist', 
    y='descuento_medio_hist', 
    alpha=0.3, 
    color='darkred',
    s=30,
    label='Empresas individuales'
)

# 2. Curva de Tendencia LOWESS
try:
    sns.regplot(
        data=df_clean, 
        x='n_licitaciones_hist', 
        y='descuento_medio_hist', 
        scatter=False, 
        lowess=True, 
        color="blue", 
        line_kws={'linestyle':'--', 'linewidth': 2}
    )
except:
    pass

# 3. TRUCO PARA LA LEYENDA: Línea invisible para forzar la etiqueta
plt.plot([], [], color='blue', linestyle='--', linewidth=2, label='Media móvil')

# Configuraciones de ejes y títulos
plt.xscale('log') # Logarítmica para ver bien a los pequeños y grandes
plt.title("Curva de Aprendizaje: Experiencia vs Agresividad", fontsize=14, fontweight='bold')
plt.xlabel("Nº de Victorias (Escala Log)", fontsize=12)
plt.ylabel("Descuento Medio (%)", fontsize=12)

# Mostrar la leyenda
plt.legend(title="Leyenda", loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "6_experiencia_vs_descuento_completa.png", dpi=150)
plt.close()

print("✅ ¡Gráfica 6 generada con la leyenda corregida!")

print("✅ ¡Nuevas gráficas generadas!")
