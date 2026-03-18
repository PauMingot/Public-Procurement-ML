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

TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"
TEST_PATH  = RUTA / "test_procesado_v2_limpio.parquet"

TARGET = 'lote_importe_adjudicacion_sin_impuestos'
PRESUPUESTO = 'lote_presupuesto_base_sin_impuestos'
VALOR_ESTIMADO = 'valor_estimado_imputado'

OUTPUT_DIR = RUTA / "graficas_eda_tfg"
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# 2. CARGA DE DATOS Y UNIÓN (Solo para EDA)
# ------------------------------------------------------------
print("Cargando datasets...")
train_df = pd.read_parquet(TRAIN_PATH)
val_df   = pd.read_parquet(VAL_PATH)
test_df  = pd.read_parquet(TEST_PATH)

# Unimos todo para tener la imagen global en el EDA
df_eda = pd.concat([train_df, val_df, test_df], ignore_index=True)
start_len = len(df_eda)
print(f"Total registros iniciales: {start_len}")

# ------------------------------------------------------------
# 3. LIMPIEZA BASE PARA EL EDA
# ------------------------------------------------------------
print("Aplicando filtros de coherencia...")

# 1. Solo adjudicadas (éxito)
if 'es_exito' in df_eda.columns:
    df_eda = df_eda[df_eda['es_exito'] == 1]

# 2. Eliminar sobrecostes ilógicos (Adjudicación > Presupuesto)
if TARGET in df_eda.columns and PRESUPUESTO in df_eda.columns:
    df_eda = df_eda[df_eda[TARGET] <= df_eda[PRESUPUESTO]]

# 3. Eliminar anomalías (Presupuesto > Valor Estimado)
if PRESUPUESTO in df_eda.columns and VALOR_ESTIMADO in df_eda.columns:
    df_eda = df_eda[df_eda[PRESUPUESTO] <= df_eda[VALOR_ESTIMADO]]
    
# 4. Eliminar NaNs y Ceros en el Target
df_eda.dropna(subset=[TARGET], inplace=True) 
df_eda = df_eda[df_eda[TARGET] > 0].copy() 

print(f"Registros tras limpieza: {len(df_eda)} (Se han filtrado {start_len - len(df_eda)} anómalos)")

# --- NUEVOS FILTROS DE CORTE QUIRÚRGICO ---

# 5. Filtro de Precios Simbólicos
# Eliminamos exclusivamente los contratos de 0.01€, 1.00€ o céntimos sueltos,
# pero conservamos los contratos reales de importes bajos (ej. 40€ o 90€).
UMBRAL_MINIMO_EUROS = 2.0 
df_eda = df_eda[df_eda[TARGET] >= UMBRAL_MINIMO_EUROS]

# 6. Filtro de Coherencia Económica (Precios Unitarios y Acuerdos Marco)
# Si el Valor Estimado es muchísimo mayor que el Presupuesto (ej. > 10 veces),
# estamos ante la licitación de un precio por unidad y no el valor total.
df_eda = df_eda[df_eda[VALOR_ESTIMADO] <= (df_eda[PRESUPUESTO] * 10)]

print(f"Registros finales tras filtros de negocio: {len(df_eda)}")
print(f"Se han filtrado {start_len - len(df_eda)} anomalías estructurales y precios simbólicos.")

# ------------------------------------------------------------
# 4. GENERACIÓN DE GRÁFICAS (EDA)
# ------------------------------------------------------------
print("\nGenerando gráficas de Análisis Exploratorio...")

# --- Gráfica 1: Distribución del Target (Importe Adjudicación) ---
print(" 1. Histograma del Importe (Escala Logarítmica)...")
plt.figure(figsize=(10, 6))

media = df_eda[TARGET].mean()
mediana = df_eda[TARGET].median()

# Usamos log_scale=True para normalizar visualmente los datos atípicos
sns.histplot(df_eda[TARGET], bins=80, kde=True, log_scale=True, color='#4C72B0', alpha=0.7)

# Añadimos líneas para la media y mediana
plt.axvline(media, color='#C44E52', linestyle='--', linewidth=2, label=f'Media: {media:,.0f} €')
plt.axvline(mediana, color='#55A868', linestyle='-', linewidth=2, label=f'Mediana: {mediana:,.0f} €')

plt.title('Distribución del Importe de Adjudicación (Escala Logarítmica)', fontsize=14, fontweight='bold')
plt.xlabel('Importe (€) - Escala Log')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '1_distribucion_importe_log.png', dpi=300)
plt.close()

# --- Gráfica 2: Scatterplot: Presupuesto vs Adjudicación (CON ZOOM) ---
print(" 2. Scatterplot: Presupuesto vs Adjudicación (Zoom P99)...")
if PRESUPUESTO in df_eda.columns:
    plt.figure(figsize=(8, 8))
    
    # Calculamos el percentil 99 para hacer el corte (zoom)
    limite_superior_x = df_eda[PRESUPUESTO].quantile(0.99)
    limite_superior_y = df_eda[TARGET].quantile(0.99)
    # Límite absoluto máximo para el gráfico
    max_plot = max(limite_superior_x, limite_superior_y)
    
    # Hacemos el scatter
    sns.scatterplot(data=df_eda, x=PRESUPUESTO, y=TARGET, alpha=0.15, color='#4C72B0', s=12)
    
    # Línea teórica y = x
    plt.plot([1, max_plot], [1, max_plot], color='#C44E52', linestyle='--', linewidth=2, label='Límite y=x (Sin Baja)')
    
    # Aplicamos el ZOOM matemático a los ejes
    plt.xlim(df_eda[PRESUPUESTO].quantile(0.01), max_plot)
    plt.ylim(df_eda[TARGET].quantile(0.01), max_plot)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Presupuesto vs Importe Adjudicación\n(Zoom: excluyendo el 1% más extremo)', fontsize=13, fontweight='bold')
    plt.xlabel('Presupuesto Base (€) - Log')
    plt.ylabel('Importe Adjudicación (€) - Log')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '2_scatter_presupuesto_adjudicacion_zoom.png', dpi=300)
    plt.close()

# --- Gráfica 3: Matrices de Correlación (Magnitudes Económicas y Lotes) ---
print(" 3. Matrices de Correlación (Variables Clave y Lotes)...")

# Añadimos las variables de la estructura de lotes (Sin Data Leakage)
cols_corr_seguras = [
    TARGET,               # lote_importe_adjudicacion_sin_impuestos
    PRESUPUESTO,          # lote_presupuesto_base_sin_impuestos
    VALOR_ESTIMADO,       # valor_estimado_imputado
    'peso_relativo_lote', # % que representa el lote sobre el total
    'es_loteado',         # Binaria (1 = sí, 0 = no)
    'lote'                # Número/ID del lote
]

# Nos aseguramos de que existan en el DataFrame antes de graficar
cols_corr_seguras = [c for c in cols_corr_seguras if c in df_eda.columns]

if len(cols_corr_seguras) > 1:
    # Renombramos para que el gráfico quede profesional y legible en LaTeX
    nombres_limpios = {
        TARGET: 'Importe Adjudicación', 
        PRESUPUESTO: 'Presupuesto Base', 
        VALOR_ESTIMADO: 'Valor Estimado',
        'peso_relativo_lote': 'Peso Relativo Lote',
        'es_loteado': 'Es Loteado',
        'lote': 'Nº de sublotes'
    }
    df_corr = df_eda[cols_corr_seguras].rename(columns=nombres_limpios)
    
    # Aumentamos el tamaño a 8x7 para acomodar las 6 variables sin que se aprieten
    plt.figure(figsize=(8, 7))
    
    # Pearson (Lineal)
    sns.heatmap(df_corr.corr(method='pearson'), annot=True, cmap='vlag', fmt=".3f", 
                vmin=-1, vmax=1, square=True, annot_kws={"size": 11})
    plt.title("Matriz de correlación de Pearson (relaciones lineales)", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3a_correlacion_pearson_lotes.png', dpi=300)
    plt.close()

    # Spearman (No Lineal / Monótona)
    plt.figure(figsize=(8, 7))
    sns.heatmap(df_corr.corr(method='spearman'), annot=True, cmap='vlag', fmt=".3f", 
                vmin=-1, vmax=1, square=True, annot_kws={"size": 11})
    plt.title("Matriz de correlación de Spearman (relaciones monótonas)", fontsize=14, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / '3b_correlacion_spearman_lotes.png', dpi=300)
    plt.close()
    
    print("    Gráficas de correlación (con lotes) generadas con éxito.")
else:
    print(" No hay suficientes columnas para generar correlaciones.")
    
    
# --- Gráfica 4: Boxplots Categóricos ---
print(" 4. Generando Boxplots Categóricos...")
variables_categoricas = ['tipo_contrato', 'organo_cluster_label', 'tipo_procedimiento', 'sistema_contratacion']

for var in variables_categoricas:
    if var in df_eda.columns:
        plt.figure(figsize=(12, 7))
        
        # Ordenar categorías por la mediana del importe (ayuda a visualizar mejor)
        orden = df_eda.groupby(var)[TARGET].median().sort_values(ascending=False).index
        
        # Paleta de colores atractiva
        sns.boxplot(
            data=df_eda, x=TARGET, y=var, order=orden, palette='Set2',
            showfliers=True, fliersize=2, flierprops={"marker": "o", "alpha": 0.3, "color": "gray"}
        )
        
        plt.xscale('log')
        plt.title(f'Distribución del Importe por {var.replace("_", " ").title()}', fontsize=14, fontweight='bold')
        plt.xlabel('Importe Adjudicación (€) - Escala Log')
        plt.ylabel('') # Quitamos el nombre del eje Y porque las etiquetas ya lo explican
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'4_boxplot_{var}.png', dpi=300)
        plt.close()

print(f"\n¡EDA finalizado con éxito! Todas las gráficas se han guardado en la carpeta:\n{OUTPUT_DIR}")


# ------------------------------------------------------------
# 5. INSPECCIÓN DE ANOMALÍAS (LICITACIONES CERCANAS A CERO)
# ------------------------------------------------------------
print("\n--- 🕵️ INSPECCIÓN DE ANOMALÍAS: ADJUDICACIONES MÁS BAJAS ---")

# Seleccionamos las columnas que quieres inspeccionar
columnas_inspeccion = [
    TARGET, 
    PRESUPUESTO, 
    VALOR_ESTIMADO, 
    'tipo_contrato', 
    'tipo_procedimiento'
]

# Aseguramos que las columnas existen en el dataset
columnas_presentes = [c for c in columnas_inspeccion if c in df_eda.columns]

if len(columnas_presentes) > 0:
    # Ordenamos de menor a mayor importe y sacamos el Top 10
    top_10_mas_baratas = df_eda[columnas_presentes].sort_values(by=TARGET, ascending=True).head(10)
    
    print("Top 10 Licitaciones con menor importe de adjudicación:")
    # Usamos to_markdown para que se imprima bonito en la consola
    print(top_10_mas_baratas.to_markdown(index=False, floatfmt=",.2f"))
    
    # Opcional: Guardarlo en un CSV por si lo quieres adjuntar en un anexo de tu TFG
    top_10_mas_baratas.to_csv(OUTPUT_DIR / "top_10_licitaciones_mas_baratas.csv", index=False)
    print(f"\nSe ha guardado esta tabla en: {OUTPUT_DIR / 'top_10_licitaciones_mas_baratas.csv'}")
else:
    print("Faltan algunas de las columnas solicitadas para la inspección.")
    
    
# ------------------------------------------------------------
# 5.2 INSPECCIÓN: ¿HAY CONTRATOS "REALES" DE MENOS DE 100€?
# ------------------------------------------------------------
print("\n--- 🕵️ INSPECCIÓN: CONTRATOS COHERENTES DE MENOS DE 100€ ---")

# Queremos ver licitaciones donde el importe sea menor a 100€
# PERO que sean coherentes: es decir, que el Presupuesto y el Valor Estimado
# también sean pequeños (por ejemplo, menores a 500€), descartando los "Acuerdos Marco".
umbral_bajo = 100.0
umbral_coherencia_max = 500.0

contratos_pequenos_reales = df_eda[
    (df_eda[TARGET] < umbral_bajo) & 
    (df_eda[TARGET] > 0) & # Por si hay algún 0 colado
    (df_eda[PRESUPUESTO] <= umbral_coherencia_max) & 
    (df_eda[VALOR_ESTIMADO] <= umbral_coherencia_max)
]

columnas_ver = [TARGET, PRESUPUESTO, VALOR_ESTIMADO, 'tipo_contrato', 'tipo_procedimiento']
columnas_ver = [c for c in columnas_ver if c in df_eda.columns]

if len(contratos_pequenos_reales) > 0:
    print(f"¡Atención! Se han encontrado {len(contratos_pequenos_reales)} contratos menores a 100€ que parecen REALES.")
    
    # Mostramos los 10 primeros para ver qué pinta tienen
    muestra_pequenos = contratos_pequenos_reales[columnas_ver].sort_values(by=TARGET, ascending=False).head(10)
    print("\nEjemplos de estos contratos pequeños:")
    print(muestra_pequenos.to_markdown(index=False, floatfmt=",.2f"))
else:
    print("No se ha encontrado ningún contrato menor a 100€ que tenga presupuesto y valor estimado coherentes.")
    print("Conclusión: Es seguro borrar todos los menores de 100€, ya que todos son anomalías/acuerdos marco.")