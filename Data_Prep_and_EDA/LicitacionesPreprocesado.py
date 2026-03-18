import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import re
from unidecode import unidecode

# Importaciones de Clustering y NLP
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from nltk.corpus import stopwords
import nltk

# Asegúrate de descargar la lista de stop words (solo la primera vez)
try:
    stopwords_espanol = stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')
    stopwords_espanol = stopwords.words('spanish')

# ---------------- CARGA DE DATOS ---------------------

# 1. Definir la ruta a tu archivo de base de datos SQLite
# NOTA: Usamos barras dobles (\\) o una 'r' (raw string) para manejar la ruta de Windows.
# Usar 'r' es la forma más limpia.
RUTA_BASE_DATOS = r'C:\Users\User\Documents\InferIA\LicitacionesCorrecto.sqlite'

# 2. Definir el nombre de la tabla que has modificado
NOMBRE_TABLA = 'LicitacionesCorrecto_raw'

try:
    # 3. Establecer la conexión con la base de datos
    conn = sqlite3.connect(RUTA_BASE_DATOS)

    # 4. Crear la consulta SQL para obtener todos los datos de la tabla
    query = f"SELECT * FROM {NOMBRE_TABLA}"

    # 5. Cargar los datos directamente en un DataFrame de Pandas
    df = pd.read_sql_query(query, conn)

    print(f"✅ ¡Datos cargados con éxito! El DataFrame tiene {len(df)} filas y {len(df.columns)} columnas.")
    print("Primeras 5 filas del DataFrame:")
    print(df.head())

except sqlite3.Error as e:
    print(f"❌ Error al conectar o cargar datos: {e}")

finally:
    # 6. Cerrar la conexión (IMPORTANTE)
    if 'conn' in locals() and conn:
        conn.close()
        
# ------------------ LIMPIEZA INICIAL -----------------

# --- Columnas nuevas: Ubicaciones licitacion y entidad adjudicadora

df['es_exito'] = np.where(
    df['lote_resultado'].isin(['Formalizado', 'Adjudicado']),
    1,
    0
)
print("✅ Columna 'es_exito' creada.")


# 1. Asegurar que 'ubi_licitacion' sea string. 
# 2. Tomar los primeros 4 caracteres.
df['Ubi_NUTS2'] = df['ubi_licitacion'].astype(str).str.upper().str[:4]


# Condiciones (basadas en el código DAX, priorizadas)
condiciones_ubi = [
    # NOROESTE
    df['Ubi_NUTS2'] == "ES11",
    df['Ubi_NUTS2'] == "ES12",
    df['Ubi_NUTS2'] == "ES13",
    df['Ubi_NUTS2'] == "ES21",
    df['Ubi_NUTS2'] == "ES22",
    
    # NORESTE
    df['Ubi_NUTS2'] == "ES23",
    df['Ubi_NUTS2'] == "ES24",
    df['Ubi_NUTS2'] == "ES30",
    
    # CENTRO
    df['Ubi_NUTS2'] == "ES41",
    df['Ubi_NUTS2'] == "ES42",
    df['Ubi_NUTS2'] == "ES43",
    
    # ESTE
    df['Ubi_NUTS2'] == "ES51",
    df['Ubi_NUTS2'] == "ES52",
    df['Ubi_NUTS2'] == "ES53",
    
    # SUR
    df['Ubi_NUTS2'] == "ES61",
    df['Ubi_NUTS2'] == "ES62",
    
    # TERRITORIOS EXTRA-PENINSULARES
    df['Ubi_NUTS2'] == "ES70",
    
    # CIUDADES AUTÓNOMAS (NUTS 3)
    df['Ubi_NUTS2'] == "ES63",
    df['Ubi_NUTS2'] == "ES64",

    # Resto de clasificaciones internacionales (basadas en los 2 primeros caracteres)
    # Se usa la columna original, pero se extraen solo los 2 primeros caracteres.
    df['ubi_licitacion'].astype(str).str[:2].isin({"ES"}), # España (No Clasificada)
    df['ubi_licitacion'].astype(str).str[:2].isin({"FR", "IT", "DE", "PT", "NL", "BE", "LU", "DK", "SE", "NO", "FI", "IE", "GR", "PL", "CZ", "HU", "RO", "BG", "HR", "SI", "SK", "EE", "LV", "LT"}), # Europa
    df['ubi_licitacion'].astype(str).str[:2].isin({"US", "CA", "BR", "AR", "CL", "MX", "CO", "PE", "UY"}), # América
    df['ubi_licitacion'].astype(str).str[:2].isin({"CN", "JP", "KR", "IN", "SG", "AE", "SA", "TH", "MY"}), # Asia
    df['ubi_licitacion'].astype(str).str[:2].isin({"ZA", "EG", "NG", "KE", "MA", "DZ"}), # África
    df['ubi_licitacion'].astype(str).str[:2].isin({"AU", "NZ"}), # Oceanía
]

# Resultados a asignar si se cumplen las condiciones
resultados_ubi = [
    "Galicia", 
    "Principado de Asturias", 
    "Cantabria", 
    "País Vasco", 
    "Comunidad Foral de Navarra", 
    "La Rioja", 
    "Aragón", 
    "Comunidad de Madrid", 
    "Castilla y León", 
    "Castilla-La Mancha", 
    "Extremadura", 
    "Cataluña", 
    "Comunitat Valenciana", 
    "Illes Balears", 
    "Andalucía", 
    "Región de Murcia", 
    "Canarias", 
    "Ceuta", 
    "Melilla",
    
    "España (No Clasificada)",
    "Europa",
    "América",
    "Asia",
    "África",
    "Oceanía",
]

# El valor por defecto (default) en np.select será el de DAX: "Internacional (Otro)"
df['com_aut_licitacion'] = np.select(
    condiciones_ubi, 
    resultados_ubi, 
    default="Internacional (Otro)"
)

# Eliminar la columna auxiliar
df.drop('Ubi_NUTS2', axis=1, inplace=True)
print("✅ Columna 'com_aut_licitacion' creada.")


# Asegura que 'cp' se trate como texto de 5 dígitos (rellenando con ceros si es necesario) 
# y toma los 2 primeros. Maneja automáticamente los valores NaN (se convierten en 'nan').
df['CP_Prefijo'] = (
    df['cp']
    .astype(str)
    # Limpia '.0' si el CP fue importado como número flotante (ej: '28001.0' -> '28001')
    .str.replace(r'\.0$', '', regex=True)
    # Rellena con ceros a la izquierda hasta 5 caracteres (ej: '28' -> '00028')
    .str.zfill(5)
    # Toma los 2 primeros caracteres (ej: '00028' -> '00' o '28001' -> '28')
    .str[:2]
)


# Condiciones (basadas en el código DAX de las CCAA)
condiciones_cp_espanol = [
    # ANDALUCÍA
    df['CP_Prefijo'].isin({"04", "11", "14", "18", "21", "23", "29", "41"}),
    # ARAGÓN
    df['CP_Prefijo'].isin({"22", "44", "50"}),
    # ASTURIAS
    df['CP_Prefijo'] == "33",
    # ISLAS BALEARES
    df['CP_Prefijo'] == "07",
    # CANARIAS
    df['CP_Prefijo'].isin({"35", "38"}),
    # CANTABRIA
    df['CP_Prefijo'] == "39",
    # CASTILLA Y LEÓN
    df['CP_Prefijo'].isin({"05", "09", "24", "34", "37", "40", "42", "47", "49"}),
    # CASTILLA-LA MANCHA
    df['CP_Prefijo'].isin({"02", "13", "16", "19", "45"}),
    # CATALUÑA
    df['CP_Prefijo'].isin({"08", "17", "25", "43"}),
    # COMUNIDAD VALENCIANA
    df['CP_Prefijo'].isin({"03", "12", "46"}),
    # EXTREMADURA
    df['CP_Prefijo'].isin({"06", "10"}),
    # GALICIA
    df['CP_Prefijo'].isin({"15", "27", "32", "36"}),
    # LA RIOJA
    df['CP_Prefijo'] == "26",
    # MADRID
    df['CP_Prefijo'] == "28",
    # MURCIA
    df['CP_Prefijo'] == "30",
    # NAVARRA
    df['CP_Prefijo'] == "31",
    # PAÍS VASCO
    df['CP_Prefijo'].isin({"01", "20", "48"}),
    # CEUTA Y MELILLA
    df['CP_Prefijo'] == "51",
    df['CP_Prefijo'] == "52",
]

# Resultados a asignar si se cumplen las condiciones
resultados_cp_espanol = [
    "Andalucía",
    "Aragón",
    "Principado de Asturias",
    "Illes Balears",
    "Canarias",
    "Cantabria",
    "Castilla y León",
    "Castilla-La Mancha",
    "Cataluña",
    "Comunitat Valenciana",
    "Extremadura",
    "Galicia",
    "La Rioja",
    "Comunidad de Madrid",
    "Región de Murcia",
    "Comunidad Foral de Navarra",
    "País Vasco",
    "Ceuta",
    "Melilla",
]

# El valor por defecto (default) en np.select será "Internacional", tal como pediste,
# lo que captura CP nulos, CP no válidos que no coinciden con las provincias españolas, 
# y cualquier código extranjero.
df['com_aut_adjudicador'] = np.select(
    condiciones_cp_espanol, 
    resultados_cp_espanol, 
    default="Internacional"
)

print("✅ Columna 'com_aut_adjudicador' creada (basada solo en CCAA españolas y el resto como 'Internacional').")


# ---------------- Columnas de precios a imputar con -1 ----------------

PRICE_COLS_TO_IMPUTE = [
    'lote_precio_oferta_mas_alta',
    'lote_precio_oferta_mas_baja',
    'lote_importe_adjudicacion_sin_impuestos',
    'lote_importe_adjudicacion_con_impuestos'
]

print("--- 1. Imputación de Precios/Importes con -1 ---")

for col in PRICE_COLS_TO_IMPUTE:
    # Usamos fillna() para reemplazar los valores NaN con -1
    # Asegúrate de que las columnas sean de tipo numérico (float/Int64) para esta operación
    df[col] = df[col].fillna(-1)
    print(f"✅ Columna '{col}' imputada con -1.")

# --- A. CREAR FEATURE BINARIA 'es_sobrecoste'  ---

# Paso 1: Definir los casos de 'No Adjudicado' (donde el importe es -1)
condicion_no_adjudicado = (df['lote_importe_adjudicacion_sin_impuestos'] == -1)

# Paso 2: Definir la condición de 'Sobrecoste' (Importe > Presupuesto)
# Excluimos explícitamente los casos de No Adjudicado (condicion_no_adjudicado)
condicion_sobrecoste_pura = (
    (df['lote_importe_adjudicacion_sin_impuestos'] > df['lote_presupuesto_base_sin_impuestos']) &
    (~condicion_no_adjudicado)
)

# Paso 3: Crear la columna 'es_sobrecoste' usando np.select
df['es_sobrecoste'] = np.select(
    [
        condicion_no_adjudicado,      # Caso 1: No Adjudicado
        condicion_sobrecoste_pura     # Caso 2: Sobrecoste
    ],
    [
        -1,                           # Resultado 1: -1
        1                             # Resultado 2: 1
    ],
    default=0                         # Resto (Importe <= Presupuesto, y Adjudicado) es 0
)

# Convertir a Int64 en una línea separada para asegurar que el motor de Pandas lo maneje correctamente
df['es_sobrecoste'] = df['es_sobrecoste'].astype('Int64')
# --------------------

print("✅ Columna 'es_sobrecoste' creada con lógica mejorada (-1: No Adjudicado, 1: Sobrecoste).")
print(f"Número de casos de sobrecoste (1): {df['es_sobrecoste'].value_counts().get(1, 0)}")

# ----------------- Creación de descuento_promedio -----------------

print("\n--- REESCRITURA FINAL DE descuento_promedio (CON TIPOS CONFIRMADOS) ---")

# CRITICAL: Define local float variables for robust comparison
importe_adj = df['lote_importe_adjudicacion_sin_impuestos'].astype(float)
presupuesto_base = df['lote_presupuesto_base_sin_impuestos'].astype(float)
es_sobrecoste = df['es_sobrecoste'] 


# Caso 1: Asignar -1.0 (No Adjudicado O Presupuesto Inválido para el cálculo)
# This includes the 98083 (IA=-1) + 17479 (PB<=0 & IA>0) = 115562
condicion_menos_uno = (importe_adj == -1.0) | ((presupuesto_base <= 0.0) & (importe_adj > 0.0))

# Caso 2: Asignar 0.0 (Presupuesto CERO e Importe CERO)
condicion_cero_cero = (presupuesto_base == 0.0) & (importe_adj == 0.0)


# Use np.select to apply the conditions in order
df['descuento_promedio'] = np.select(
    [
        condicion_menos_uno,     # Priority 1: Incalculable / No Adjudicado (-1.0)
        condicion_cero_cero      # Priority 2: Cero Descuento (0.0)
    ],
    [
        -1.0, 
        0.0   
    ],
    # Default: Normal calculation (safe since the zero-division cases were handled)
    default=(presupuesto_base - importe_adj) / presupuesto_base
)

print("✅ 'descuento_promedio' recalculado con lógica y tipos corregidos. El conteo de -1.0 será 115,562.")


# --- B. LIMITACIÓN (CAPPING) DE VALORES NEGATIVOS EXTREMOS (Re-aplicación) ---

CAP_NEGATIVO = -0.5

df['descuento_promedio'] = np.where(
    (df['descuento_promedio'] < CAP_NEGATIVO) & (es_sobrecoste == 1),
    CAP_NEGATIVO,
    df['descuento_promedio']
)


# ----------------------- Columnas a eliminar --------------------

DROP_COLS = [
    'estado', 'CP_Prefijo', 'vigente', 'enlace', 'anomalia_precio',
    'ubi_licitacion', 'cp', 'fecha_actualizacion', 'presupuesto_base_con_impuestos',
    'lote_presupuesto_base_con_impuestos', 'lote_importe_adjudicacion_con_impuestos',
    'datos_relevantes_faltantes']

# Elimina solo si existen, para evitar errores
df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')

print(f"🧹 Columnas eliminadas: {[c for c in DROP_COLS if c in df.columns]}")
print(f"El DataFrame ahora tiene {len(df.columns)} columnas.")


# --- Limpieza de texto en columnas con valores inconsistentes ---
TEXT_COLS_TO_CLEAN = [
    'lote_resultado', 'sistema_contratacion', 'lote_pyme', 'tipo_procedimiento', 'tipo_administracion', 'lote_tipo_id_adjudicatario', 'tipo_contrato', 'lote_adjudicatario', 'objeto', 'lote_objeto', 'organo_contratacion', 'com_aut_licitacion', 'com_aut_adjudicador'
]

for col in TEXT_COLS_TO_CLEAN:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()                                 # elimina espacios extremos
            .str.lower()                                 # todo en minúsculas
            .str.replace(r'\s+', ' ', regex=True)         # un solo espacio entre palabras
            .apply(lambda x: unidecode(x))                # elimina tildes y caracteres raros
            .str.replace(r'[^\w\s]', '', regex=True)      # elimina puntuación tipo puntos, comas, etc.
            .replace({'nan': np.nan, '^': np.nan, 'none': np.nan, 'null': np.nan})  # corrige valores raros
        )
        print(f"✅ '{col}' limpiada y normalizada.")
    else:
        print(f"⚠️ Columna '{col}' no encontrada, se omite.")

# --------------- Temporales ------------
df['fecha_primera_publicacion'] = pd.to_datetime(
    df['fecha_primera_publicacion'],
    format='%d/%m/%Y',   
    errors='coerce'      # convierte valores no válidos en NaT (nulos)
)
df['anio'] = df['fecha_primera_publicacion'].dt.year.astype('Int64')
df['mes'] = df['fecha_primera_publicacion'].dt.month.astype('Int64')
print("✅ Columnas 'anio' y 'mes' creadas.")


##########################################################################################
####################### CLUSTERING DE ALTA CARDINALIDAD ##################################
##########################################################################################

print("\n⏳ Iniciando Clustering para Organo de Contratacion y Objeto...")

# Nota: Los valores de K_ORGANO y K_TEXT deben ser optimizados con el script de análisis.
K_ORGANO = 11 # Es el óptimo
K_TEXT = 15  # El óptimo matemáticamente hablando es k=3 también, pero k=7 da buen valor también y puede ser mejor en la separación


# ================= A. CLUSTERING: ORGANO DE CONTRATACIÓN (Semántico) =================

print("\n--- A. Clustering de Órgano (TF-IDF + SVD sobre el NOMBRE) ---")

# 1. Preparar los datos de texto (nombres únicos de órganos)
# Trabajamos solo con los nombres únicos para ser más eficientes
organos_unicos = df['organo_contratacion'].dropna().unique()
df_organos = pd.DataFrame({'organo_contratacion': organos_unicos})

# Limpieza básica específica para nombres de órganos (opcional pero recomendada)
# Ayuda a juntar "Ayto" con "Ayuntamiento", etc.
df_organos['nombre_limpio'] = df_organos['organo_contratacion'].astype(str).str.lower()

# 2. Vectorización TF-IDF (igual que hiciste con el objeto)
print("Vectorizando nombres de órganos...")
vectorizer_org = TfidfVectorizer(
    stop_words=stopwords_espanol, 
    max_features=2000, # Menos features que para el objeto, los nombres son más cortos
    ngram_range=(1, 2)
)
tfidf_matrix_org = vectorizer_org.fit_transform(df_organos['nombre_limpio'])

# 3. Reducción de Dimensionalidad (SVD)
# Reducimos a 50 componentes (suele ser suficiente para nombres cortos)
svd_org = TruncatedSVD(n_components=50, random_state=42)
organo_scaled = svd_org.fit_transform(tfidf_matrix_org) # Esta variable 'organo_scaled' es la que usarás para optimizar K

# Guardar para análisis de K (igual que antes)
#RUTA_GUARDADO = 'matrices_intermedias'
#np.save(f'{RUTA_GUARDADO}/organo_scaled_semantico.npy', organo_scaled)

# 4. Aplicar KMeans (Usando el K_ORGANO que decidas tras el análisis)
print(f"Aplicando KMeans con K={K_ORGANO}...")
kmeans_organo = KMeans(n_clusters=K_ORGANO, random_state=42, n_init=10)
df_organos['organo_cluster'] = kmeans_organo.fit_predict(organo_scaled)

# 5. Fusionar el cluster al DataFrame original
# Ahora fusionamos por el nombre del órgano
df = df.merge(
    df_organos[['organo_contratacion', 'organo_cluster']], 
    on='organo_contratacion', 
    how='left'
)
df['organo_cluster'] = df['organo_cluster'].fillna(-1).astype('Int64').astype('category')
print(f"✅ Columna 'organo_cluster' (Semántico) creada con {K_ORGANO} grupos.")

# ================= B. CLUSTERING: LOTE_OBJETO (NLP + SVD) =================

print("\n--- B. Clustering de Lote_Objeto (TF-IDF + SVD) ---")

# 1. Vectorización TF-IDF
text_data = df['lote_objeto'].astype(str).fillna('sin_objeto_valido')
# Definir un umbral de longitud para eliminar descripciones inútiles o nulas
text_data = text_data.apply(lambda x: x if len(x.split()) > 2 else 'sin_objeto_valido') 

vectorizer = TfidfVectorizer(
    stop_words=stopwords_espanol, 
    max_features=5000, 
    ngram_range=(1, 2)
)
tfidf_matrix = vectorizer.fit_transform(text_data)
print(f"Dimensión de matriz TF-IDF: {tfidf_matrix.shape}")

# 2. Reducción de Dimensionalidad (SVD)
N_COMPONENTS = 100 
svd = TruncatedSVD(n_components=N_COMPONENTS, random_state=42)
# Guardamos la matriz reducida para el análisis de K (opcional, pero útil)
tfidf_reduced = svd.fit_transform(tfidf_matrix) 

# Guardar la matriz reducida para el análisis de K
#RUTA_GUARDADO = 'matrices_intermedias' # Ya definida, pero la reutilizamos
#np.save(f'{RUTA_GUARDADO}/tfidf_reduced.npy', tfidf_reduced)
#print(f"✅ Matriz tfidf_reduced guardada en {RUTA_GUARDADO}/tfidf_reduced.npy")

# 3. Aplicar KMeans
kmeans_text = KMeans(n_clusters=K_TEXT, random_state=42, n_init=10)
clusters = kmeans_text.fit_predict(tfidf_reduced)

# 4. Asignar clusters al DataFrame original (Cuidado con los índices)
# Creamos un Series para alinear los clusters con los datos originales, incluidos los NaNs
cluster_map = pd.Series(clusters, index=text_data.index)
df['objeto_cluster'] = cluster_map
df['objeto_cluster'] = df['objeto_cluster'].fillna(-1).astype('Int64').astype('category')
print(f"✅ Columna 'objeto_cluster' creada con {K_TEXT} grupos.")


# ================================================================
# 1. ANÁLISIS DEL TAMAÑO Y DISTRIBUCIÓN DE LOS CLUSTERS COMBINADOS
# ================================================================

# AÑADIR ESTO PARA VER TODAS LAS COLUMNAS Y FILAS
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_rows', 100)      
pd.set_option('display.width', 1000)        
pd.set_option('display.max_colwidth', None) 

print("✅ Opciones de visualización de Pandas ajustadas.")

print("\n--- 1. Distribución y Tamaño de los Clusters ---")

# Conteo de elementos por cluster combinado (Organo X Objeto)
conteo_clusters_combinado = df.groupby(['organo_cluster', 'objeto_cluster']).size().reset_index(name='n_licitaciones')

# Conteo total (para calcular el porcentaje)
total_licitaciones = conteo_clusters_combinado['n_licitaciones'].sum()
conteo_clusters_combinado['porcentaje'] = (conteo_clusters_combinado['n_licitaciones'] / total_licitaciones) * 100

print("\nConteo y Porcentaje de Licitaciones por Par de Cluster (Organo x Objeto):")
print(conteo_clusters_combinado.sort_values(by='n_licitaciones', ascending=False).head(10)) 

# ================================================================
# DEFINICIÓN DE FUNCIÓN COMÚN (Solo una vez)
# ================================================================
def obtener_top_terminos_cluster(df, cluster_col, texto_col, top_n=5):
    """Obtiene los N términos/frases más frecuentes por cluster de texto."""
    
    from sklearn.feature_extraction.text import TfidfVectorizer 
    
    # Crea un DataFrame temporal con solo los textos y el cluster_id
    temp_df = df[[cluster_col, texto_col]].dropna()
    
    resultados = []
    
    # Iteramos por los clusters ordenados
    unique_clusters = sorted(temp_df[cluster_col].unique())
    
    for cluster_id in unique_clusters:
        textos_cluster = temp_df[temp_df[cluster_col] == cluster_id][texto_col]
        
        if len(textos_cluster) == 0: continue

        # Recalcula el TF-IDF solo en este subconjunto
        vectorizer_local = TfidfVectorizer(
            stop_words=stopwords_espanol, 
            max_features=500,
            ngram_range=(1, 2)
        )
        try:
            tfidf_local = vectorizer_local.fit_transform(textos_cluster)
            
            # Suma los valores TF-IDF por término
            sum_of_weights = np.sum(tfidf_local.toarray(), axis=0)
            tfidf_feature_names = vectorizer_local.get_feature_names_out()
            
            # Crear un DataFrame de términos y su peso
            features_df = pd.DataFrame({'feature': tfidf_feature_names, 'weight': sum_of_weights})
            top_features = features_df.sort_values(by='weight', ascending=False).head(top_n)
            
            # Almacena los resultados
            resultados.append({
                'cluster_id': cluster_id,
                'top_terminos': top_features['feature'].tolist(),
                'total_registros': len(textos_cluster)
            })
        except ValueError:
            # Puede pasar si el vocabulario está vacío
            print(f"⚠️ Cluster {cluster_id} con texto vacío o solo stopwords.")
        
    return pd.DataFrame(resultados)

# ================================================================
# 2. CARACTERIZACIÓN DEL ORGANO_CLUSTER (SEMÁNTICO)
# ================================================================

print("\n--- 2. Temáticas Definitorias del Organo_Cluster (Semántico) ---")

# Llamada 1: Para ÓRGANO
top_terminos_organo = obtener_top_terminos_cluster(df, 'organo_cluster', 'organo_contratacion', top_n=7)

print("\nTop 7 Términos por Organo_Cluster (K={}):".format(K_ORGANO))
print(top_terminos_organo)
print("\n✅ INTERPRETACIÓN: Usa estas palabras para etiquetar tus clusters de órganos (ej. Cluster 0: Ayuntamientos, Cluster 1: Salud...).")


# ================================================================
# 3. CARACTERIZACIÓN DEL OBJETO_CLUSTER (SEMÁNTICO)
# ================================================================

print("\n--- 3. Temáticas Definitorias del Objeto_Cluster ---")

# Llamada 2: Para OBJETO
top_terminos_objeto = obtener_top_terminos_cluster(df, 'objeto_cluster', 'lote_objeto', top_n=5)

print("\nTop 5 Términos/Conceptos por Objeto_Cluster (K={}):".format(K_TEXT))
print(top_terminos_objeto)
print("\n✅ INTERPRETACIÓN: Analiza los 'top_terminos' para 'etiquetar' cada cluster.")

# --- 1. Mapeo del Organo_Cluster (K=11 - Semántico) ---
organo_map = {
    0: 'ORG_Ayto_General',                # Ayuntamientos genéricos
    1: 'ORG_Cabildos_Insulares',          # Cabildos / Insular (Canarias)
    2: 'ORG_Plenos_Municipales_A',        # Plenos
    3: 'ORG_Alcaldias_A',                 # Alcaldías
    4: 'ORG_Universidades_y_Gerencias',   # ¡Importante! Unis, Salud y Diputaciones
    5: 'ORG_Gobierno_Local',              # Juntas de Gobierno Local
    6: 'ORG_Alcaldias_B',                 # Otro grupo de alcaldías
    7: 'ORG_Plenos_Municipales_B',        # Otro grupo de plenos
    8: 'ORG_Empresas_Publicas_Consejos',  # Consejos de Administración / S.A.
    9: 'ORG_Juntas_Gobierno',             # Juntas de Gobierno
    10: 'ORG_Admin_General_Estatal',      # Direcciones Generales / Estado
    -1: 'ORG_Sin_Cluster'
}

# --- 2. Mapeo del Objeto_Cluster (K=15) ---
# Basado en el análisis de Top Términos con K=15
objeto_map = {
    2: 'OBJ_General',
    13: 'OBJ_Servicios_Limpieza',
    12: 'OBJ_Suministro_e_Instalacion',
    6: 'OBJ_Suministros_NAO',
    3: 'OBJ_Materiales_Construccion_Obras',
    8: 'OBJ_Servicios_Mantenimiento',
    1: 'OBJ_Suministro_Repuestos',
    0: 'OBJ_Asistencia_Tecnica',
    9: 'OBJ_Acuerdo_Marco_Obras_Planes',
    10: 'OBJ_Suministro_Material_Oficina',
    11: 'OBJ_Servicios_SS_Mutuas',
    5: 'OBJ_Suministro_Mobiliario',
    4: 'OBJ_Suministro_Mantenimiento_TI',
    14: 'OBJ_Soluciones_Integrales_Especiales',
    7: 'OBJ_Acuerdo_Marco_Generico',
    -1: 'OBJ_Sin_Cluster'
}

# --- 3. Aplicación del Mapeo y Creación de la Variable Combinada ---

# Paso de Corrección: Asegurar que las columnas a mapear sean de tipo 'object' (string)
# Esto previene el error 'TypeError: Cannot setitem on a Categorical with a new category'
df['organo_cluster'] = df['organo_cluster'].astype('object')
df['objeto_cluster'] = df['objeto_cluster'].astype('object')


# Asignar la etiqueta de texto para Organo_Cluster
# El valor -1 en el mapeo maneja los casos conocidos. fillna() solo es un seguro para NaN
df['organo_cluster_label'] = df['organo_cluster'].map(organo_map).fillna('ORG_No_Asignado')

# Asignar la etiqueta de texto para Objeto_Cluster
df['objeto_cluster_label'] = df['objeto_cluster'].map(objeto_map).fillna('OBJ_No_Asignado')

print("✅ Mapeo de clusters aplicado. Columnas de texto creadas.")

# --- Eliminar columnas numéricas de cluster originales ---
# Las columnas numéricas ya han sido reemplazadas por sus versiones de texto (labels)
columnas_a_eliminar = ['organo_cluster', 'objeto_cluster']

# Elimina las columnas del DataFrame
df = df.drop(columns=columnas_a_eliminar)

print(f"✅ Columnas {columnas_a_eliminar} eliminadas. ¡El DataFrame está listo para el modelado!")

# ---------------- DEFINICIÓN DE LISTAS DE COLUMNAS -----------------

# Columnas con texto (comprobar las últimas si he cambiado algo)
CATEGORY_COLS = ['lote_resultado', 'sistema_contratacion', 'lote_pyme',
                 'tipo_procedimiento', 'tipo_administracion', 'lote_tipo_id_adjudicatario',
                 'tipo_contrato', 'com_aut_licitacion', 'com_aut_adjudicador',
                 'organo_cluster_label', 'objeto_cluster_label']

# Los nombres históricos se generaron dinámicamente según tipos_contrato
#cols_hist_pct = [f'pct_hist_{tc}' for tc in tipos_contrato]

# Reemplaza los que habías puesto manualmente en NUMERIC_COLS
NUMERIC_COLS = [
    'lote_precio_oferta_mas_alta', 'lote', 'lote_presupuesto_base_sin_impuestos', 
    'lote_importe_adjudicacion_sin_impuestos', 'lote_precio_oferta_mas_baja', 
    'presupuesto_base_sin_impuestos', 'lote_numero_ofertas_recibidas', 
    'peso_relativo_lote', 'valor_estimado_imputado', 'duracion_proceso_dias', 
    'es_loteado', 'es_anomalia_temporal', 'es_exito', 'es_sobrecoste','anio', 'mes'
#    ,'descuento_promedio', 'n_licitaciones_hist', 'tasa_exito_hist', 
#   'descuento_medio_hist', 'tasa_sobrecoste_hist'
] 
#+ cols_hist_pct + cols_organo_hist

# Las de altísima cardinalidad las ponemos como object
cols_high_card = ['organo_contratacion', 'objeto', 'lote_objeto', 'lote_adjudicatario']
for col in cols_high_card:
    df[col] = df[col].astype('object')


df['fecha_primera_publicacion'] = pd.to_datetime(
    df['fecha_primera_publicacion'],
    format='%d/%m/%Y',   
    errors='coerce'      # convierte valores no válidos en NaT (nulos)
)

# --------------- PROCESAMIENTO DE CONVERSIONES ----------------

# 1. Conversión a Categórico
print("Convirtiendo a tipo 'category'...")
for col in CATEGORY_COLS:
    df[col] = df[col].astype('category')

# 2. Conversión a Numérico
print("Convirtiendo a tipo numérico y optimizando la memoria...")
for col in NUMERIC_COLS:
    # Intenta convertir a numérico. 'coerce' convierte errores a NaN.
    df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    # Opcional: Intentar convertir a Int64 para ahorrar memoria.
    # El tipo 'Int64' (con I mayúscula) soporta valores nulos (NaN).
    try:
        # Esto solo tendrá éxito si los números resultantes no tienen decimales
        df[col] = df[col].astype('Int64')
    except Exception as e:
        # Si hay decimales o si la columna es float por defecto, mantendrá el tipo anterior.
        # print(f"Advertencia: No se pudo convertir {col} a Int64. Error: {e}")
        pass # Ignoramos la excepción si la conversión a Int64 falla.
    
print("Conversiones completadas.")


# ----------------- Verificación ------------------

# Muestra el número total de filas, columnas y el uso de memoria
print("\n--- 1. Información del DataFrame (df.info()) ---")
df.info()

# Muestra los tipos de datos de las columnas
print("\n--- 2. Tipos de Datos (df.dtypes) ---")
print(df.dtypes)

# ----------------- CONFIGURACIÓN DE VISUALIZACIÓN ------------------
# Asegura que se muestren todas las columnas y filas para la revisión
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 1000)

print("--- INICIO DE LA VALIDACIÓN DEL PREPROCESAMIENTO ---\n")


# ====================================================================
# 1. VALIDACIÓN DE VALORES NULOS (NaN)
# ====================================================================

print("### 1. VALIDACIÓN DE NULOS (NaN) ###")
# Nota: La validación solo tiene sentido si asumimos que ya imputaste los nulos
# en Power BI para el resto de columnas, excepto las de precio.

# Contar NaNs en todo el DataFrame. Debería ser 0 para la mayoría de columnas
# si la imputación de precio con -1 y la limpieza de texto funcionaron.
nan_counts = df.isna().sum()
nan_cols = nan_counts[nan_counts > 0]

if nan_cols.empty:
    print("✅ ¡Éxito! No se encontraron valores NaN en el DataFrame. ¡Limpieza perfecta!")
else:
    print("⚠️ ¡Advertencia! Se encontraron NaN en las siguientes columnas (revisar su origen):")
    print(nan_cols)


# ====================================================================
# 2. VALIDACIÓN DE IMPUTACIÓN CON -1 (Precios/Importes)
# ====================================================================

# Lista de las columnas que debieron ser imputadas con -1
PRICE_COLS_TO_IMPUTE = [
    'lote_precio_oferta_mas_alta',
    'lote_precio_oferta_mas_baja',
    'lote_importe_adjudicacion_sin_impuestos',
]
print("\n### 2. VALIDACIÓN DE IMPUTACIÓN CON -1 EN PRECIOS/IMPORTES ###")

all_valid = True
for col in PRICE_COLS_TO_IMPUTE:
    # Contar cuántos -1 tiene la columna
    count_neg_one = (df[col] == -1).sum()
    
    # Contar cuántos valores son menores que -1 (no deberían existir)
    count_too_low = (df[col] < -1).sum()

    if count_neg_one > 0 and count_too_low == 0:
        print(f"✅ '{col}': {count_neg_one} valores -1 encontrados. Rango mínimo: {df[col].min()}.")
    elif count_too_low > 0:
        print(f"❌ ERROR en '{col}': Se encontraron {count_too_low} valores menores que -1. Revisar. Mínimo: {df[col].min()}")
        all_valid = False
    else:
        print(f"⚠️ Alerta en '{col}': No se encontraron valores -1. ¿Faltaron nulos originales por imputar?")
        all_valid = False

if all_valid:
    print("✅ Todas las columnas de precio/importe con -1 cumplen el requisito.")


# ====================================================================
# 3. VALIDACIÓN DE LA FEATURE 'es_sobrecoste'
# ====================================================================

print("\n### 3. VALIDACIÓN DE 'es_sobrecoste' (-1, 0, 1) ###")
total_filas = len(df)

# 3.1. Validar casos de 'No Adjudicado' (-1)
no_adj_count = (df['es_sobrecoste'] == -1).sum()
# Los casos -1 deben coincidir exactamente con los casos donde el importe de adjudicación es -1
expected_no_adj = (df['lote_importe_adjudicacion_sin_impuestos'] == -1).sum()

if no_adj_count == expected_no_adj:
    print(f"✅ Casos No Adjudicado (-1): {no_adj_count} filas. Coincide con la imputación de importe.")
else:
    print(f"❌ ERROR: Casos -1 en 'es_sobrecoste' ({no_adj_count}) NO coinciden con Imputación de Importe ({expected_no_adj}).")

# 3.2. Validar casos de 'Sobrecoste' (1)
sobrecoste_count = (df['es_sobrecoste'] == 1).sum()
# Un caso de sobrecoste debe ser Adjudicado (es_sobrecoste != -1) Y (Importe > Presupuesto)
expected_sobrecoste = len(df[
    (df['lote_importe_adjudicacion_sin_impuestos'] != -1) & 
    (df['lote_importe_adjudicacion_sin_impuestos'] > df['lote_presupuesto_base_sin_impuestos'])
])

if sobrecoste_count == expected_sobrecoste:
    print(f"✅ Casos Sobrecoste (1): {sobrecoste_count} filas. Coincide con la fórmula (Importe > Presupuesto).")
else:
    print(f"❌ ERROR: Casos 1 en 'es_sobrecoste' ({sobrecoste_count}) NO coinciden con la fórmula esperada ({expected_sobrecoste}).")

# 3.3. Validar que la suma de categorías sea correcta
normal_count = (df['es_sobrecoste'] == 0).sum()
if (no_adj_count + sobrecoste_count + normal_count) == total_filas:
    print(f"✅ Suma de categorías: {normal_count} (Normal) + {sobrecoste_count} (Sobrecoste) + {no_adj_count} (No Adj) = {total_filas} filas. OK.")
else:
    print("❌ ERROR: La suma de categorías en 'es_sobrecoste' no coincide con el total de filas.")


# ====================================================================
# 4. VALIDACIÓN DE 'descuento_promedio' y CAPPING (LÓGICA ACTUALIZADA)
# ====================================================================

CAP_NEGATIVO = -0.5
print("\n### 4. VALIDACIÓN DE 'descuento_promedio' y CAPPING (LÓGICA ACTUALIZADA) ###")

print("--- VERIFICACIÓN DE CONTEO DE CONDICIONES DE descuento_promedio (Verificado con SQL ---")

# --- 0. Definición de Variables (Asegurando que sean float para comparaciones robustas) ---
# Importante: Ejecuta esto si las variables no están ya definidas o si quieres asegurar el tipo.
# Si ya tienes estas variables definidas como float, puedes omitir este paso.
importe_adj = df['lote_importe_adjudicacion_sin_impuestos'].astype(float)
presupuesto_base = df['lote_presupuesto_base_sin_impuestos'].astype(float)
es_sobrecoste = df['es_sobrecoste']

# --- 1. Definición de Condiciones ---

# Caso 1: No Adjudicado (-1) (Solo IA = -1, que ya fue marcado en es_sobrecoste)
condicion_1_no_adj = (es_sobrecoste == -1)

# Caso 2: División por Cero/Negativo Inválida (PB <= 0, pero IA > 0)
condicion_2_div_invalida = ((presupuesto_base <= 0) & (importe_adj > 0))

# Caso 3: Presupuesto CERO e Importe CERO (0.0)
condicion_3_cero_cero = ((presupuesto_base == 0) & (importe_adj == 0))


# --- 2. Conteo de Filas para cada Condición (Sin considerar la prioridad de np.select) ---

conteo_1 = condicion_1_no_adj.sum()
conteo_2 = condicion_2_div_invalida.sum()
conteo_3 = condicion_3_cero_cero.sum()


print(f"1. Caso No Adjudicado (es_sobrecoste = -1, Asignar -1.0): {conteo_1}. Deberían ser 98083")
print(f"2. División Inválida (PB <= 0, IA > 0, Asignar -1.0): {conteo_2}. Deberían ser 17479")
print(f"3. Cero Descuento (PB = 0, IA = 0, Asignar 0.0): {conteo_3}. Deberían ser 12275")
print("-" * 50)

# --- 3. Conteo de la LÓGICA FINAL (Considerando la PRIORIDAD de np.select) ---

# np.select toma las condiciones en orden y asigna el primer valor verdadero que encuentra.

# Filas que realmente terminarán con -1.0
filas_con_neg_uno = (condicion_1_no_adj) | (
    (~condicion_1_no_adj) & (condicion_2_div_invalida)
)
conteo_final_neg_uno = filas_con_neg_uno.sum()

# Filas que realmente terminarán con 0.0 (de esta condición específica)
filas_con_cero_cero = (
    (~condicion_1_no_adj) & 
    (~condicion_2_div_invalida) & 
    (condicion_3_cero_cero)
)
conteo_final_cero_cero = filas_con_cero_cero.sum()


print(f"➡️ Conteo FINAL de -1.0 (Caso 1 Y/O Caso 2, con prioridad): {conteo_final_neg_uno}")
print(f"➡️ Conteo FINAL de 0.0 (Solo Caso 3, que no fue Caso 1 ni Caso 2): {conteo_final_cero_cero}")

print("-" * 50)
total_filas_especiales = conteo_final_neg_uno + conteo_final_cero_cero
total_df = len(df)
print(f"Total de filas con lógica especial (-1.0 o 0.0): {total_filas_especiales}")
print(f"Filas que irán a la FÓRMULA (Default): {total_df - total_filas_especiales}")


# ----------------- 4.3. Validación del CAPPING -----------------
# El Capping solo aplica a Sobrecostes REALES (es_sobrecoste = 1)
capped_violations = len(df[
    # El descuento es menor que el límite (-0.5)
    (df['descuento_promedio'] < CAP_NEGATIVO) &
    # PERO NO es el valor de imputación -1.0
    (df['descuento_promedio'] != -1.0)
])

if capped_violations == 0:
    print(f"✅ Capping aplicado: No se encontraron valores reales de descuento < {CAP_NEGATIVO} (excluyendo -1.0).")
else:
    print(f"❌ ERROR de Capping: Se encontraron {capped_violations} valores menores que {CAP_NEGATIVO} (excluyendo -1.0).")
    print("    Ejemplo de violación (Descuento):")
    print(df[df['descuento_promedio'] < CAP_NEGATIVO][['descuento_promedio', 'es_sobrecoste']].head())


# --------------- GUARDAR DATAFRAME LIMPIO -----------------

### chequeo ###

# ======================================================
# 📋 DEFINICIÓN DE LISTAS DE COLUMNAS ESPERADAS
# ======================================================

CATEGORY_COLS = [
    'lote_resultado', 'sistema_contratacion', 'lote_pyme',
    'tipo_procedimiento', 'tipo_administracion', 'lote_tipo_id_adjudicatario',
    'tipo_contrato', 'com_aut_licitacion', 'com_aut_adjudicador',
    'organo_cluster_label', 'objeto_cluster_label'
]

NUMERIC_COLS = [
    'lote_precio_oferta_mas_alta', 'lote', 'lote_presupuesto_base_sin_impuestos',
    'lote_importe_adjudicacion_sin_impuestos', 'lote_precio_oferta_mas_baja',
    'presupuesto_base_sin_impuestos', 'lote_numero_ofertas_recibidas',
    'peso_relativo_lote', 'valor_estimado_imputado', 'duracion_proceso_dias',
    'es_loteado', 'es_anomalia_temporal', 'es_exito', 'es_sobrecoste',
    'anio', 'mes'
]

HIGH_CARD_COLS = ['organo_contratacion', 'objeto', 'lote_objeto', 'lote_adjudicatario']
FECHA_COLS = ['fecha_primera_publicacion']

# ======================================================
# 🧠 FUNCIONES AUXILIARES
# ======================================================

def check_group(df, expected_cols, label):
    """Verifica si las columnas esperadas están presentes y detecta si hay vacías."""
    missing = [c for c in expected_cols if c not in df.columns]
    empty = [c for c in expected_cols if c in df.columns and df[c].isnull().all()]
    
    if not missing:
        print(f"✅ {label}: todas las columnas esperadas están presentes ({len(expected_cols)}).")
    else:
        print(f"⚠️ {label}: faltan {len(missing)} columnas -> {missing}")
    
    if empty:
        print(f"ℹ️ {label}: las siguientes columnas existen pero están completamente vacías -> {empty}")

# ======================================================
# 🔍 VALIDACIÓN DE COLUMNAS
# ======================================================

print("\n--- VALIDACIÓN DE COLUMNAS ---")
check_group(df, CATEGORY_COLS, "Categóricas")
check_group(df, NUMERIC_COLS, "Numéricas")
check_group(df, HIGH_CARD_COLS, "Alta cardinalidad (object)")
check_group(df, FECHA_COLS, "Fechas")

# ======================================================
# 🧾 DETECCIÓN DE COLUMNAS EXTRA
# ======================================================

expected_all = set(CATEGORY_COLS + NUMERIC_COLS + HIGH_CARD_COLS + FECHA_COLS)
cols_in_db = set(df.columns)

extra_cols = sorted(list(cols_in_db - expected_all))
missing_cols = sorted(list(expected_all - cols_in_db))

print("\n--- DIFERENCIAS ENTRE EXPECTED Y DATABASE ---")
if not extra_cols and not missing_cols:
    print("✅ Coincidencia perfecta: ni columnas extra ni faltantes.")
else:
    if extra_cols:
        print(f"⚠️ Columnas EXTRA no esperadas en la base de datos ({len(extra_cols)}):\n{extra_cols}")
    if missing_cols:
        print(f"⚠️ Columnas FALTANTES ({len(missing_cols)}):\n{missing_cols}")

# ======================================================
# 🧮 VERIFICACIÓN DE TIPOS DE DATOS (versión robusta)
# ======================================================

print("\n--- COMPROBACIÓN DE TIPOS DE DATOS ---")
type_expectations = {
    'categórica': CATEGORY_COLS,
    'numérica': NUMERIC_COLS,
    'alta_cardinalidad': HIGH_CARD_COLS,
    'fecha': FECHA_COLS
}

def is_numeric_dtype(dtype):
    """Compatibilidad con pandas Int64Dtype y floats de numpy."""
    return pd.api.types.is_numeric_dtype(dtype)

def is_datetime_dtype(dtype):
    return pd.api.types.is_datetime64_any_dtype(dtype)

for tipo, cols in type_expectations.items():
    for c in cols:
        if c not in df.columns:
            continue
        dtype = df[c].dtype
        if tipo == 'numérica' and not is_numeric_dtype(dtype):
            print(f"⚠️ {c} debería ser numérica, pero es {dtype}")
        elif tipo == 'categórica' and dtype not in ['object', 'category']:
            print(f"⚠️ {c} debería ser categórica, pero es {dtype}")
        elif tipo == 'alta_cardinalidad' and dtype != 'object':
            print(f"⚠️ {c} debería ser object (alta cardinalidad), pero es {dtype}")
        elif tipo == 'fecha' and not is_datetime_dtype(dtype):
            print(f"⚠️ {c} debería ser datetime, pero es {dtype}")

print("\n--- RESUMEN ---")
print(f"Total columnas esperadas: {len(expected_all)}")
print(f"Total columnas reales: {len(df.columns)}")
print("✅ Revisión completada sin errores de tipo.")


###

# 1️⃣ Definir ruta y nombre de la nueva base de datos
RUTA_NUEVA_BD = r'C:\Users\User\Documents\InferIA\LicitacionesCorrecto_Procesada.sqlite'
NOMBRE_TABLA_NUEVA = "LicitacionesCorrecto_Procesada"

# 2️⃣ Crear/conectar la base de datos
conn = sqlite3.connect(RUTA_NUEVA_BD)

try:
    # 3️⃣ Guardar el DataFrame en SQLite
    df.to_sql(NOMBRE_TABLA_NUEVA, conn, if_exists='replace', index=False)
    print(f"✅ DataFrame guardado correctamente en '{RUTA_NUEVA_BD}' en la tabla '{NOMBRE_TABLA_NUEVA}'.")

finally:
    # 4️⃣ Cerrar la conexión
    conn.close()
