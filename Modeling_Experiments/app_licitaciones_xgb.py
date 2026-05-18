import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Simulador de Licitaciones IA", layout="wide")

# abrir el anaconda prompt
# cd /d C:\Users\User\Documents\InferIA
# streamlit run app_licitaciones.py

# ------------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ------------------------------------------------------------
# Ajusta esto a tu ruta local
RUTA = Path(r"C:\Users\User\Documents\InferIA") 

TRAIN_PATH = RUTA / "train_procesado_v2_limpio.parquet"
VAL_PATH   = RUTA / "val_procesado_v2_limpio.parquet"
TEST_PATH  = RUTA / "test_procesado_v2_limpio.parquet"

# Modelo ganador XGBoost (predice % de baja)
MODEL_PATH = RUTA / "salida_modelo_XGB_Baja" / "xgb_ganador_baja.pkl"
EMPRESA_COL = 'lote_adjudicatario'

# ------------------------------------------------------------
# FUNCIONES
# ------------------------------------------------------------

@st.cache_resource
def cargar_datos_y_modelo():
    """Carga modelo y construye la base de datos de perfiles (Cached)"""
    modelo = joblib.load(MODEL_PATH)
    
    df_train = pd.read_parquet(TRAIN_PATH)
    df_val   = pd.read_parquet(VAL_PATH)
    df_test  = pd.read_parquet(TEST_PATH)
    full_df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # Limpieza idéntica a tu script
    if 'es_exito' in full_df.columns: full_df = full_df[full_df['es_exito'] == 1]
    target = 'lote_importe_adjudicacion_sin_impuestos'
    presu = 'lote_presupuesto_base_sin_impuestos'
    if target in full_df.columns and presu in full_df.columns:
        full_df = full_df[full_df[target] <= full_df[presu]]
    if target in full_df.columns: full_df = full_df[full_df[target] > 0]

    full_df['fecha_primera_publicacion'] = pd.to_datetime(full_df['fecha_primera_publicacion'])
    full_df = full_df.sort_values('fecha_primera_publicacion')

    perfil_reciente = full_df.drop_duplicates(subset=[EMPRESA_COL], keep='last').set_index(EMPRESA_COL)
    
    cols_utiles = [c for c in perfil_reciente.columns if 'hist' in c or 'media' in c or 'medio' in c or 'pct_' in c]
    for c in ['fecha_primera_publicacion', 'com_aut_adjudicador', 'presupuesto_medio_hist']:
        if c not in cols_utiles and c in perfil_reciente.columns: cols_utiles.append(c)
        
    df_perfiles = perfil_reciente[cols_utiles].copy()
    df_perfiles.rename(columns={'fecha_primera_publicacion': 'fecha_ultima_licitacion_empresa'}, inplace=True)
    
    # --- ¡NUEVO! FILTRO DE CORDURA (SANITY CHECK) ---
    # Eliminamos empresas cuyo descuento medio histórico sea absurdo (> 90%)
    # Esto elimina a las que tienen errores de precio unitario o datos corruptos.
    if 'descuento_medio_hist' in df_perfiles.columns:
        n_antes = len(df_perfiles)
        df_perfiles = df_perfiles[df_perfiles['descuento_medio_hist'] < 0.90] # Máximo 90% de descuento medio
        n_despues = len(df_perfiles)
        # (Opcional) Podríamos mostrar un aviso en logs, pero en Streamlit mejor silencioso o st.write
        
    return df_perfiles, modelo

def pct_baja_a_euros(pct_baja, presupuesto_base):
    """Convierte una predicción de baja (%) en oferta adjudicada (€)."""
    pct_baja_segura = np.clip(np.asarray(pct_baja, dtype=float), 0.0, 100.0)
    presupuesto_base = np.asarray(presupuesto_base, dtype=float)
    return presupuesto_base * (1.0 - pct_baja_segura / 100.0)

def encontrar_rivales(licitacion_dict, df_perfiles, tolerancia, umbral_tipo, umbral_geo, filtrar_geo, max_dias):
    rivales = df_perfiles.copy()
    
    # 1. Especialización
    tipo_norm = licitacion_dict['tipo_contrato'].lower().replace(' ', '_')
    col_esp = f"pct_hist_{tipo_norm}"
    if col_esp in rivales.columns:
        rivales = rivales[rivales[col_esp] >= umbral_tipo]

    # 2. Tamaño
    presu = licitacion_dict['lote_presupuesto_base_sin_impuestos']
    if 'presupuesto_medio_hist' in rivales.columns:
        min_p = presu * (1 - tolerancia)
        max_p = presu * (1 + tolerancia)
        rivales = rivales[rivales['presupuesto_medio_hist'].between(min_p, max_p)]
    
    # 3. Geografía
    if filtrar_geo:
        ca = licitacion_dict['com_aut_licitacion'].lower().replace(' ', '_')
        col_geo = f"pct_hist_com_aut_{ca}"
        if col_geo in rivales.columns:
            rivales = rivales[rivales[col_geo] >= umbral_geo]

    # 4. Recencia
    if max_dias is not None and 'fecha_ultima_licitacion_empresa' in rivales.columns:
        fecha_hoy = pd.to_datetime(licitacion_dict['fecha_primera_publicacion'])
        dias = (fecha_hoy - rivales['fecha_ultima_licitacion_empresa']).dt.days
        rivales = rivales[dias <= max_dias]
        
    if len(rivales) > 100:
        rivales = rivales.sort_values('n_licitaciones_hist', ascending=False).head(100)
        
    return rivales

def predecir_escenario(licitacion_dict, df_rivales, modelo):
    simulacion_df = pd.DataFrame([licitacion_dict] * len(df_rivales))
    
    for col in df_rivales.columns:
        if col in df_rivales.columns and col != 'fecha_ultima_licitacion_empresa': 
             simulacion_df[col] = df_rivales[col].values

    if 'fecha_ultima_licitacion_empresa' in df_rivales.columns:
        fecha_nueva = pd.to_datetime(licitacion_dict['fecha_primera_publicacion'])
        fechas_empresas = pd.to_datetime(df_rivales['fecha_ultima_licitacion_empresa'].values)
        simulacion_df['dias_desde_ultima_licitacion'] = (fecha_nueva - fechas_empresas).days
        simulacion_df['dias_desde_ultima_licitacion'] = simulacion_df['dias_desde_ultima_licitacion'].fillna(365).clip(lower=0)
    
    # Parches técnicos
    simulacion_df['presupuesto_base_sin_impuestos'] = simulacion_df['lote_presupuesto_base_sin_impuestos']
    if 'lote' not in simulacion_df.columns: simulacion_df['lote'] = 1
    if 'peso_relativo_lote' not in simulacion_df.columns: simulacion_df['peso_relativo_lote'] = 1.0
    for c in ['es_loteado', 'es_anomalia_temporal']: simulacion_df[c] = 0

    cat_cols = simulacion_df.select_dtypes(include=['object', 'category']).columns
    for c in cat_cols: simulacion_df[c] = simulacion_df[c].astype(str)

    try:
        pred_baja_pct = modelo.predict(simulacion_df)
        presu = float(licitacion_dict['lote_presupuesto_base_sin_impuestos'])
        pred_euros = pct_baja_a_euros(pred_baja_pct, presu)
        
        df_res = df_rivales.copy()
        df_res['oferta_predicha_€'] = pred_euros
        df_res['baja_estimada_%'] = (presu - pred_euros) / presu * 100
        
        # Añadir recencia para visualización
        if 'dias_desde_ultima_licitacion' in simulacion_df.columns:
            df_res['dias_desde_ultima_licitacion'] = simulacion_df['dias_desde_ultima_licitacion'].values

        return df_res.sort_values('oferta_predicha_€')
    except Exception as e:
        st.error(f"Error predicción: {e}")
        return None

# ------------------------------------------------------------
# INTERFAZ STREAMLIT
# ------------------------------------------------------------

st.title("🏗️ Simulador de Licitaciones con IA")

# Carga de datos
with st.spinner("Cargando datos y modelo..."):
    try:
        df_perfiles, modelo = cargar_datos_y_modelo()
        st.success(f"Sistema online. Base de datos: {len(df_perfiles):,} empresas.")
    except Exception as e:
        st.error(f"Error crítico de carga: {e}")
        st.stop()

# --- LÓGICA PARA EXTRAER LISTAS DINÁMICAS ---
# Obtenemos todas las columnas que empiezan por 'pct_hist_' para saber los tipos
all_cols = df_perfiles.columns.tolist()

# 1. Tipos de Contrato Disponibles
# Buscamos columnas tipo 'pct_hist_servicios', etc. pero que NO sean de comunidades autónomas
lista_tipos = [c.replace('pct_hist_', '') for c in all_cols if c.startswith('pct_hist_') and 'com_aut' not in c]
lista_tipos = sorted(list(set(lista_tipos))) # Únicos y ordenados

# 2. Comunidades Autónomas Disponibles
lista_ccaa = [c.replace('pct_hist_com_aut_', '') for c in all_cols if c.startswith('pct_hist_com_aut_')]
lista_ccaa = sorted(list(set(lista_ccaa)))

# --- SIDEBAR: CONFIGURACIÓN ---
with st.sidebar:
    st.header("1. Configurar Licitación")
    presupuesto = st.number_input("Presupuesto Base (€)", value=150000, step=1000)
    
    # Usamos las listas dinámicas
    tipo_contrato = st.selectbox("Tipo Contrato", lista_tipos)
    comunidad = st.selectbox("Comunidad Autónoma", lista_ccaa)
    
    st.header("2. Filtros de Competencia")
    tolerancia = st.slider("Margen Presupuesto (+/-)", 0.1, 1.0, 0.6, help="Busca empresas que suelan licitar por este importe +/- X%")
    umbral_tipo = st.slider(f"Espec. en {tipo_contrato}", 0.0, 1.0, 0.2, help="% mínimo de contratos de este tipo ganados")
    
    filtrar_geo = st.checkbox("Filtrar por Geografía", value=True)
    umbral_geo = 0.0
    if filtrar_geo:
        umbral_geo = st.slider(f"Experiencia en {comunidad}", 0.0, 1.0, 0.1, help="% mínimo de contratos ganados en esta C.A.")
    
    max_dias = st.number_input("Máx. Días Inactividad", value=730, help="Descartar empresas que no ganan nada desde hace X días")
    
    btn_calc = st.button("🔍 Simular Subasta", type="primary")

# --- LÓGICA PRINCIPAL ---
if btn_calc:
    # Construir diccionario
    nueva_licitacion = {
        'lote_presupuesto_base_sin_impuestos': presupuesto,
        'valor_estimado_imputado': presupuesto,
        'duracion_proceso_dias': 45,
        'anio': 2025, 'mes': 5,
        'tipo_contrato': tipo_contrato,
        'com_aut_licitacion': comunidad,
        'sistema_contratacion': 'No aplica',
        'tipo_procedimiento': 'Abierto',
        'tipo_administracion': 'Administracion Local',
        'lote_pyme': 'No', 
        'lote_tipo_id_adjudicatario': 'NIF Definitivo',
        'organo_cluster_label': 'ORG_Ayto_General', 
        'objeto_cluster_label': 'OBJ_General',      
        'fecha_primera_publicacion': '2025-08-01'
    }

    rivales = encontrar_rivales(nueva_licitacion, df_perfiles, tolerancia, umbral_tipo, umbral_geo, filtrar_geo, max_dias)
    
    if not rivales.empty:
        resultados = predecir_escenario(nueva_licitacion, rivales, modelo)
        
        if resultados is not None:
            # --- DASHBOARD ---
            
            # KPIs
            mediana = resultados['oferta_predicha_€'].median()
            minimo = resultados['oferta_predicha_€'].min()
            baja_med = resultados['baja_estimada_%'].median()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Presupuesto", f"{presupuesto:,.0f} €")
            col2.metric("Precio Mercado (IA)", f"{mediana:,.0f} €", delta=f"-{baja_med:.1f}% Baja")
            
            # CORRECCIÓN: delta_color="inverse"
            col3.metric("Oferta Más Agresiva", f"{minimo:,.0f} €", delta_color="inverse")
            
            # Gráfico
            st.subheader("Distribución de Ofertas")
            fig, ax = plt.subplots(figsize=(10, 3))
            sns.histplot(resultados['oferta_predicha_€'], kde=True, ax=ax, color="skyblue")
            ax.axvline(mediana, color='red', linestyle='--', label='Mediana')
            ax.set_xlabel("Oferta (€)")
            st.pyplot(fig)
            
            # Tabla
            st.subheader(f"Top Competidores ({len(rivales)} encontrados)")
            
            col_tipo = f"pct_hist_{tipo_contrato}"
            col_geo = f"pct_hist_com_aut_{comunidad}"
            
            cols_show = ['oferta_predicha_€', 'baja_estimada_%', 'descuento_medio_hist', 'n_licitaciones_hist', 'dias_desde_ultima_licitacion']
            if col_tipo in resultados.columns: cols_show.append(col_tipo)
            if col_geo in resultados.columns: cols_show.append(col_geo)
            
            st.dataframe(
                resultados[cols_show].style.format({
                    'oferta_predicha_€': '{:,.0f}€',
                    'baja_estimada_%': '{:.1f}%',
                    'descuento_medio_hist': '{:.1%}',
                    'dias_desde_ultima_licitacion': '{:.0f}',
                    col_tipo: '{:.1%}',
                    col_geo: '{:.1%}'
                }),
                use_container_width=True
            )
            
    else:
        st.warning("⚠️ No se encontraron empresas con los filtros seleccionados. Intenta ampliar la tolerancia.")
else:
    st.info("👈 Configura la licitación en el menú lateral y pulsa 'Simular'.")