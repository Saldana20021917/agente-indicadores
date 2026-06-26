import streamlit as st
import pandas as pd
import os
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Configuración de la interfaz web
st.set_page_config(page_title="Agente de Indicadores", page_icon="📊", layout="wide")
st.title("📊 Agente Inteligente de Indicadores Consolidados")

# 2. Conexión a Google Sheets (Enlace directo a CSV)
SHEET_URL = "https://docs.google.com/spreadsheets/d/1zvCu0hgikR0uHFCPNCKbpN5jdqFUKBur/export?format=csv"

@st.cache_data(ttl=600)  # Actualiza los datos en caché cada 10 minutos
def cargar_datos(url):
    return pd.read_csv(url)

try:
    df = cargar_datos(SHEET_URL)
    st.success(f"✅ Base de datos sincronizada. Registros detectados: {df.shape[0]}")
except Exception as e:
    st.error(f"Error al conectar con la base de datos: {e}")
    st.stop()

# 3. Configuración de la API Key de Gemini
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("🔑 No se encontró la GOOGLE_API_KEY en los secretos de Streamlit.")
    st.stop()

# 4. Inicializar el modelo Gemini
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# 5. INSTRUCCIONES ESTADÍSTICAS Y DE NEGOCIO (Contexto del Dataset)
INSTRUCCIONES_ANALISTA = """
Eres un analista de datos experto en indicadores socioeconómicos de Colombia.
Operas sobre un DataFrame de pandas llamado `df`.

REGLAS DE PROCESAMIENTO ESTADÍSTICO:
1. FILTRADO: Antes de operar, filtra la columna 'Indicador' usando coincidencia exacta o parcial inteligente (ej. df[df['Indicador'] == '% acceso a internet']).
2. COINCIDENCIA DE CATEGORÍAS: 
   - 'Region' puede ser: 'Antioquia', 'Área de influencia', 'Caribe', 'Nacional', 'Pacífico', 'Resto', 'Santanderes'.
   - 'Año' contiene valores numéricos (ej. 2024, 2025).
   - 'Depto' contiene los nombres de los 32 departamentos, Bogotá y la categoría consolidada 'Colombia'.
3. SUMAS VS PROMEDIOS (CRÍTICO):
   - Si el indicador expresa tasas, porcentajes, proporciones o incidencias (ej. '% acceso a internet', '% desempleo jóvenes mujeres', 'Incidencia ajustada', 'Porcentaje de personas...'), NUNCA sumes los valores directos entre territorios o años. Usa siempre el promedio aritmético `.mean()` para consolidar o agrupar.
   - Si el indicador representa recuentos absolutos o totales (ej. 'Personas en pobreza monetaria (miles)', 'Total homicidios anual', 'Cantidad de micronegocios', 'Total Extorsión anual'), usa siempre la suma `.sum()` al consolidar regiones o agrupaciones.
4. AGRUPACIONES: Cuando te pidan comparar o consolidar por zonas, usa correctamente estructurado el comando `df.groupby()`.
5. RESPUESTA: Entrega la respuesta de forma natural, directa y fluida en español. NO narres los pasos técnicos que realizaste (evita decir cosas como "apliqué la regla", "usé la función .mean()", o "filtré los datos por la columna"). Actúa como un consultor presentando un informe ejecutivo limpio, claro y analítico.
"""

# 6. Configurar el Agente Analista con Tool Calling
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type="tool-calling",  # Invoca funciones nativas, evitando errores de parseo de texto
    prefix=INSTRUCCIONES_ANALISTA,
    handle_parsing_errors=True,  # Captura desviaciones remanentes de formato
    allow_dangerous_code=True
)

# 7. Interfaz del Chat e Historial de Mensajes
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Estoy listo para consultar la base de datos de indicadores socioeconómicos. ¿Qué necesitas saber?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ej: ¿Cuál es el promedio de % acceso a internet en el caribe en el año 2024?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Analizando la base de datos..."):
            try:
                response = agent.invoke({"input": prompt}) 
                raw_output = response.get("output", "No se generó un output válido.")
                
                # FILTRO DE LIMPIEZA: Extrae el texto plano si la API devuelve la estructura cruda con firmas y metadatos
                if isinstance(raw_output, list):
                    output_text = "".join([item.get("text", "") for item in raw_output if isinstance(item, dict) and "text" in item])
                elif isinstance(raw_output, dict) and "text" in raw_output:
                    output_text = raw_output["text"]
                else:
                    output_text = str(raw_output)
                
                # Mostrar resultado limpio en la interfaz web
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la consulta: {e}")
