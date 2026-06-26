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

@st.cache_data(ttl=600)  # Actualiza los datos cada 10 minutos
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
    st.error("🔑 No se encontró la GOOGLE_API_KEY en los secretos.")
    st.stop()

# 4. Inicializar el modelo Gemini
llm = ChatGoogleGenerativeAI(model="gemini-flash-latest", temperature=0)

# 5. Configurar el Agente Analista
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type="zero-shot-react-description",
    allow_dangerous_code=True
)

# 6. Interfaz del Chat
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "¡Hola! Estoy listo para consultar la base de datos de indicadores. ¿Qué necesitas saber?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ej: ¿Cuál es el valor del indicador en 2024?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Analizando la base de datos..."):
            try:
                response = agent.invoke(prompt)
                output_text = response["output"]
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})
            except Exception as e:
                st.error(f"Ocurrió un error al procesar la consulta: {e}")