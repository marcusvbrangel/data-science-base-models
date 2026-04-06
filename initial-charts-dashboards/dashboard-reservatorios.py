
'''
NOTES:
para executar faça:
streamlit run dashboard-reservatorios.py
'''

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest

# --- CONFIGURAÇÃO DA PÁGINA (UI/UX) ---
st.set_page_config(page_title="PetroStream - Dashboard de Reservatórios", layout="wide")

# --- TITULO E BARRA LATERAL ---
st.title("🚢 Monitor de Ativos Offshore - Pré-Sal")
st.sidebar.header("Painel de Controle")
st.sidebar.markdown("Selecione os parâmetros para análise:")

# --- 1. SIMULAÇÃO DE DADOS (O Coração da Matriz) ---
@st.cache_data # Isso faz o app ser ultra rápido, cacheando os dados
def carregar_dados():
    dias = np.arange(1, 101)
    dados = pd.DataFrame({
        'Dia': dias,
        'Pressao_PSI': 3500 - (dias * 2) + np.random.randn(100) * 10,
        'Vazao_bpd': 5000 + np.random.randn(100) * 200,
        'Temperatura_C': 60 + np.random.randn(100) * 2
    })
    # Injetando uma anomalia proposital no dia 80
    dados.loc[80, 'Pressao_PSI'] = 4500 
    return dados

df = carregar_dados()

# --- 2. FILTROS INTERATIVOS (A Experiência do Usuário) ---
filtro_dias = st.sidebar.slider("Intervalo de Dias", 1, 100, (1, 100))
df_filtrado = df[(df['Dia'] >= filtro_dias[0]) & (df['Dia'] <= filtro_dias[1])]

# --- 3. INDICADORES CHAVE (KPIs) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Pressão Atual", f"{df_filtrado['Pressao_PSI'].iloc[-1]:.0f} PSI", delta="-2% (Queda)")
with col2:
    st.metric("Vazão Média", f"{df_filtrado['Vazao_bpd'].mean():.0f} bpd")
with col3:
    st.metric("Status do Ativo", "Operacional", delta_color="normal")

# --- 4. VISUALIZAÇÃO INTERATIVA (Plotly + Streamlit) ---
st.subheader("Análise de Tendência e Escoamento")
opcao_grafico = st.selectbox("Escolha a Variável para Visualizar", ["Pressao_PSI", "Vazao_bpd", "Temperatura_C"])

fig = px.line(df_filtrado, x='Dia', y=opcao_grafico, 
              title=f"Evolução de {opcao_grafico} no Tempo",
              template="plotly_dark") # Estilo Dark para parecer sala de controle
st.plotly_chart(fig, use_container_width=True)

# --- 5. INTELIGÊNCIA ARTIFICIAL: DETECÇÃO DE ANOMALIAS (O "Cérebro") ---
st.divider()
st.subheader("🤖 Diagnóstico de IA: Detecção de Falhas")

if st.button("Executar Varredura de Anomalias"):
    # Usando o Isolation Forest que estudamos
    model = IsolationForest(contamination=0.05)
    df_filtrado['Anomalia'] = model.fit_predict(df_filtrado[[opcao_grafico]])
    
    anomalias = df_filtrado[df_filtrado['Anomalia'] == -1]
    
    if not anomalias.empty:
        st.warning(f"Atenção! Detectamos {len(anomalias)} pontos anômalos no sensor.")
        st.write(anomalias)
        
        fig_anomalia = px.scatter(df_filtrado, x='Dia', y=opcao_grafico, color='Anomalia',
                                 color_discrete_map={1: 'blue', -1: 'red'},
                                 title="Pontos Vermelhos indicam Potencial Falha de Sensor")
        st.plotly_chart(fig_anomalia, use_container_width=True)
    else:
        st.success("Nenhuma anomalia crítica detectada nos sensores.")

# --- 6. EXPORTAÇÃO (O Dia a Dia do Engenheiro) ---
st.sidebar.divider()
csv = df_filtrado.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("Baixar Relatório (CSV)", data=csv, file_name="relatorio_poco.csv")
