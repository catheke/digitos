# app.py

import streamlit as st
import numpy as np
import joblib

# Configurar o layout da página
st.set_page_config(page_title="Classificador MNIST com XGBoost", layout="centered")
st.title("🧠 Classificador de Dígitos Manuscritos (MNIST)")

st.markdown("""
Esta aplicação usa um modelo treinado com **XGBoost** para prever dígitos manuscritos (0 a 9) com base no conjunto MNIST.

👉 Insere os 784 valores dos pixels (ou carrega uma imagem pré-processada em vetor).
""")

# Carregar o modelo treinado
try:
    model = joblib.load("mnist_model.pkl")
except FileNotFoundError:
    st.error("❌ O ficheiro 'mnist_model.pkl' não foi encontrado. Verifica se está no mesmo diretório.")
    st.stop()

# Criar um uploader de ficheiro com os dados de entrada
uploaded_file = st.file_uploader("📁 Carregar ficheiro CSV (com vetor de 784 pixels)", type=["csv"])

if uploaded_file is not None:
    try:
        # Ler o vetor do ficheiro carregado
        import pandas as pd
        input_data = pd.read_csv(uploaded_file, header=None).values

        if input_data.shape[1] != 784:
            st.error("❌ O ficheiro deve conter exatamente 784 colunas (pixels).")
        else:
            # Fazer a previsão
            prediction = model.predict(input_data)
            st.success(f"✅ Previsão do modelo: **{prediction[0]}**")
    except Exception as e:
        st.error(f"❌ Erro ao processar o ficheiro: {e}")

# Alternativa: inserção manual de 784 valores (oculto por padrão)
with st.expander("Ou preencher manualmente os 784 valores dos pixels (0-255)"):
    valores = st.text_area("Insere os 784 valores separados por vírgula", "")
    if st.button("🔮 Prever manualmente"):
        try:
            vetor = np.array([float(x) for x in valores.split(",")])
            if vetor.shape[0] != 784:
                st.error("❌ Precisam ser exatamente 784 valores.")
            else:
                prediction = model.predict(vetor.reshape(1, -1))
                st.success(f"✅ Previsão do modelo: **{prediction[0]}**")
        except:
            st.error("❌ Verifica se os valores estão correctamente inseridos.")

# Rodapé
st.markdown("---")
st.caption("🧪 Desenvolvido com XGBoost + Streamlit | Filipe Tchivela")
