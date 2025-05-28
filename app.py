# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from PIL import Image
import io
import base64
from sklearn.metrics import confusion_matrix, classification_report

# Configura��o inicial do Streamlit
st.set_page_config(
    page_title="Classificador de D�gitos MNIST - Filipe Tchivela",
    page_icon="\u270d\ufe0f",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para um visual premium
st.markdown("""
    <style>
    .main {background-color: #f0f2f6;}
    .stButton>button {background-color: #1f77b4; color: white; border-radius: 10px;}
    .stButton>button:hover {background-color: #135a8d;}
    .sidebar .sidebar-content {background-color: #2e2e2e; color: white;}
    .stTitle {font-size: 2.5em; color: #1f77b4;}
    .stHeader {font-size: 1.8em; color: #333;}
    .stCaption {color: #555;}
    </style>
""", unsafe_allow_html=True)

# Fun��o para carregar o modelo
@st.cache_resource
def load_model():
    return joblib.load('mnist_logistic_model.pkl')

# Fun��o para pr�-processar a imagem carregada
def preprocess_image(image):
    # Converter para grayscale e redimensionar para 28x28
    img = image.convert('L').resize((28, 28))
    # Converter para array numpy e normalizar
    img_array = np.array(img) / 255.0
    # Achatar a imagem para o formato do modelo (1, 784)
    img_flat = img_array.reshape(1, -1)
    return img_flat, img_array

# Fun��o para codificar a imagem em base64 para exibi��o
def get_image_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Carregar o modelo
model = load_model()

# Carregar os dados de teste para m�tricas (opcional, se dispon�veis)
@st.cache_data
def load_test_data():
    test_df = pd.read_csv("mnist_test.csv")
    X_test = test_df.drop("label", axis=1) / 255.0
    y_test = test_df["label"]
    return X_test, y_test

# Sidebar com navega��o
st.sidebar.title("Navega��o")
page = st.sidebar.radio("Ir para:", ["Home", "Sobre Mim", "An�lise do Modelo", "Testar o Modelo"])

# P�gina principal (Home)
if page == "Home":
    st.title("\u270d\ufe0f Classificador de D�gitos MNIST")
    st.markdown("""
    Bem-vindo � aplica��o de classifica��o de d�gitos manuscritos do dataset MNIST, desenvolvida por **Filipe Tchivela**.  
    Esta aplica��o utiliza um modelo de **Regress�o Log�stica** treinado para reconhecer d�gitos de 0 a 9 a partir de imagens 28x28 pixels.  
    Navegue pelo menu lateral para:
    - **Sobre Mim**: Conhe�a o desenvolvedor.
    - **An�lise do Modelo**: Veja m�tricas e visualiza��es do desempenho.
    - **Testar o Modelo**: Carregue suas pr�prias imagens para previs�o.
    """)

    # Exibir exemplos aleat�rios do conjunto de teste
    st.header("Exemplos de D�gitos")
    try:
        X_test, y_test = load_test_data()
        indices = np.random.choice(len(X_test), 5, replace=False)
        cols = st.columns(5)
        for i, idx in enumerate(indices):
            with cols[i]:
                img = X_test.iloc[idx].values.reshape(28, 28)
                st.image(img, caption=f"D�gito: {y_test.iloc[idx]}", use_column_width=True)
    except FileNotFoundError:
        st.warning("Arquivo 'mnist_test.csv' n�o encontrado. Carregue os dados para visualizar exemplos.")

# P�gina Sobre Mim
elif page == "Sobre Mim":
    st.title("Sobre Mim")
    st.markdown("""
    <div style='text-align: center;'>
        <h2>Filipe Tchivela</h2>
        <p>Estudante do 3� ano de Ci�ncia da Computa��o na Universidade Mndume</p>
        <p>Apaixonado por intelig�ncia artificial, aprendizado de m�quina e desenvolvimento de software.</p>
        <p>Este projeto � parte do meu portf�lio, demonstrando habilidades em machine learning e desenvolvimento de aplica��es interativas com Streamlit.</p>
        <p>Contato: filipe.tchivela@example.com | LinkedIn: <a href='https://linkedin.com/in/filipe-tchivela'>filipe-tchivela</a></p>
    </div>
    """, unsafe_allow_html=True)

# P�gina An�lise do Modelo
elif page == "An�lise do Modelo":
    st.title("An�lise do Modelo")
    st.header("Desempenho do Modelo de Regress�o Log�stica")
    
    try:
        X_test, y_test = load_test_data()
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        st.subheader("Acur�cia Geral")
        st.write(f"Acur�cia no conjunto de teste: **{acc:.4f}**")

        # Matriz de Confus�o
        st.subheader("Matriz de Confus�o")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
        ax.set_xlabel('Classe Predita')
        ax.set_ylabel('Classe Verdadeira')
        ax.set_title('Matriz de Confus�o')
        st.pyplot(fig)

        # Relat�rio de Classifica��o
        st.subheader("Relat�rio de Classifica��o")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.table(pd.DataFrame(report).transpose())

        # Resumo
        st.subheader("Resumo")
        st.write("""
        O modelo de Regress�o Log�stica apresenta um desempenho equilibrado para a classifica��o de d�gitos MNIST.  
        Algumas classes (como o d�gito 5) podem ter desempenho ligeiramente inferior devido a semelhan�as visuais com outros d�gitos.  
        **Pr�ximos passos sugeridos**:
        - Testar modelos mais complexos (SVM, Random Forest, CNNs).
        - Aplicar aumento de dados (data augmentation).
        - Ajustar hiperpar�metros.
        """)
    except FileNotFoundError:
        st.error("Arquivo 'mnist_test.csv' n�o encontrado. Carregue os dados para visualizar as m�tricas.")

# P�gina Testar o Modelo
elif page == "Testar o Modelo":
    st.title("Testar o Modelo")
    st.header("Carregue uma Imagem para Previs�o")
    
    uploaded_file = st.file_uploader("Escolha uma imagem de um d�gito (28x28 pixels, escala de cinza)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Carregar e exibir a imagem
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem Carregada", width=200)
        
        # Pr�-processar a imagem
        img_flat, img_array = preprocess_image(image)
        
        # Fazer a previs�o
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]
        
        # Exibir o resultado
        st.subheader("Resultado da Previs�o")
        st.write(f"**D�gito Previsto**: {prediction}")
        
        # Exibir probabilidades
        st.subheader("Probabilidades por Classe")
        prob_df = pd.DataFrame({
            "D�gito": range(10),
            "Probabilidade (%)": [f"{p*100:.2f}" for p in probabilities]
        })
        st.table(prob_df)

        # Visualizar a imagem pr�-processada
        st.subheader("Imagem Pr�-processada (28x28)")
        st.image(img_array, caption="Imagem ap�s redimensionamento", width=200)