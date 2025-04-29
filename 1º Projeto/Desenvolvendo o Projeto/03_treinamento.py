import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Caminho base da pasta onde o script está salvo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Carregar os dados processados do arquivo CSV
def carregar_dados():
    print("Carregando dados...")
    caminho_dados = os.path.join(BASE_DIR, "bitcoin_data.csv")
    btc = pd.read_csv(caminho_dados, index_col=0)
    
    btc = btc.apply(pd.to_numeric, errors='coerce')
    btc.dropna(inplace=True)
    
    print(f"Dados carregados com sucesso! {btc.shape[0]} linhas e {btc.shape[1]} colunas.")
    return btc

# 2. Separar features e rótulo
def preparar_dados(btc):
    print("Preparando os dados para treinamento...")
    X = btc[["Close", "SMA_14", "EMA_14", "RSI_14", "Volatility", "Volume_MA_14"]]
    y = btc["Target"]
    
    X = X.astype(float)
    y = y.astype(int)
    
    return X, y

# 3. Carregar o modelo treinado
def carregar_modelo(nome_arquivo="modelo_rna.keras"):
    caminho_modelo = os.path.join(BASE_DIR, nome_arquivo)
    print("Carregando modelo salvo...")
    modelo = load_model(caminho_modelo)
    print("Modelo carregado com sucesso!")
    return modelo

# 4. Treinar o modelo
def treinar_modelo(modelo, X, y, epochs=50, batch_size=32):
    print("Dividindo os dados em treino e teste...")
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Iniciando o treinamento...")
    historico = modelo.fit(X_treino, y_treino, epochs=epochs, batch_size=batch_size, validation_data=(X_teste, y_teste))
    print("Treinamento concluído!")
    
    return modelo, historico

# 5. Salvar modelo atualizado após treinamento
def salvar_modelo(modelo, nome_arquivo="modelo_rna_treinado.keras"):
    caminho_modelo = os.path.join(BASE_DIR, nome_arquivo)
    modelo.save(caminho_modelo)
    print(f"Modelo treinado salvo em '{caminho_modelo}'")

# Executando o treinamento
def main():
    btc = carregar_dados()
    X, y = preparar_dados(btc)
    modelo = carregar_modelo()
    modelo, historico = treinar_modelo(modelo, X, y)
    salvar_modelo(modelo)
    print("Processo de treinamento concluído!")

if __name__ == "__main__":
    main()
