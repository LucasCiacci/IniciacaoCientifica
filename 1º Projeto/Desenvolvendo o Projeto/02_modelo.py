import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

# Configurando a saída para evitar erros de encoding
sys.stdout.reconfigure(encoding='utf-8')

# Caminho base da pasta onde o script está salvo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Carregar os dados processados do arquivo CSV
def carregar_dados():
    print("Carregando dados...")
    caminho_dados = os.path.join(BASE_DIR, "bitcoin_data.csv")
    btc = pd.read_csv(caminho_dados, index_col=0)
    print(f"Dados carregados com sucesso! {btc.shape[0]} linhas e {btc.shape[1]} colunas.")
    return btc

# 2. Separar features e rótulo
def preparar_dados(btc):
    print("Preparando os dados para o modelo...")
    X = btc[["Close", "SMA_14", "EMA_14", "RSI_14", "Volatility", "Volume_MA_14"]]
    y = btc["Target"]
    return X, y

# 3. Criar o modelo da Rede Neural Artificial
def criar_modelo():
    print("Criando o modelo de Rede Neural...")
    modelo = Sequential([
        Input(shape=(6,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    print("Modelo criado com sucesso!")
    return modelo

# 4. Salvar o modelo para uso posterior
def salvar_modelo(modelo, nome_arquivo="modelo_rna.keras"):
    caminho_modelo = os.path.join(BASE_DIR, nome_arquivo)
    modelo.save(caminho_modelo, include_optimizer=False)
    print(f"Modelo salvo em '{caminho_modelo}'")

# Executando a construção do modelo
def main():
    btc = carregar_dados()
    X, y = preparar_dados(btc)
    modelo = criar_modelo()
    salvar_modelo(modelo)
    print("Processo de criação do modelo concluído!")

if __name__ == "__main__":
    main()
