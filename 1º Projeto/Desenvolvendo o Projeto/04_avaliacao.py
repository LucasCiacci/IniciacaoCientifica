import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
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
    print("Preparando os dados para avaliação...")
    X = btc[["Close", "SMA_14", "EMA_14", "RSI_14", "Volatility", "Volume_MA_14"]]
    y = btc["Target"].astype(int)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Carregar o modelo treinado
def carregar_modelo(nome_arquivo="modelo_rna_treinado.keras"):
    caminho_modelo = os.path.join(BASE_DIR, nome_arquivo)
    print("Carregando modelo treinado...")
    modelo = load_model(caminho_modelo)
    print("Modelo carregado com sucesso!")
    return modelo

# 4. Avaliar o modelo
def avaliar_modelo(modelo, X_teste, y_teste):
    print("Realizando previsões...")
    previsoes = (modelo.predict(X_teste) > 0.5).astype(int)
    
    acuracia = accuracy_score(y_teste, previsoes)
    matriz_confusao = confusion_matrix(y_teste, previsoes)
    relatorio_classificacao = classification_report(y_teste, previsoes)
    
    print(f"Acurácia do modelo: {acuracia:.4f}")
    print("Matriz de Confusão:")
    print(matriz_confusao)
    print("Relatório de Classificação:")
    print(relatorio_classificacao)
    
    # Plotando matriz de confusão
    plt.figure(figsize=(6, 4))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Desceu', 'Subiu'], yticklabels=['Desceu', 'Subiu'])
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de Confusão")
    plt.show()

# Executando a avaliação
def main():
    btc = carregar_dados()
    X_treino, X_teste, y_treino, y_teste = preparar_dados(btc)
    modelo = carregar_modelo()
    avaliar_modelo(modelo, X_teste, y_teste)
    print("Processo de avaliação concluído!")

if __name__ == "__main__":
    main()
