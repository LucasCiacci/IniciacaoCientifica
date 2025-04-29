import yfinance as yf
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler

# Caminho base da pasta onde o script está salvo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. Baixando os dados do Bitcoin (BTC-USD) dos últimos 2 anos
def baixar_dados():
    print("Baixando dados do Bitcoin...")
    btc = yf.download("BTC-USD", period="2y", interval="1d")
    print("Download concluído!")
    return btc

# 2. Criando indicadores técnicos
def calcular_indicadores(btc):
    print("Calculando indicadores técnicos...")
    btc["SMA_14"] = btc["Close"].rolling(window=14).mean()
    btc["EMA_14"] = btc["Close"].ewm(span=14, adjust=False).mean()
    btc["RSI_14"] = 100 - (100 / (1 + btc["Close"].pct_change().rolling(14)
                                   .apply(lambda x: (x[x > 0].sum() / -x[x < 0].sum()) if -x[x < 0].sum() != 0 else 1)))
    btc["Volatility"] = btc["Close"].rolling(window=14).std()
    btc["Volume_MA_14"] = btc["Volume"].rolling(window=14).mean()
    btc["Target"] = (btc["Close"].shift(-1) > btc["Close"]).astype(int)
    print("Indicadores calculados com sucesso!")
    return btc

# 3. Preparação e Limpeza dos Dados
def preparar_dados(btc):
    print("Realizando limpeza e preparação dos dados...")
    btc.dropna(inplace=True)
    scaler = MinMaxScaler()
    colunas_para_normalizar = ["Close", "SMA_14", "EMA_14", "RSI_14", "Volatility", "Volume_MA_14"]
    btc[colunas_para_normalizar] = scaler.fit_transform(btc[colunas_para_normalizar])
    print("Dados preparados com sucesso!")
    return btc

# 4. Salvando os dados
def salvar_dados(btc):
    caminho_arquivo = os.path.join(BASE_DIR, "bitcoin_data.csv")
    btc.to_csv(caminho_arquivo, index=True)
    print(f"Dados salvos em '{caminho_arquivo}'")

# Executando o pipeline completo
def main():
    dados_btc = baixar_dados()
    dados_btc = calcular_indicadores(dados_btc)
    dados_btc = preparar_dados(dados_btc)
    salvar_dados(dados_btc)
    print("Processo concluído!")

if __name__ == "__main__":
    main()
