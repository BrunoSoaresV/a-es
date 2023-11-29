import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime as dt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, LSTM
import yfinance as yf
import warnings
import os
import time
from acoes import selecionar_acao
ocultar = """<style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>"""
st.set_page_config(page_title="Ações", page_icon=None,
                   layout="centered", initial_sidebar_state="auto", menu_items=None)
st.markdown(ocultar, unsafe_allow_html=True)
st.title('Previsão do preço de ações')
acao = st.selectbox('Selecione a ação que você quer ver a previsão ', selecionar_acao)
warnings.filterwarnings('ignore')
yf.pdr_override()
inicio = dt.datetime(2010, 1, 1)
fim = dt.datetime(2023, 1, 1)
dados = web.DataReader(acao, inicio, fim)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler1=StandardScaler()
scaler_dados = scaler.fit_transform(dados['Close'].values.reshape(-1, 1))
dias = 60
x = []
y = []
for a in range(dias, len(scaler_dados)):
    x.append(scaler_dados[a-dias:a, 0])
    y.append(scaler_dados[a, 0])
x, y = np.array(x), np.array(y)
x = np.reshape(x, (x.shape[0], x.shape[1], 1))
modelo = Sequential()
modelo.add(LSTM(units=50, return_sequences=True, input_shape=(x.shape[1], 1)))
modelo.add(Dropout(0.2))
modelo.add(LSTM(units=50, return_sequences=True))
modelo.add(Dropout(0.2))
modelo.add(LSTM(units=50))
modelo.add(Dropout(0.2))
modelo.add(Dense(units=1))
if os.path.isfile('modelo.h5'):
    try:
        arquivo_json = open('modelo.json', 'r')
        modelocarregado = arquivo_json.read()
        arquivo_json.close()
        modelo = model_from_json(modelocarregado)
        modelo.load_weights("modelo.h5")
    except Exception as e:
        st.error(f"Erro ao carregar modelo salvo: {e}")
modelo.compile(optimizer='adam', loss='mean_squared_error')
modelo.fit(x, y, epochs=25)
modelo_json = modelo.to_json()
with open("modelo.json", "w") as arquivo_json:
    arquivo_json.write(modelo_json)
modelo.save_weights("modelo.h5")
inicio1 = dt.datetime(2010, 1, 1)
fim1 = dt.datetime.now()
valor = web.DataReader(acao, inicio1, fim1)
precoatual = valor['Close'].values
total = pd.concat((dados['Close'], valor['Close']), axis=0)
modelo_input = total[len(total)-len(valor)-dias:].values
modelo_input = modelo_input.reshape(-1, 1)
modelo_input = scaler.transform(modelo_input)
x1 = []
for a in range(dias, len(modelo_input)):
    x1.append(modelo_input[a-dias:a, 0])
x1 = np.array(x1)
x1 = np.reshape(x1, (x1.shape[0], x1.shape[1], 1))
preco = modelo.predict(x1)
preco = scaler.inverse_transform(preco)
dados_reais = [modelo_input[len(modelo_input)+1-dias:len(modelo_input+1), 0]]
dados_reais = np.array(dados_reais)
dados_reais = np.reshape(dados_reais, (dados_reais.shape[0], dados_reais.shape[1], 1))
previsao = modelo.predict(dados_reais)
previsao = scaler.inverse_transform(previsao)
fig = plt.figure()
plt.plot(precoatual, color="black", label=f"Preço atual da ação: {acao} ")
plt.plot(preco, color="green", label=f"Previsão de preço da ação: {acao}")
plt.title(f"Preços da ação {acao}")
plt.xlabel("Tempo")
plt.ylabel(f"Preços da ação {acao}")
plt.legend()
plt.show()
st.pyplot(fig)
st.write(f"Previsão: {previsao}")
