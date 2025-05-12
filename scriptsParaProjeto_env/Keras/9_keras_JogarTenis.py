#%% BIBLIOTECAS
import pandas as pd     # dataframe - tab de bco de dados
from sklearn.preprocessing import OneHotEncoder # slide vol.5
from keras.models import Sequential  # tipo de RNA
from keras.layers import Input, Dense      # RNA total conectada
import numpy as np      # bib numérica

#%% CARGA DOS DADOS
df_jogar = pd.read_csv('jogarTenis.csv')  # CSV p/ tab BD
print('Tabela de dados:\n', df_jogar)
input('Aperte uma tecla para continuar: \n')

#%% SELEÇÃO DOS DADOS
# rotulos ou marcadores
dias = df_jogar['Dia']      # coluna Dia do BD
print("Rotulos:\n", dias)
input('Aperte uma tecla para continuar: \n')

# matriz de treinamento (registros com campos ou atributos)
X = df_jogar.loc[:, 'Aparencia':'Vento']   # de Aparência até Vento
print("Matriz de entradas (treinamento):\n", X)
input('Aperte uma tecla para continuar: \n')

# vetor de classes
y = df_jogar['Joga'].map({'sim': 1, 'nao': 0})
    ## converte 'sim' 'nao' em bit
print("Vetor de classes (treinamento):\n", y)
input('Aperte uma tecla para continuar: \n')

#%% ONE HOT ENCODER - pois os dados são nominais
encoder = OneHotEncoder(sparse_output=False)
X = encoder.fit_transform( df_jogar.loc[:, 'Aparencia':'Vento'] )
    ## ver slide vol.5
print("Matriz de entradas codificadas:\n", X)
input('Aperte uma tecla para continuar: \n')

X = X.astype('float32')  # converte int para float32

#%% CONFIG REDE NEURAL
model = Sequential()    # tipo de RNA
model.add(Input(shape=(X.shape[1],)))
model.add(Dense(64, activation='relu'))
    ## camada oculta com 64 neurons e 10 var entrada (X.shape[1],)
model.add(Dense(1, activation='sigmoid'))  # 1 saída: sim ou não
    ## função de ativação da camada de saída 'sigmoid'
    ## mais adequada para problemas de classificação binária

#%% COMPILAR O MODELO
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    ## função de perda: binary_crossentropy, 
    ## mais adequada para problemas de classificação binária.

#%% TREINAMENTO DA REDE
model.fit(X, y, epochs=2000, verbose=1)

#%% testes
print('\n')
X_inverso = encoder.inverse_transform(X)    # descodifica X
for caso, inverso in zip(X, X_inverso):
    prev = model.predict(np.array([caso]))
    print('caso: ', caso, ' ', inverso, ' previsto: ', prev)

## A função zip combina os dois arrays em um único iterador, 
## onde cada elemento é uma tupla contendo um elemento de cada array.

input('Aperte uma tecla para continuar: \n')

#%% teste de dado "não visto:"
X_novo = ['nublado','fria','alta','fraco']
X_novo_codificado = encoder.transform([X_novo])
    ## Cuidado: diferente de 'encoder.fit_transform'
print("\nNovo caso codificado: ", X_novo_codificado)

# previsão
print( X_novo, '=', model.predict(X_novo_codificado))
print("\n")
