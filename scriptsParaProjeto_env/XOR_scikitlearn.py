#%% BIBLIOTECAS
from sklearn.neural_network import MLPClassifier

#%% CARGA DOS DADOS
X = [ [0,0], [0,1], [1,0], [1,1] ]
y = [0, 1, 1, 0]

#%% CONFIG REDE NEURAL
#mlp = MLPClassifier()

#mlp.fit(X,y)  #Executa treinamento - ver console -

mlp = MLPClassifier(verbose= True, 
                    max_iter= 10000, 
                    tol= 1e-6, # 0.000001
                    activation= 'relu')

#%% TREINAMENTO DA REDE
mlp.fit(X,y) # Executa treinamento - VER NO CONSOLE

#%% TESTE
print(mlp.predict([ [0,0] ]))
print(mlp.predict([ [0,1] ]))
print(mlp.predict([ [1,0] ]))
print(mlp.predict([ [1,1] ]))

#%% ALGUNS PARÂMETROS DA REDE
print("Classes = ", mlp.classes_) # Lista de classes detectadas durante o treinamento
print("Erro = ", mlp.loss_) # Fator de perda (erro) acumulado no modelo
print("Amostras visitadas = ", mlp.t_) # Número total de amostras processadas durante o treinamento
print("Atributos de entrada = ", mlp.n_features_in_) # Número de características de entrada esperadas pelo modelo
print("N ciclos = ", mlp.n_iter_) # Quantidade de iterações (épocas) realizadas no treinamento
print("N de camadas = ", mlp.n_layers_) # Número de camadas da rede
print("N de neuros saída = ", mlp.n_outputs_) # Número de neurônios na camada de saída
print("F de ativação = ", mlp.out_activation_) # Função de ativação usada na camada de saída
