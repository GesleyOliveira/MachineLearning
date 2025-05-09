#%% BIBLIOTECAS
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import kagglehub
import os

#%% CARGA DOS DADOS COM KAGGLEHUB
path = kagglehub.dataset_download("purusinghvi/email-spam-classification-dataset")
print("Path to dataset files:", path)

csv_filename = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_filename = file
        break

if not csv_filename:
    raise FileNotFoundError("Nenhum arquivo .csv encontrado.")

csv_path = os.path.join(path, csv_filename)
df = pd.read_csv(csv_path)

#%% PRÉ-PROCESSAMENTO
x = df['text']
y = df['label']

# Separação treino/teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vetorização dos textos
vectorizer = TfidfVectorizer()
x_train_vect = vectorizer.fit_transform(x_train)
x_test_vect = vectorizer.transform(x_test)

#%% CONFIGURAÇÃO DA REDE NEURAL
mlp = MLPClassifier(verbose=True, 
                    max_iter=1000, 
                    tol=1e-4, 
                    activation='relu', 
                    hidden_layer_sizes=(100,))

#%% TREINAMENTO DA REDE
mlp.fit(x_train_vect, y_train)

#%% TESTE
predictions = mlp.predict(x_test_vect)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall:", recall_score(y_test, predictions))

#%% EXEMPLO CUSTOMIZADO
new_email = ['Congratulations! You have won a $1,000 Walmart gift card. Click here to claim now.']
new_email_vect = vectorizer.transform(new_email)
print("Prediction:", mlp.predict(new_email_vect))

#%% PARÂMETROS DA REDE
print("Classes = ", mlp.classes_)
print("Erro = ", mlp.loss_)
print("Amostras visitadas = ", mlp.t_)
print("Atributos de entrada = ", mlp.n_features_in_)
print("N ciclos = ", mlp.n_iter_)
print("N de camadas = ", mlp.n_layers_)
print("N de neurônios saída = ", mlp.n_outputs_)
print("F de ativação = ", mlp.out_activation_)
