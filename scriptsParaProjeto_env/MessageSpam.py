import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import kagglehub
import os

# Baixar o dataset diretamente do Kaggle usando kagglehub
path = kagglehub.dataset_download("purusinghvi/email-spam-classification-dataset")
print("Path to dataset files:", path)

# Verificar qual é o nome do arquivo CSV dentro da pasta baixada
csv_filename = None
for file in os.listdir(path):
    if file.endswith(".csv"):
        csv_filename = file
        break

if not csv_filename:
    raise FileNotFoundError("Nenhum arquivo .csv encontrado no diretório do dataset.")

csv_path = os.path.join(path, csv_filename)

# Carregar os dados
df = pd.read_csv(csv_path)
print(df.head())

# Separar os dados em recursos e rótulos
x = df['text']
y = df['label']

# Dividir os dados em conjunto de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Vetorizar os textos
vectorizer = TfidfVectorizer()
x_trained_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)

# Treinar o modelo SVM
model = SVC()
model.fit(x_trained_vectorized, y_train)

# Avaliar o modelo
predictions = model.predict(x_test_vectorized)
print('Accuracy: ', accuracy_score(y_test, predictions)) 
print('Precision: ', precision_score(y_test, predictions))
print('Recall: ', recall_score(y_test, predictions))

# Testar com um exemplo personalizado
new_email = 'Dear friend, I have a great investment opportunity for you!'
new_email_vectorized = vectorizer.transform([new_email])
prediction = model.predict(new_email_vectorized)
print('Prediction:', prediction)
