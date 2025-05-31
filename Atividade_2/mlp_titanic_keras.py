import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Carregar dados do Titanic
data = pd.read_csv('tested.csv')


# Selecionar colunas relevantes
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
data.dropna(inplace=True)

# Codificar variáveis categóricas
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Separar X e y
X = data.drop('Survived', axis=1)
y = data['Survived']

# Escalonar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Configurações para testar
configs = [
    {"name": "A - (10,5) - relu", "layers": [10, 5], "activation": "relu"},
    {"name": "B - (32,16) - relu", "layers": [32, 16], "activation": "relu"},
    {"name": "C - (10,5) - sigmoid", "layers": [10, 5], "activation": "sigmoid"},
    {"name": "D - (32,16) - sigmoid", "layers": [32, 16], "activation": "sigmoid"},
]

# Loop para treinar e avaliar cada configuração
for cfg in configs:
    print("\n" + "="*60)
    print(f"🔧 Testando configuração: {cfg['name']}")

    # Criar modelo sequencial
    model = Sequential()
    model.add(Dense(cfg['layers'][0], activation=cfg['activation'], input_shape=(X_train.shape[1],)))
    model.add(Dense(cfg['layers'][1], activation=cfg['activation']))
    model.add(Dense(1, activation='sigmoid'))  # Saída binária

    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # pode ajustar para testes com learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Early stopping para evitar overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    # Treinar modelo
    history = model.fit(
        X_train, y_train,
        epochs=100,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0
    )

    # Prever e avaliar
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    acc = model.evaluate(X_test, y_test, verbose=0)[1]

    # Exibir resultados
    print(f"Acurácia final: {acc:.4f}")
    print("\n📊 Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\n📋 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    print("\n" + "="*60)
    print("🔁 Testando variações com Scikit-Learn (MLPClassifier)")

    mlp_configs = [
    {"name": "SKL - (10,5) - relu", "hidden_layer_sizes": (10, 5), "activation": "relu", "max_iter": 300},
    {"name": "SKL - (32,16) - relu", "hidden_layer_sizes": (32, 16), "activation": "relu", "max_iter": 300},
    {"name": "SKL - (10,5) - logistic", "hidden_layer_sizes": (10, 5), "activation": "logistic", "max_iter": 300},
    {"name": "SKL - (32,16) - logistic", "hidden_layer_sizes": (32, 16), "activation": "logistic", "max_iter": 300}
]

for cfg in mlp_configs:
    print("\n" + "-"*50)
    print(f"⚙️  {cfg['name']}")
    mlp = MLPClassifier(hidden_layer_sizes=cfg["hidden_layer_sizes"],
                        activation=cfg["activation"],
                        max_iter=cfg["max_iter"],
                        random_state=42)
    
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    
    acc = mlp.score(X_test, y_test)
    print(f"Acurácia: {acc:.4f}")
    print("📊 Matriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("📋 Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

