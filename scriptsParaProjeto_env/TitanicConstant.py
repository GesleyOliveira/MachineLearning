import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Carregar os dados
data = pd.read_csv('./archive/tested.csv') 

# Pré-processamento
data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Survived']]
data.dropna(inplace=True)

# Manipular rótulos para balancear artificialmente
# Mudar algumas mulheres sobreviventes (Survived = 1) para mortas (0)
mulheres_vivas = data[(data['Sex'] == 'female') & (data['Survived'] == 1)]
idx_matar_mulheres = mulheres_vivas.sample(frac=0.2, random_state=42).index
data.loc[idx_matar_mulheres, 'Survived'] = 0

# Mudar alguns homens mortos (Survived = 0) para vivos (1)
homens_mortos = data[(data['Sex'] == 'male') & (data['Survived'] == 0)]
idx_salvar_homens = homens_mortos.sample(frac=0.2, random_state=42).index
data.loc[idx_salvar_homens, 'Survived'] = 1

# Codificar variáveis categóricas
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])

# Separar features e alvo
X = data.drop('Survived', axis=1)
y = data['Survived']

# Dividir em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configurar e treinar o modelo
mlp = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=1000, tol=1e-6, learning_rate='adaptive', activation='relu', verbose=True, alpha=0.01, random_state=42)
mlp.fit(X_train, y_train)

# Avaliar o modelo
y_pred = mlp.predict(X_test)
print(classification_report(y_test, y_pred))

# Parâmetros da rede
print("Classes = ", mlp.classes_)
print("Erro (Loss) = ", mlp.loss_)
print("Amostras visitadas = ", mlp.t_)
print("Atributos de entrada = ", mlp.n_features_in_)
print("N ciclos = ", mlp.n_iter_)
print("N de camadas = ", mlp.n_layers_)
print("Tamanhos das camadas ocultas = ", mlp.hidden_layer_sizes)
print("N de neurônios na saída = ", mlp.n_outputs_)
print("Função de ativação de saída = ", mlp.out_activation_)
