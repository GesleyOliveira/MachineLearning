import pandas as pd
import random
from faker import Faker

fake = Faker('pt_BR')

# Valores possíveis
bairros = ['Pacaembu', 'Campolim', 'Centro', 'Vossoroca', 'Jd Primavera']
produtos = ['micro-ondas', 'TV', 'Fogão', 'freezer', 'ap de som', 'computador']
marcas = ['Brastemp', 'LG', 'Samsung', 'Electrolux', 'Gradiente']
estados = ['funciona', 'defeito']

# Função para gerar pontuação com base nas regras
def gerar_pontuacao(produto, estado):
    base = {
        'micro-ondas': 100,
        'TV': 150,
        'Fogão': 50,
        'freezer': 200,
        'ap de som': 40,
        'computador': 110
    }
    pontos = base.get(produto, 100)
    if estado == 'defeito':
        pontos = int(pontos * 0.6)
    return pontos

dados = []

for _ in range(1000):
    produto = random.choice(produtos)
    estado = random.choice(estados)
    pontuacao = gerar_pontuacao(produto, estado)
    dados.append({
        'idade': random.randint(18, 70),
        'bairro': random.choice(bairros),
        'produto': produto,
        'descricao': fake.word(),
        'marca': random.choice(marcas),
        'estado': estado,
        'pontuacao': pontuacao
    })

df = pd.DataFrame(dados)
df.to_csv('dados_trocas.csv', index=False)
print("Arquivo 'dados_trocas.csv' gerado com sucesso!")
