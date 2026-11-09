# HQMania — Classificador de avaliações com Perceptron

Atividade da disciplina de **Processamento de Linguagem Natural** (Fatec Registro).

Estudo de caso da loja de quadrinhos **HQMania**: classificar avaliações de clientes como
**positivas (1)** ou **negativas (0)** usando uma **rede neural Perceptron de uma única camada**,
implementada do zero em Python, sem bibliotecas de Machine Learning.

## Fluxo

```
Avaliações → Pré-processamento → Vetorização → Perceptron → Classificação
```

1. **Base de dados**: 24 avaliações (12 positivas, 12 negativas) em `dados/avaliacoes.csv`.
2. **Pré-processamento**: minúsculas → remover pontuação → separar palavras → remover stopwords.
   As stopwords vêm do NLTK (português), **mantendo "não" e "sem"**, que invertem o sentido da frase.
3. **Vetorização**: Bag of Words **binário**, implementado manualmente (sem `CountVectorizer`).
   O vocabulário tem 103 palavras, montado só a partir da base de treino.
4. **Perceptron**: um peso por palavra + bias, pesos iniciais aleatórios em `[-0.5, 0.5)`
   (semente 42), ativação `f(soma) = 1 se soma > 0.5, senão 0`.
5. **Treinamento**: regra `peso = peso + taxa * erro * entrada` (taxa 0.1), no máximo 100 épocas,
   com parada antecipada quando uma época termina sem erros.
6. **Teste e acurácia**: `acurácia = (corretas / total) * 100`.
7. **Avaliações novas**: 8 frases inéditas em `dados/novas_avaliacoes.csv`, testadas com a
   mesma rede e o mesmo vocabulário (palavras desconhecidas são ignoradas).

## Estrutura

```
hqmania-perceptron/
├── dados/
│   ├── avaliacoes.csv          # base de treino (24 avaliações)
│   └── novas_avaliacoes.csv    # avaliações inéditas (8)
├── base_dados.py               # carrega e valida o CSV
├── preprocessamento.py         # limpeza do texto e stopwords
├── vetorizacao.py              # vocabulário e Bag of Words
├── perceptron.py               # modelo, ativação e treino
├── avaliacao.py                # teste, acurácia e detalhamento das previsões
└── main.py                     # executa o fluxo completo e imprime os resultados
```

## Como rodar

Pré-requisitos: **Python 3.12+** e o gerenciador [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/cksalmeida/hqmania-perceptron.git
cd hqmania-perceptron
uv run main.py
```

O `uv run` cria o ambiente virtual e instala a única dependência (`nltk`) automaticamente.
Na primeira execução, a lista de stopwords do NLTK é baixada sozinha (precisa de internet).

## Resultados

| Teste | Acertos | Acurácia |
|---|---|---|
| Antes do treino (pesos aleatórios) | 11 / 24 | 45.83% |
| Base de treino, depois do treino (4 épocas) | 24 / 24 | 100.00% |
| Avaliações novas | 4 / 8 | 50.00% |

A queda de 100% para 50% mostra que testar com a própria base de treino dá uma acurácia otimista.
As principais limitações observadas são palavras fora do vocabulário (ignoradas),
variações de plural e gênero ("fraca" × "fraco"), ironia, e o bias final (0.537 > 0.5),
que faz uma avaliação sem palavras conhecidas ser classificada como positiva.

## Restrições da atividade

- Sem scikit-learn, TensorFlow, Keras, PyTorch ou ferramentas prontas de Bag of Words.
- NLTK usado **somente** para obter a lista de stopwords.
- Resultados reproduzíveis: semente fixa (42) e ordem fixa das avaliações no treino.
