"""Etapa 2 — Pré-processamento dos textos das avaliações.

Ordem do enunciado: minúsculas → remover pontuação → separar palavras → remover stopwords.
"""

import string

import nltk
from nltk.corpus import stopwords as corpus_stopwords

# Palavras de negação que NÃO devem ser removidas, mesmo estando na lista do NLTK.
# Removê-las apagaria a informação que inverte o sentido da frase
# (ex.: "personagens sem graça" viraria só "personagens graça").
PALAVRAS_MANTIDAS = {"não", "sem"}

# Tabela que troca cada sinal de pontuação por um espaço.
TABELA_PONTUACAO = str.maketrans(string.punctuation, " " * len(string.punctuation))


def carregar_stopwords() -> set[str]:
    """Devolve as stopwords do NLTK em português, menos as PALAVRAS_MANTIDAS."""
    try:
        lista = corpus_stopwords.words("portuguese")
    except LookupError:
        # O corpus ainda não foi baixado neste computador: baixa e tenta de novo.
        nltk.download("stopwords", quiet=True)
        lista = corpus_stopwords.words("portuguese")
    return set(lista) - PALAVRAS_MANTIDAS


def para_minusculas(texto: str) -> str:
    """Passo 1: converte o texto para letras minúsculas."""
    return texto.lower()


def remover_pontuacao(texto: str) -> str:
    """Passo 2: troca cada sinal de pontuação por espaço (os acentos são mantidos)."""
    return texto.translate(TABELA_PONTUACAO)


def separar_palavras(texto: str) -> list[str]:
    """Passo 3: separa o texto em uma lista de palavras."""
    return texto.split()


def remover_stopwords(palavras: list[str], stopwords: set[str]) -> list[str]:
    """Passo 4: remove as stopwords, mantendo a ordem das palavras restantes."""
    return [palavra for palavra in palavras if palavra not in stopwords]


def preprocessar(texto: str, stopwords: set[str]) -> list[str]:
    """Aplica os 4 passos, na ordem do enunciado, e devolve a lista final de palavras."""
    texto = para_minusculas(texto)
    texto = remover_pontuacao(texto)
    palavras = separar_palavras(texto)
    return remover_stopwords(palavras, stopwords)
