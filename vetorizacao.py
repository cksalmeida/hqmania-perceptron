"""Etapa 3 — Vocabulário e Bag of Words binário, implementados manualmente."""


def criar_vocabulario(listas_de_palavras: list[list[str]]) -> list[str]:
    """Junta todas as palavras distintas e devolve uma lista em ordem alfabética."""
    palavras_distintas: set[str] = set()
    for palavras in listas_de_palavras:
        palavras_distintas.update(palavras)
    # Ordenamos para que cada palavra tenha sempre a mesma posição no vetor
    # (um set não garante ordem). sorted() usa o código Unicode, então
    # palavras com letra acentuada no início (ex.: "ótima") vão para o fim.
    return sorted(palavras_distintas)


def vetorizar(palavras: list[str], vocabulario: list[str]) -> list[int]:
    """Cria o vetor binário: 1 se o termo do vocabulário está nas palavras, 0 se não."""
    presentes = set(palavras)
    return [1 if termo in presentes else 0 for termo in vocabulario]


def vetorizar_lista(
    listas_de_palavras: list[list[str]], vocabulario: list[str]
) -> list[list[int]]:
    """Aplica vetorizar a cada lista de palavras."""
    return [vetorizar(palavras, vocabulario) for palavras in listas_de_palavras]
