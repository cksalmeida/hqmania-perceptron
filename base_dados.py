"""Etapa 1 — Carregamento da base de avaliações da HQMania."""

import csv


def carregar_avaliacoes(caminho: str) -> list[dict]:
    """Lê o CSV de avaliações e devolve uma lista de dicionários {id, texto, classe}."""
    avaliacoes: list[dict] = []
    ids_vistos: set[int] = set()

    with open(caminho, encoding="utf-8", newline="") as arquivo:
        leitor = csv.DictReader(arquivo)

        for linha in leitor:
            # O CSV guarda tudo como texto; convertemos id e classe para inteiro.
            id_avaliacao = int(linha["id"])
            texto = linha["texto"].strip()
            classe = int(linha["classe"])

            if id_avaliacao in ids_vistos:
                raise ValueError(f"O id {id_avaliacao} aparece mais de uma vez na base.")
            if not texto:
                raise ValueError(f"A avaliação de id {id_avaliacao} está com o texto vazio.")
            if classe not in (0, 1):
                raise ValueError(
                    f"A avaliação de id {id_avaliacao} tem classe {classe}; "
                    "a classe deve ser 0 (negativa) ou 1 (positiva)."
                )

            ids_vistos.add(id_avaliacao)
            avaliacoes.append({"id": id_avaliacao, "texto": texto, "classe": classe})

    return avaliacoes
