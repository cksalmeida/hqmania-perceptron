"""Etapa 6 — Teste da rede na base e cálculo da acurácia."""

from perceptron import Perceptron
from vetorizacao import vetorizar

NOMES_CLASSES = {0: "negativa", 1: "positiva"}


def testar_modelo(
    rede: Perceptron, avaliacoes: list[dict], vetores: list[list[int]]
) -> list[dict]:
    """Prevê a classe de cada avaliação e registra se a rede acertou."""
    resultados: list[dict] = []
    for avaliacao, vetor in zip(avaliacoes, vetores):
        classe_prevista = rede.prever(vetor)
        resultados.append(
            {
                "id": avaliacao["id"],
                "texto": avaliacao["texto"],
                "classe_real": avaliacao["classe"],
                "classe_prevista": classe_prevista,
                "acertou": classe_prevista == avaliacao["classe"],
            }
        )
    return resultados


def calcular_acuracia(resultados: list[dict]) -> dict:
    """Calcula total, corretas, incorretas, acurácia (%) e ids das avaliações erradas."""
    total = len(resultados)
    corretas = sum(1 for r in resultados if r["acertou"])
    incorretas = total - corretas
    # Fórmula do enunciado: (corretas / total) * 100
    acuracia = (corretas / total) * 100 if total > 0 else 0.0
    ids_incorretos = [r["id"] for r in resultados if not r["acertou"]]
    return {
        "total": total,
        "corretas": corretas,
        "incorretas": incorretas,
        "acuracia": acuracia,
        "ids_incorretos": ids_incorretos,
    }


def descrever_classe(classe: int) -> str:
    """Mostra a classe com o nome por extenso, ex.: '1 (positiva)'."""
    return f"{classe} ({NOMES_CLASSES[classe]})"


def imprimir_resultados(resultados: list[dict]) -> None:
    """Imprime, para cada avaliação: id, texto, classe real, classe prevista e resultado."""
    for r in resultados:
        resultado = "✔ ACERTOU" if r["acertou"] else "✘ ERROU"
        print(f"[id {r['id']}] {r['texto']}")
        print(
            f"    Real: {descrever_classe(r['classe_real'])} | "
            f"Prevista: {descrever_classe(r['classe_prevista'])} | {resultado}"
        )


def imprimir_acuracia(metricas: dict) -> None:
    """Imprime as métricas no formato do enunciado."""
    ids = metricas["ids_incorretos"]
    print(f"Total de avaliações: {metricas['total']}")
    print(f"Classificações corretas: {metricas['corretas']}")
    print(f"Classificações incorretas: {metricas['incorretas']}")
    print(f"Acurácia: {metricas['acuracia']:.2f}%")
    print(f"Avaliações incorretas: {', '.join(map(str, ids)) if ids else 'nenhuma'}")


# --- Etapa 7: avaliações novas ---


def detalhar_previsao(
    rede: Perceptron, palavras: list[str], vocabulario: list[str]
) -> dict:
    """Separa palavras conhecidas (com peso) e desconhecidas, e calcula soma e classe."""
    conhecidas: list[tuple[str, float]] = []
    desconhecidas: list[str] = []
    for palavra in palavras:
        if palavra in vocabulario:
            par = (palavra, rede.pesos[vocabulario.index(palavra)])
            if par not in conhecidas:  # vetor binário: palavra repetida conta 1 vez
                conhecidas.append(par)
        elif palavra not in desconhecidas:
            desconhecidas.append(palavra)

    # A soma e a classe vêm da própria rede, com o mesmo vetor BoW do treino.
    vetor = vetorizar(palavras, vocabulario)
    soma = rede.soma_ponderada(vetor)
    return {
        "conhecidas": conhecidas,
        "desconhecidas": desconhecidas,
        "bias": rede.bias,
        "soma": soma,
        "classe_prevista": rede.funcao_ativacao(soma),
    }


def imprimir_detalhes_novas(
    rede: Perceptron,
    avaliacoes: list[dict],
    listas_de_palavras: list[list[str]],
    vocabulario: list[str],
) -> None:
    """Imprime, para cada avaliação nova, a conta da rede e se ela acertou."""
    for avaliacao, palavras in zip(avaliacoes, listas_de_palavras):
        detalhes = detalhar_previsao(rede, palavras, vocabulario)
        acertou = detalhes["classe_prevista"] == avaliacao["classe"]
        resultado = "✔ ACERTOU" if acertou else "✘ ERROU"
        desconhecidas = detalhes["desconhecidas"]

        print(f"[id {avaliacao['id']}] {avaliacao['texto']}")
        print(f"    Palavras: {palavras}")
        print("    Conhecidas (peso):")
        if detalhes["conhecidas"]:
            for palavra, peso in detalhes["conhecidas"]:
                print(f"      {palavra}: {peso:.4f}")
        else:
            print("      Nenhuma palavra conhecida: soma = bias")
        print(
            "    Desconhecidas (ignoradas): "
            f"{', '.join(desconhecidas) if desconhecidas else 'nenhuma'}"
        )
        print(f"    Bias: {detalhes['bias']:.4f} | Soma: {detalhes['soma']:.4f}")
        print(
            f"    Esperada: {descrever_classe(avaliacao['classe'])} | "
            f"Prevista: {descrever_classe(detalhes['classe_prevista'])} | {resultado}"
        )
        print()
