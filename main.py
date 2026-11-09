"""Ponto de entrada do projeto: orquestra o fluxo e imprime os resultados."""

from avaliacao import (
    calcular_acuracia,
    imprimir_acuracia,
    imprimir_detalhes_novas,
    imprimir_resultados,
    testar_modelo,
)
from base_dados import carregar_avaliacoes
from preprocessamento import (
    PALAVRAS_MANTIDAS,
    carregar_stopwords,
    para_minusculas,
    preprocessar,
    remover_pontuacao,
    remover_stopwords,
    separar_palavras,
)
from perceptron import Perceptron
from vetorizacao import criar_vocabulario, vetorizar, vetorizar_lista

CAMINHO_BASE = "dados/avaliacoes.csv"
CAMINHO_NOVAS = "dados/novas_avaliacoes.csv"
IDS_PASSO_A_PASSO = [2, 14]
IDS_VETORES = [1, 13, 15, 19]
IDS_CONTA_DETALHADA = [1, 13]
MAX_EPOCAS = 100
PALAVRAS_PESOS_FINAIS = [
    "excelentes", "incrível", "fraca", "história", "desenhos",
    "não", "sem", "recomendo", "maravilha", "chata",
]

# Exemplo do enunciado (p. 2), para conferir a vetorização.
VOCABULARIO_EXEMPLO = ["história", "incrível", "excelente", "fraca"]
FRASES_EXEMPLO = ["A história é incrível", "A história é fraca"]


def mostrar_passo_a_passo(avaliacao: dict, stopwords: set[str]) -> None:
    """Imprime o resultado de cada um dos 4 passos do pré-processamento."""
    original = avaliacao["texto"]
    minusculas = para_minusculas(original)
    sem_pontuacao = remover_pontuacao(minusculas)
    palavras = separar_palavras(sem_pontuacao)
    sem_stopwords = remover_stopwords(palavras, stopwords)

    print(f"[id {avaliacao['id']}] Original:".ljust(25) + original)
    print("  1. Minúsculas:".ljust(25) + minusculas)
    print("  2. Sem pontuação:".ljust(25) + sem_pontuacao)
    print("  3. Palavras:".ljust(25) + str(palavras))
    print("  4. Sem stopwords:".ljust(25) + str(sem_stopwords))
    print()


def mostrar_exemplo_enunciado(stopwords: set[str]) -> None:
    """Vetoriza as frases do enunciado com o vocabulário de exemplo."""
    print(f"Vocabulário de exemplo: {VOCABULARIO_EXEMPLO}")
    for frase in FRASES_EXEMPLO:
        palavras = preprocessar(frase, stopwords)
        vetor = vetorizar(palavras, VOCABULARIO_EXEMPLO)
        print(f"  {frase!r:26} -> {palavras} -> {vetor}")
    print()


def mostrar_vocabulario(vocabulario: list[str]) -> None:
    """Imprime o tamanho do vocabulário e cada palavra com sua posição."""
    print(f"Tamanho do vocabulário: {len(vocabulario)}")
    for posicao, palavra in enumerate(vocabulario):
        print(f"  [{posicao}] {palavra}")
    print()


def mostrar_vetor(avaliacao: dict, vetor: list[int], vocabulario: list[str]) -> None:
    """Resume um vetor: tamanho, quantos 1 e em quais posições (posição:palavra)."""
    posicoes_com_1 = [
        f"{posicao}:{vocabulario[posicao]}"
        for posicao, valor in enumerate(vetor)
        if valor == 1
    ]
    print(f"[id {avaliacao['id']}] {avaliacao['texto']}")
    print(f"  Tamanho do vetor: {len(vetor)} | Posições com 1: {len(posicoes_com_1)}")
    print(f"  {', '.join(posicoes_com_1)}")
    print()


def mostrar_conta_detalhada(
    avaliacao: dict, vetor: list[int], vocabulario: list[str], rede: Perceptron
) -> None:
    """Mostra a soma ponderada termo a termo, a ativação e se a previsão acertou."""
    print(f"[id {avaliacao['id']}] {avaliacao['texto']}")
    for posicao, valor in enumerate(vetor):
        if valor == 1:
            print(f"    {vocabulario[posicao]}: {rede.pesos[posicao]:.4f}")
    print(f"  + bias: {rede.bias:.4f}")

    soma = rede.soma_ponderada(vetor)
    prevista = rede.funcao_ativacao(soma)
    resultado = "ACERTOU" if prevista == avaliacao["classe"] else "ERROU"
    print(f"  = soma: {soma:.4f} -> ativação (soma > 0.5?): {prevista}")
    print(f"  Classe real: {avaliacao['classe']} -> {resultado}")
    print()


def mostrar_primeira_correcao(
    avaliacao: dict, vetor: list[int], vocabulario: list[str]
) -> None:
    """Simula a 1ª correção numa rede separada (mesma semente), sem tocar na rede real."""
    rede_demo = Perceptron(n_entradas=len(vocabulario))
    prevista = rede_demo.prever(vetor)
    erro = avaliacao["classe"] - prevista
    print(f"[id {avaliacao['id']}] real {avaliacao['classe']}, prevista {prevista}")
    print(f"  erro = {avaliacao['classe']} - {prevista} = {erro:+d}")

    pesos_antes = list(rede_demo.pesos)
    bias_antes = rede_demo.bias
    rede_demo.atualizar(vetor, erro)

    for posicao, valor in enumerate(vetor):
        if valor == 1:
            print(
                f"    {vocabulario[posicao]}: "
                f"{pesos_antes[posicao]:.4f} -> {rede_demo.pesos[posicao]:.4f}"
            )
    print(f"    bias: {bias_antes:.4f} -> {rede_demo.bias:.4f}")
    print()


def imprimir_secao(titulo: str) -> None:
    """Imprime o título de uma seção grande da saída."""
    print("=" * 70)
    print(titulo)
    print("=" * 70)
    print()


def imprimir_subtitulo(titulo: str) -> None:
    """Imprime o título de uma parte dentro de uma seção."""
    print(f"--- {titulo} ---")


def main() -> None:
    """Executa o fluxo completo: base → pré-processamento → vetorização → Perceptron."""
    imprimir_secao("1. BASE DE DADOS")
    avaliacoes = carregar_avaliacoes(CAMINHO_BASE)
    total = len(avaliacoes)
    positivas = sum(1 for a in avaliacoes if a["classe"] == 1)
    negativas = total - positivas
    print(f"Total: {total} | Positivas: {positivas} | Negativas: {negativas}")
    print()

    imprimir_secao("2. PRÉ-PROCESSAMENTO")
    stopwords = carregar_stopwords()
    mantidas = " e ".join(f'"{p}"' for p in sorted(PALAVRAS_MANTIDAS))
    print(f"Stopwords usadas: {len(stopwords)} (do NLTK, mantendo {mantidas} nos textos)")
    print()

    imprimir_subtitulo("Passo a passo")
    for a in avaliacoes:
        if a["id"] in IDS_PASSO_A_PASSO:
            mostrar_passo_a_passo(a, stopwords)

    imprimir_subtitulo("Avaliações pré-processadas")
    listas_de_palavras = [preprocessar(a["texto"], stopwords) for a in avaliacoes]
    for a, palavras in zip(avaliacoes, listas_de_palavras):
        print(f"[{a['id']}] ({a['classe']}) {palavras}")
    print()

    imprimir_secao("3. VETORIZAÇÃO (BAG OF WORDS)")
    imprimir_subtitulo("Exemplo do enunciado")
    mostrar_exemplo_enunciado(stopwords)

    imprimir_subtitulo("Vocabulário da base")
    vocabulario = criar_vocabulario(listas_de_palavras)
    mostrar_vocabulario(vocabulario)

    imprimir_subtitulo("Vetores Bag of Words")
    vetores = vetorizar_lista(listas_de_palavras, vocabulario)
    for a, vetor in zip(avaliacoes, vetores):
        if a["id"] in IDS_VETORES:
            mostrar_vetor(a, vetor, vocabulario)

    if any(len(vetor) != len(vocabulario) for vetor in vetores):
        raise ValueError("Há vetores com tamanho diferente do vocabulário.")
    print(f"Verificação: os {len(vetores)} vetores têm tamanho {len(vocabulario)}. OK")
    print()

    imprimir_secao("4. PERCEPTRON (ANTES DO TREINO)")
    rede = Perceptron(n_entradas=len(vocabulario))
    primeiros = ", ".join(f"{p:.4f}" for p in rede.pesos[:5])
    print(f"Número de pesos: {len(rede.pesos)}")
    print(f"Taxa de aprendizagem: {rede.taxa_aprendizagem}")
    print(f"5 primeiros pesos: {primeiros}")
    print(f"Bias: {rede.bias:.4f}")
    print()

    imprimir_subtitulo("Conta detalhada antes do treino")
    for a, vetor in zip(avaliacoes, vetores):
        if a["id"] in IDS_CONTA_DETALHADA:
            mostrar_conta_detalhada(a, vetor, vocabulario, rede)

    # Linha de base: a mesma rede, testada ANTES de treinar (pesos sorteados).
    metricas_antes = calcular_acuracia(testar_modelo(rede, avaliacoes, vetores))
    print(
        f"Linha de base (pesos iniciais): "
        f"{metricas_antes['corretas']} / {metricas_antes['total']} acertos"
    )
    print()

    imprimir_subtitulo("Validação do tamanho do vetor")
    try:
        rede.prever([1, 1, 0, 0])
    except ValueError as erro:
        print(f"ValueError: {erro}")
    print()

    imprimir_secao("5. TREINAMENTO")
    imprimir_subtitulo("Primeira correção em detalhe (rede de demonstração)")
    mostrar_primeira_correcao(avaliacoes[0], vetores[0], vocabulario)

    imprimir_subtitulo("Épocas")
    classes = [a["classe"] for a in avaliacoes]
    historico = rede.treinar(vetores, classes, max_epocas=MAX_EPOCAS)
    for numero, erros in enumerate(historico, start=1):
        print(f"Época {numero}: {erros} erros")
    if historico[-1] == 0:
        print(f"Treino parou na época {len(historico)} (parada antecipada: época sem erros).")
    else:
        print(f"Treino usou o máximo de {MAX_EPOCAS} épocas sem zerar os erros.")
    print()

    imprimir_subtitulo("Pesos finais selecionados")
    for palavra in PALAVRAS_PESOS_FINAIS:
        print(f"  {palavra}: {rede.pesos[vocabulario.index(palavra)]:.4f}")
    print(f"  bias: {rede.bias:.4f}")
    print()

    imprimir_secao("6. TESTE NA BASE E ACURÁCIA")
    resultados = testar_modelo(rede, avaliacoes, vetores)
    imprimir_subtitulo("Resultado por avaliação")
    imprimir_resultados(resultados)
    print()

    imprimir_subtitulo("Acurácia")
    metricas_depois = calcular_acuracia(resultados)
    imprimir_acuracia(metricas_depois)
    print()

    print(
        f"Antes do treino: {metricas_antes['corretas']}/{metricas_antes['total']} "
        f"({metricas_antes['acuracia']:.2f}%) → "
        f"Depois do treino: {metricas_depois['corretas']}/{metricas_depois['total']} "
        f"({metricas_depois['acuracia']:.2f}%)"
    )
    print()

    imprimir_secao("7. TESTE COM AVALIAÇÕES NOVAS")
    # Mesma rede treinada, mesmas stopwords e mesmo vocabulário: nada é retreinado.
    novas = carregar_avaliacoes(CAMINHO_NOVAS)
    palavras_novas = [preprocessar(a["texto"], stopwords) for a in novas]
    vetores_novos = vetorizar_lista(palavras_novas, vocabulario)

    imprimir_subtitulo("Detalhes por avaliação")
    imprimir_detalhes_novas(rede, novas, palavras_novas, vocabulario)

    imprimir_subtitulo("Acurácia nas avaliações novas")
    metricas_novas = calcular_acuracia(testar_modelo(rede, novas, vetores_novos))
    imprimir_acuracia(metricas_novas)
    print()

    print(
        f"Base de treino: {metricas_depois['corretas']}/{metricas_depois['total']} "
        f"({metricas_depois['acuracia']:.2f}%) | "
        f"Avaliações novas: {metricas_novas['corretas']}/{metricas_novas['total']} "
        f"({metricas_novas['acuracia']:.2f}%)"
    )


if __name__ == "__main__":
    main()
