"""Etapa 4 — Estrutura do Perceptron de uma única camada (sem treino)."""

import random


class Perceptron:
    """Perceptron com um peso por característica, bias e ativação degrau (limiar 0.5)."""

    def __init__(
        self, n_entradas: int, taxa_aprendizagem: float = 0.1, semente: int = 42
    ) -> None:
        """Cria a rede com pesos e bias iniciais aleatórios entre -0.5 e 0.5."""
        self.n_entradas = n_entradas
        self.taxa_aprendizagem = taxa_aprendizagem
        # Gerador próprio: os sorteios desta rede não dependem do random global.
        self.gerador = random.Random(semente)
        # Ordem dos sorteios: primeiro os pesos (um por posição), depois o bias.
        self.pesos = [self.gerador.uniform(-0.5, 0.5) for _ in range(n_entradas)]
        self.bias = self.gerador.uniform(-0.5, 0.5)

    @staticmethod
    def funcao_ativacao(soma: float) -> int:
        """Função degrau do enunciado: 1 se soma > 0.5, senão 0."""
        return 1 if soma > 0.5 else 0

    def soma_ponderada(self, entradas: list[int]) -> float:
        """Calcula Σ(entradas[i] * pesos[i]) + bias."""
        if len(entradas) != len(self.pesos):
            raise ValueError(
                f"O vetor tem {len(entradas)} posições, "
                f"mas a rede espera {len(self.pesos)}."
            )
        soma = 0.0
        for x, w in zip(entradas, self.pesos):
            soma += x * w
        return soma + self.bias

    def prever(self, entradas: list[int]) -> int:
        """Devolve a classe prevista (0 ou 1) para um vetor de entradas."""
        return self.funcao_ativacao(self.soma_ponderada(entradas))

    # --- Etapa 5: treinamento ---

    def atualizar(self, entradas: list[int], erro: int) -> None:
        """Aplica peso = peso + taxa * erro * entrada a cada peso e ajusta o bias."""
        for i, x in enumerate(entradas):
            self.pesos[i] = self.pesos[i] + self.taxa_aprendizagem * erro * x
        # O bias é como um peso cuja entrada vale sempre 1.
        self.bias = self.bias + self.taxa_aprendizagem * erro

    def treinar(
        self, vetores: list[list[int]], classes: list[int], max_epocas: int = 100
    ) -> list[int]:
        """Treina a rede e devolve o número de erros de cada época."""
        if len(vetores) != len(classes):
            raise ValueError(
                f"Há {len(vetores)} vetores, mas {len(classes)} classes; "
                "os dois precisam ter o mesmo tamanho."
            )

        historico_erros: list[int] = []
        for _ in range(max_epocas):
            erros_na_epoca = 0
            # Ordem fixa: as avaliações na ordem em que vieram do CSV.
            for entradas, classe_real in zip(vetores, classes):
                classe_prevista = self.prever(entradas)
                erro = classe_real - classe_prevista
                if erro != 0:
                    self.atualizar(entradas, erro)
                    erros_na_epoca += 1
            historico_erros.append(erros_na_epoca)

            # Parada antecipada: uma época inteira sem erros.
            if erros_na_epoca == 0:
                break

        return historico_erros
