import torch
import torch.nn as nn


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def get_activation(name: str) -> nn.Module:
    """
    Retorna a função de ativação solicitada.
    """

    activations = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }

    if name not in activations:
        raise ValueError(
            f"Ativação inválida: {name}. "
            f"Opções disponíveis: {list(activations.keys())}"
        )

    return activations[name]()


def build_hidden_layers(
    input_dim: int,
    hidden_dims: tuple[int, ...] | list[int],
    activation: str,
    dropout: float,
    batch_norm: bool,
) -> tuple[nn.Sequential, int]:
    """
    Constrói as camadas ocultas de uma MLP.

    Cada camada oculta pode conter:
        1. Camada linear;
        2. Batch Normalization (opcional);
        3. Função de ativação;
        4. Dropout (opcional).
    """

    if input_dim <= 0:
        raise ValueError("input_dim precisa ser positivo.")

    if not hidden_dims:
        raise ValueError("A rede precisa possuir ao menos uma camada oculta.")

    if any(neurons <= 0 for neurons in hidden_dims):
        raise ValueError("Todas as camadas precisam ter neurônios.")

    if not 0 <= dropout < 1:
        raise ValueError("dropout deve estar entre 0 e 1.")

    layers = []
    previous_dim = input_dim

    for neurons in hidden_dims:

        layers.append(
            nn.Linear(previous_dim, neurons)
        )

        if batch_norm:
            layers.append(
                nn.BatchNorm1d(neurons)
            )

        layers.append(
            get_activation(activation)
        )

        if dropout > 0:
            layers.append(
                nn.Dropout(dropout)
            )

        previous_dim = neurons

    return nn.Sequential(*layers), previous_dim


# ============================================================
# REDE NEURAL PARA CLASSIFICAÇÃO
# ============================================================

class MLPClassifier(nn.Module):
    """
    MLP para classificação binária.

    Saída:
        Um logit por exemplo.

    A função sigmoid não é aplicada aqui.
    Utilizamos BCEWithLogitsLoss durante o treinamento.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.2,
        batch_norm: bool = False,
    ):

        super().__init__()

        self.hidden_layers, last_dim = build_hidden_layers(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.output_layer = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.hidden_layers(x)

        logits = self.output_layer(x)

        return logits


# ============================================================
# REDE NEURAL PARA REGRESSÃO
# ============================================================

class MLPRegressor(nn.Module):
    """
    MLP para regressão.

    Saída:
        Um valor contínuo por exemplo.

    Não utiliza função de ativação na camada de saída,
    permitindo prever valores contínuos.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        activation: str = "relu",
        dropout: float = 0.2,
        batch_norm: bool = False,
    ):

        super().__init__()

        self.hidden_layers, last_dim = build_hidden_layers(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        self.output_layer = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.hidden_layers(x)

        predictions = self.output_layer(x)

        return predictions


# ============================================================
# TESTE DAS ARQUITETURAS
# ============================================================

if __name__ == "__main__":

    torch.manual_seed(42)

    input_dim = 163
    batch_size = 256

    # Simular um batch de entrada.
    X = torch.randn(batch_size, input_dim)

    # --------------------------------------------------------
    # Classificação
    # --------------------------------------------------------

    classifier = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=(128, 64),
        activation="relu",
        dropout=0.2,
        batch_norm=False,
    )

    classification_output = classifier(X)

    print("=" * 60)
    print("REDE DE CLASSIFICAÇÃO")
    print("=" * 60)

    print("Formato da entrada:", tuple(X.shape))
    print("Formato da saída:", tuple(classification_output.shape))
    print("Parâmetros treináveis:", sum(
        p.numel() for p in classifier.parameters()
        if p.requires_grad
    ))

    print("Função de perda: BCEWithLogitsLoss")

    # --------------------------------------------------------
    # Regressão
    # --------------------------------------------------------

    regressor = MLPRegressor(
        input_dim=input_dim,
        hidden_dims=(128, 64),
        activation="relu",
        dropout=0.2,
        batch_norm=False,
    )

    regression_output = regressor(X)

    print("\n" + "=" * 60)
    print("REDE DE REGRESSÃO")
    print("=" * 60)

    print("Formato da entrada:", tuple(X.shape))
    print("Formato da saída:", tuple(regression_output.shape))
    print("Parâmetros treináveis:", sum(
        p.numel() for p in regressor.parameters()
        if p.requires_grad
    ))

    print("Função de perda: MSELoss")