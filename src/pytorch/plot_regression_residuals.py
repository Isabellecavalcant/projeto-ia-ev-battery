from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.pytorch.preprocessing import prepare_data
from src.pytorch.dataset import create_dataloaders
from src.pytorch.models import MLPRegressor
from src.pytorch.evaluate import predict, regression_metrics


ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = ROOT / "models/regression/final_R1_seed_42.pt"
DEVICE = torch.device("cpu")
BATCH_SIZE = 256


def main():
    print("=" * 65)
    print("ANÁLISE DE RESÍDUOS — REGRESSÃO R1")
    print("=" * 65)

    # 1. Reproduzir a preparação e a divisão originais dos dados.
    data = prepare_data("regression")

    if data.target_scaler is None:
        raise RuntimeError("O target_scaler não foi encontrado.")

    loaders = create_dataloaders(
        data,
        batch_size=BATCH_SIZE,
        seed=42,
    )

    # 2. Reconstruir a arquitetura do checkpoint final.
    model = MLPRegressor(
        input_dim=data.X_train.shape[1],
        hidden_dims=[32, 32],
        activation="relu",
        dropout=0.0,
        batch_norm=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint["task"] != "regression" or checkpoint["seed"] != 42:
        raise ValueError("O checkpoint não corresponde ao modelo R1 seed 42.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print(f"\nCheckpoint: {CHECKPOINT_PATH.name}")
    print(f"Melhor época: {checkpoint['epoch']}")
    print(f"Melhor val_loss: {checkpoint['best_val_loss']:.8f}")

    # 3. Obter previsões do conjunto de teste.
    predictions_scaled, targets_scaled = predict(
        model,
        loaders["test"],
    )

    # 4. Converter previsões e alvos para a escala original.
    predictions = data.target_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).ravel()

    actual = data.y_test_original.reshape(-1)

    # Verificação de alinhamento e quantidade.
    if len(actual) != len(predictions):
        raise ValueError("Quantidade de previsões diferente dos valores reais.")

    if not np.allclose(
        targets_scaled.reshape(-1),
        data.y_test.reshape(-1),
        atol=1e-5,
    ):
        raise ValueError("Os alvos do DataLoader não correspondem ao teste.")

    # Resíduo = valor real - valor previsto.
    residuals = actual - predictions

    # 5. Calcular métricas na escala original.
    metrics = regression_metrics(
        targets_scaled,
        predictions_scaled,
        target_scaler=data.target_scaler,
    )

    print("\nMétricas no conjunto de teste:")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.6f}")

    print("\nResumo dos resíduos:")
    print(f"Quantidade: {len(residuals)}")
    print(f"Média: {np.mean(residuals):.6f} ciclos")
    print(f"Desvio padrão: {np.std(residuals):.6f} ciclos")
    print(f"Mínimo: {np.min(residuals):.6f} ciclos")
    print(f"Máximo: {np.max(residuals):.6f} ciclos")

    # 6. Exportar dados para permitir auditoria e análise posterior.
    results = pd.DataFrame({
        "valor_real_ciclos": actual,
        "valor_previsto_ciclos": predictions,
        "residuo_ciclos": residuals,
    })

    csv_path = FIGURES / "residuos_regressao_teste.csv"
    results.to_csv(csv_path, index=False)

    # 7. Gráfico de resíduos versus valores previstos.
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.scatter(
        predictions,
        residuals,
        alpha=0.25,
        s=8,
        label="Observações do teste",
    )
    ax.axhline(
        0,
        linestyle="--",
        label="Resíduo zero",
    )

    ax.set_title("Resíduos versus valores previstos — Regressão R1")
    ax.set_xlabel("Valor previsto (ciclos)")
    ax.set_ylabel("Resíduo (ciclos)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    residual_plot = FIGURES / "residuos_regressao.png"
    fig.savefig(residual_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 8. Histograma da distribuição dos resíduos.
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.hist(
        residuals,
        bins=50,
        edgecolor="black",
        alpha=0.75,
    )
    ax.axvline(
        0,
        linestyle="--",
        label="Resíduo zero",
    )
    ax.axvline(
        np.mean(residuals),
        linestyle=":",
        label=f"Média: {np.mean(residuals):.2f} ciclos",
    )

    ax.set_title("Distribuição dos resíduos — Regressão R1")
    ax.set_xlabel("Resíduo (ciclos)")
    ax.set_ylabel("Frequência")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    histogram_plot = FIGURES / "distribuicao_residuos_regressao.png"
    fig.savefig(histogram_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nArquivos gerados:")
    print(f"- {csv_path.relative_to(ROOT)}")
    print(f"- {residual_plot.relative_to(ROOT)}")
    print(f"- {histogram_plot.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
