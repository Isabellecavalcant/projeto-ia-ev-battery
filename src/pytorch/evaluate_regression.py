from pathlib import Path

import pandas as pd
import torch

from src.pytorch.preprocessing import prepare_data
from src.pytorch.dataset import create_dataloaders
from src.pytorch.models import MLPRegressor
from src.pytorch.evaluate import predict, regression_metrics


DEVICE = torch.device("cpu")


def main():

    print("\n" + "=" * 60)
    print("AVALIAÇÃO DO MODELO DE REGRESSÃO")
    print("=" * 60)

    # 1. Preparar os dados
    data = prepare_data("regression")

    loaders = create_dataloaders(
        data,
        batch_size=256,
        seed=42,
    )

    # 2. Carregar o checkpoint
    checkpoint_path = Path(
        "models/regression/best_seed_42.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model_config = checkpoint["model_config"]

    model = MLPRegressor(**model_config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    # 3. Obter previsões da validação
    predictions, y_true = predict(
        model,
        loaders["validation"],
    )

    # 4. Calcular métricas na escala original
    metrics = regression_metrics(
        y_true=y_true,
        predictions=predictions,
        target_scaler=data.target_scaler,
    )

    # 5. Exibir resultados
    print("\nRESULTADOS NA VALIDAÇÃO")
    print("-" * 40)

    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.6f}")

    # 6. Salvar resultados
    output_dir = Path("experiments/regression")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])

    metrics_path = output_dir / "validation_metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    print("\nMétricas salvas em:")
    print(metrics_path)

    print("\n" + "=" * 60)
    print("AVALIAÇÃO CONCLUÍDA")
    print("=" * 60)


if __name__ == "__main__":
    main()