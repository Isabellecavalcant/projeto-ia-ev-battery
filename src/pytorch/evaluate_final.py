import json
from pathlib import Path

import numpy as np
import torch

from preprocessing import prepare_data
from dataset import create_dataloaders
from models import MLPClassifier, MLPRegressor
from evaluate import (
    predict,
    find_best_threshold,
    classification_metrics,
    regression_metrics,
)


DEVICE = torch.device("cpu")
BATCH_SIZE = 256


def load_model(model, checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    state_dict = checkpoint["model_state_dict"]

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


def evaluate_classification():
    print("\n" + "=" * 65)
    print("AVALIAÇÃO FINAL — CLASSIFICAÇÃO")
    print("=" * 65)

    data = prepare_data("classification")
    loaders = create_dataloaders(
        data,
        batch_size=BATCH_SIZE,
        seed=123,
    )

    model = MLPClassifier(
        input_dim=data.X_train.shape[1],
        hidden_dims=[32, 32],
        activation="gelu",
        dropout=0.0,
        batch_norm=False,
    )

    checkpoint = Path("models/classification/final_C3_seed_123.pt")
    model = load_model(model, checkpoint)

    # Primeiro, escolhemos o limiar usando apenas a validação.
    val_logits, y_val = predict(model, loaders["validation"])
    val_probabilities = 1 / (1 + np.exp(-val_logits.reshape(-1)))

    threshold = find_best_threshold(
        y_val.reshape(-1),
        val_probabilities,
    )

    print(f"Limiar escolhido na validação: {threshold:.6f}")

    # Somente depois avaliamos o conjunto de teste.
    test_logits, y_test = predict(model, loaders["test"])
    test_probabilities = 1 / (1 + np.exp(-test_logits.reshape(-1)))

    metrics = classification_metrics(
        y_test.reshape(-1),
        test_probabilities,
        threshold=threshold,
    )

    print("\nMétricas no conjunto de teste:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return {
        "task": "classification",
        "config": "C3",
        "seed": 123,
        "threshold": float(threshold),
        "metrics": metrics,
    }


def evaluate_regression():
    print("\n" + "=" * 65)
    print("AVALIAÇÃO FINAL — REGRESSÃO")
    print("=" * 65)

    data = prepare_data("regression")
    loaders = create_dataloaders(
        data,
        batch_size=BATCH_SIZE,
        seed=42,
    )

    model = MLPRegressor(
        input_dim=data.X_train.shape[1],
        hidden_dims=[32, 32],
        activation="relu",
        dropout=0.0,
        batch_norm=False,
    )

    checkpoint = Path("models/regression/final_R1_seed_42.pt")
    model = load_model(model, checkpoint)

    predictions, targets = predict(model, loaders["test"])

    metrics = regression_metrics(
        targets,
        predictions,
        target_scaler=data.target_scaler,
    )

    print("\nMétricas no conjunto de teste:")
    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return {
        "task": "regression",
        "config": "R1",
        "seed": 42,
        "metrics": metrics,
    }


def make_json_serializable(value):
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.bool_,)):
        return bool(value)

    return value


def main():
    classification_results = evaluate_classification()
    regression_results = evaluate_regression()

    results = {
        "classification": classification_results,
        "regression": regression_results,
    }

    output_path = Path("reports/final_test_metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            make_json_serializable(results),
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\n" + "=" * 65)
    print("AVALIAÇÃO FINAL CONCLUÍDA")
    print("=" * 65)
    print(f"Resultados salvos em: {output_path}")


if __name__ == "__main__":
    main()