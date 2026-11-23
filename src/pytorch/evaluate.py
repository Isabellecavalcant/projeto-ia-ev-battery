from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_recall_curve,
)


DEVICE = torch.device("cpu")


# ============================================================
# OBTENÇÃO DAS PREVISÕES
# ============================================================

def predict(model, loader):
    """
    Obtém as previsões de uma rede neural para todos
    os exemplos de um DataLoader.
    """

    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)

            output = model(X_batch)

            predictions.append(
                output.cpu().numpy()
            )

            targets.append(
                y_batch.numpy()
            )

    predictions = np.concatenate(predictions)
    targets = np.concatenate(targets)

    return predictions, targets


# ============================================================
# ESCOLHA DO LIMIAR DE CLASSIFICAÇÃO
# ============================================================

def find_best_threshold(y_true, probabilities):
    """
    Seleciona o limiar que maximiza o F1-score.

    Deve ser utilizado na validação, nunca no teste.
    """

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    # O último elemento de precision e recall não possui
    # um threshold correspondente.
    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = (
        2 * precision * recall
        / (precision + recall + 1e-10)
    )

    best_index = np.argmax(f1_scores)

    return float(thresholds[best_index])


# ============================================================
# MÉTRICAS DE CLASSIFICAÇÃO
# ============================================================

def classification_metrics(
    y_true,
    probabilities,
    threshold=0.5,
):
    """
    Calcula as principais métricas de classificação binária.
    """

    y_pred = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "threshold": float(threshold),

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),

        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),

        "pr_auc": average_precision_score(
            y_true,
            probabilities,
        ),

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ),
    }

    return metrics


# ============================================================
# MÉTRICAS DE REGRESSÃO
# ============================================================

def regression_metrics(
    y_true,
    predictions,
    target_scaler=None,
):
    """
    Calcula métricas de regressão na escala original.

    Se o alvo foi padronizado, aplica inverse_transform
    antes de calcular as métricas.
    """

    y_true = np.asarray(y_true).reshape(-1, 1)
    predictions = np.asarray(predictions).reshape(-1, 1)

    if target_scaler is not None:

        y_true = target_scaler.inverse_transform(y_true)

        predictions = target_scaler.inverse_transform(
            predictions
        )

    y_true = y_true.ravel()
    predictions = predictions.ravel()

    metrics = {
        "mae": mean_absolute_error(
            y_true,
            predictions,
        ),

        "rmse": np.sqrt(
            mean_squared_error(
                y_true,
                predictions,
            )
        ),

        "r2": r2_score(
            y_true,
            predictions,
        ),
    }

    return metrics


# ============================================================
# TESTE DE AVALIAÇÃO DA CLASSIFICAÇÃO
# ============================================================

if __name__ == "__main__":

    from src.pytorch.preprocessing import prepare_data
    from src.pytorch.dataset import create_dataloaders
    from src.pytorch.models import MLPClassifier

    # --------------------------------------------------------
    # 1. Preparar dados
    # --------------------------------------------------------

    data = prepare_data("classification")

    loaders = create_dataloaders(
        data,
        batch_size=256,
        seed=42,
    )

    # --------------------------------------------------------
    # 2. Carregar checkpoint
    # --------------------------------------------------------

    checkpoint_path = Path(
        "models/classification/best_seed_42.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model_config = checkpoint["model_config"]

    model = MLPClassifier(**model_config)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)

    # --------------------------------------------------------
    # 3. Obter previsões da validação
    # --------------------------------------------------------

    logits, y_true = predict(
        model,
        loaders["validation"],
    )

    probabilities = torch.sigmoid(
        torch.tensor(logits)
    ).numpy().ravel()

    y_true = y_true.ravel().astype(int)

    # --------------------------------------------------------
    # 4. Avaliação com threshold 0.5
    # --------------------------------------------------------

    metrics_default = classification_metrics(
        y_true,
        probabilities,
        threshold=0.5,
    )

    print("\n" + "=" * 60)
    print("CLASSIFICAÇÃO - THRESHOLD 0.5")
    print("=" * 60)

    for name, value in metrics_default.items():

        if name == "confusion_matrix":
            print(f"\n{name}:")
            print(value)

        elif isinstance(value, float):
            print(f"{name}: {value:.6f}")

    # --------------------------------------------------------
    # 5. Encontrar melhor threshold na validação
    # --------------------------------------------------------

    best_threshold = find_best_threshold(
        y_true,
        probabilities,
    )

    metrics_optimized = classification_metrics(
        y_true,
        probabilities,
        threshold=best_threshold,
    )

    print("\n" + "=" * 60)
    print("CLASSIFICAÇÃO - THRESHOLD OTIMIZADO")
    print("=" * 60)

    for name, value in metrics_optimized.items():

        if name == "confusion_matrix":
            print(f"\n{name}:")
            print(value)

        elif isinstance(value, float):
            print(f"{name}: {value:.6f}")

    # --------------------------------------------------------
    # 6. Salvar resultados
    # --------------------------------------------------------

    results = []

    for label, metrics in [
        ("threshold_0.5", metrics_default),
        ("threshold_optimized", metrics_optimized),
    ]:

        row = {
            "evaluation": label,
        }

        for key, value in metrics.items():

            if key != "confusion_matrix":
                row[key] = value

        results.append(row)

    results_dir = Path("experiments/classification")
    results_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        results_dir / "validation_metrics.csv",
        index=False,
    )

    print("\nResultados salvos em:")
    print(results_dir / "validation_metrics.csv")