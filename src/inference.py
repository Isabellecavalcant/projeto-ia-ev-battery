"""
Inferência independente para os modelos finais de classificação e regressão.

Exemplos:
    python src/inference.py --task classification --input data/novos_dados.csv
    python src/inference.py --task regression --input data/novos_dados.csv

O CSV de entrada deve conter as features originais utilizadas no treinamento.
As colunas-alvo são desnecessárias.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pytorch.models import MLPClassifier, MLPRegressor


DEVICE = torch.device("cpu")

CHECKPOINTS = {
    "classification": ROOT / "models/classification/final_C3_seed_123.pt",
    "regression": ROOT / "models/regression/final_R1_seed_42.pt",
}

PREPROCESSORS = {
    "classification": ROOT / "models/classification/preprocessing.joblib",
    "regression": ROOT / "models/regression/preprocessing.joblib",
}


def load_model(task):
    """Reconstrói a arquitetura e carrega os pesos do checkpoint final."""

    checkpoint_path = CHECKPOINTS[task]
    preprocessing_path = PREPROCESSORS[task]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}"
        )

    if not preprocessing_path.exists():
        raise FileNotFoundError(
            f"Pré-processamento não encontrado: {preprocessing_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    config = checkpoint["model_config"]

    neurons = int(config["neurons"])
    layers = int(config["layers"])

    model_class = (
        MLPClassifier if task == "classification"
        else MLPRegressor
    )

    model = model_class(
        input_dim=163,
        hidden_dims=[neurons] * layers,
        activation=config["activation"],
        dropout=float(config["dropout"]),
        batch_norm=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    preprocessing = joblib.load(preprocessing_path)

    if preprocessing["task"] != task:
        raise ValueError(
            "O artefato de pré-processamento não corresponde à tarefa."
        )

    if len(preprocessing["feature_names"]) != 163:
        raise ValueError(
            "Quantidade inesperada de features no pré-processamento."
        )

    print(f"Modelo carregado: {checkpoint_path.name}")
    print(f"Configuração: {config}")
    print(f"Melhor época: {checkpoint['epoch']}")

    return model, preprocessing


def load_threshold():
    """Recupera o limiar escolhido na validação para classificação."""

    metrics_path = ROOT / "reports/final_test_metrics.json"

    if not metrics_path.exists():
        return 0.5

    with open(metrics_path, "r", encoding="utf-8") as file:
        results = json.load(file)

    return float(
        results["classification"].get("threshold", 0.5)
    )


def run_inference(task, input_path, output_path, threshold=None):
    """Executa inferência sobre um CSV de novos exemplos."""

    input_path = Path(input_path)

    if not input_path.is_absolute():
        input_path = ROOT / input_path

    if not input_path.exists():
        raise FileNotFoundError(
            f"Arquivo de entrada não encontrado: {input_path}"
        )

    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError("O CSV de entrada está vazio.")

    model, preprocessing = load_model(task)

    preprocessor = preprocessing["preprocessor"]
    target_scaler = preprocessing["target_scaler"]

    required_columns = list(preprocessor.feature_names_in_)

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "O CSV não contém todas as features necessárias. "
            f"Colunas ausentes: {missing_columns}"
        )

    # Mantém somente as features e respeita a ordem do treinamento.
    features = dataframe[required_columns]

    # Aplica o pré-processador já ajustado durante o treinamento.
    transformed = preprocessor.transform(features)
    transformed = np.asarray(transformed, dtype=np.float32)

    if transformed.shape[1] != 163:
        raise ValueError(
            f"Esperadas 163 features, recebidas {transformed.shape[1]}."
        )

    tensor = torch.tensor(
        transformed,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        predictions = model(tensor).cpu().numpy().reshape(-1)

    result = dataframe.copy()

    if task == "classification":
        probabilities = 1.0 / (1.0 + np.exp(-predictions))

        if threshold is None:
            threshold = load_threshold()

        labels = (probabilities >= threshold).astype(int)

        result["probabilidade_falha"] = probabilities
        result["previsao_battery_failure"] = labels

        print(f"Limiar utilizado: {threshold:.6f}")

    else:
        if target_scaler is not None:
            predictions = target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).reshape(-1)

        result["previsao_remaining_life_cycles"] = predictions

    if not output_path:
        output_path = (
            ROOT / "reports" / f"inference_{task}.csv"
        )
    else:
        output_path = Path(output_path)

        if not output_path.is_absolute():
            output_path = ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print("\nInferência concluída.")
    print(f"Quantidade de exemplos: {len(result)}")
    print(f"Arquivo gerado: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Inferência com os modelos PyTorch finais."
    )

    parser.add_argument(
        "--task",
        choices=["classification", "regression"],
        required=True,
        help="Tipo de tarefa.",
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Caminho do CSV com as features originais.",
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Caminho do CSV de saída.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Limiar opcional para classificação.",
    )

    args = parser.parse_args()

    run_inference(
        task=args.task,
        input_path=args.input,
        output_path=args.output,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()