from pathlib import Path
import gc
import itertools
import json
import time

import numpy as np
import pandas as pd
import torch

from src.pytorch.preprocessing import prepare_data
from src.pytorch.dataset import create_dataloaders
from src.pytorch.models import MLPClassifier, MLPRegressor
from src.pytorch.train import train_model, set_seed
from src.pytorch.evaluate import (
    predict,
    classification_metrics,
    regression_metrics,
)


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

SEEDS = [42, 123, 2026]

EPOCHS = 20
PATIENCE = 5

BASELINE = {
    "neurons": 32,
    "layers": 2,
    "activation": "relu",
    "optimizer": "adam",
    "learning_rate": 0.001,
    "batch_size": 256,
    "dropout": 0.2,
    "weight_decay": 0.0,
    "batch_norm": False,
}


# ============================================================
# CONFIGURAÇÕES DOS EXPERIMENTOS
# ============================================================

def build_experiments():

    experiments = [
        {
            "experiment_id": "baseline",
            "factor": "baseline",
            "config": BASELINE.copy(),
        }
    ]

    variations = {
        "neurons": [16, 64],
        "layers": [1, 3],
        "activation": ["tanh", "gelu"],
        "optimizer": ["sgd", "adamw"],
        "learning_rate": [0.0001, 0.01],
        "batch_size": [128, 512],
        "dropout": [0.0, 0.3],
        "weight_decay": [0.00001, 0.001],
    }

    for factor, values in variations.items():

        for value in values:

            config = BASELINE.copy()
            config[factor] = value

            experiments.append({
                "experiment_id": f"{factor}_{str(value).replace('.', '_')}",
                "factor": factor,
                "config": config,
            })

    return experiments


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

def create_model(task, data, config):

    hidden_dims = [
        config["neurons"]
        for _ in range(config["layers"])
    ]

    model_config = {
        "input_dim": data.X_train.shape[1],
        "hidden_dims": hidden_dims,
        "activation": config["activation"],
        "dropout": config["dropout"],
        "batch_norm": config["batch_norm"],
    }

    if task == "classification":
        model = MLPClassifier(**model_config)

    else:
        model = MLPRegressor(**model_config)

    return model, model_config


# ============================================================
# EXECUÇÃO DE UMA CONFIGURAÇÃO
# ============================================================

def run_single_experiment(task, data, experiment, seed):

    config = experiment["config"]
    experiment_id = experiment["experiment_id"]

    print("\n" + "=" * 70)
    print(f"TAREFA: {task.upper()}")
    print(f"EXPERIMENTO: {experiment_id}")
    print(f"SEED: {seed}")
    print("=" * 70)

    set_seed(seed)

    model, model_config = create_model(
        task,
        data,
        config,
    )

    loaders = create_dataloaders(
        data,
        batch_size=config["batch_size"],
        seed=seed,
    )

    result = train_model(
        model=model,
        loaders=loaders,
        task=task,
        epochs=EPOCHS,
        learning_rate=config["learning_rate"],
        optimizer_name=config["optimizer"],
        weight_decay=config["weight_decay"],
        batch_size=config["batch_size"],
        patience=PATIENCE,
        seed=seed,
        model_config=model_config,
        experiment_id=f"{task}_{experiment_id}",
    )

    # --------------------------------------------------------
    # AVALIAÇÃO SOMENTE NA VALIDAÇÃO
    # --------------------------------------------------------

    predictions, y_true = predict(
        result["model"],
        loaders["validation"],
    )

    if task == "classification":

        probabilities = torch.sigmoid(
            torch.tensor(predictions)
        ).numpy().ravel()

        y_true = y_true.ravel().astype(int)

        metrics = classification_metrics(
            y_true=y_true,
            probabilities=probabilities,
            threshold=0.5,
        )

        metrics.pop("confusion_matrix", None)

    else:

        metrics = regression_metrics(
            y_true=y_true,
            predictions=predictions,
            target_scaler=data.target_scaler,
        )

    # --------------------------------------------------------
    # REGISTRO DOS RESULTADOS
    # --------------------------------------------------------

    row = {
        "task": task,
        "experiment_id": experiment_id,
        "factor": experiment["factor"],
        "seed": seed,

        **config,

        "best_epoch": result["best_epoch"],
        "best_val_loss": result["best_val_loss"],
        "best_train_loss": result["best_train_loss"],
        "train_val_gap": result["train_val_gap"],
        "training_time": result["training_time"],
        "num_parameters": result["num_parameters"],

        **metrics,

        "status": "success",
        "error": "",
    }

    print("\nMÉTRICAS:")
    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nTempo: {result['training_time']:.2f} segundos")
    print(f"Melhor época: {result['best_epoch']}")

    del model
    del loaders
    del result

    gc.collect()

    return row


# ============================================================
# EXECUÇÃO DE UMA TAREFA COMPLETA
# ============================================================

def run_task(task):

    print("\n" + "#" * 70)
    print(f"INICIANDO EXPERIMENTOS DE {task.upper()}")
    print("#" * 70)

    data = prepare_data(task)

    experiments = build_experiments()

    output_dir = Path(f"experiments/{task}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "hyperparameter_results.csv"

    # Mantém resultados anteriores caso o processo seja interrompido.
    if results_path.exists():
        existing = pd.read_csv(results_path)

        completed = set(
            zip(
                existing["experiment_id"],
                existing["seed"],
            )
        )

        results = existing.to_dict("records")

    else:
        completed = set()
        results = []

    total = len(experiments) * len(SEEDS)
    current = len(completed)

    print(f"\nConfigurações: {len(experiments)}")
    print(f"Seeds por configuração: {len(SEEDS)}")
    print(f"Execuções previstas: {total}")
    print(f"Execuções já registradas: {current}")

    for experiment in experiments:

        for seed in SEEDS:

            key = (
                experiment["experiment_id"],
                seed,
            )

            if key in completed:
                print(
                    f"\nPulando {experiment['experiment_id']} "
                    f"com seed {seed}: já registrado."
                )
                continue

            current += 1

            print(f"\nPROGRESSO: {current}/{total}")

            try:

                row = run_single_experiment(
                    task=task,
                    data=data,
                    experiment=experiment,
                    seed=seed,
                )

            except Exception as error:

                print("\nERRO NESTA CONFIGURAÇÃO:")
                print(str(error))

                row = {
                    "task": task,
                    "experiment_id": experiment["experiment_id"],
                    "factor": experiment["factor"],
                    "seed": seed,
                    **experiment["config"],
                    "status": "failed",
                    "error": str(error),
                }

            results.append(row)

            # Salva depois de cada execução para evitar perder progresso.
            pd.DataFrame(results).to_csv(
                results_path,
                index=False,
            )

    print("\n" + "=" * 70)
    print(f"EXPERIMENTOS DE {task.upper()} FINALIZADOS")
    print(f"Resultados: {results_path}")
    print("=" * 70)


# ============================================================
# RESUMO FINAL
# ============================================================

def create_summary():

    for task in ["classification", "regression"]:

        path = Path(
            f"experiments/{task}/hyperparameter_results.csv"
        )

        if not path.exists():
            continue

        df = pd.read_csv(path)

        successful = df[df["status"] == "success"]

        if successful.empty:
            continue

        numeric_columns = [
            column
            for column in [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "mae",
                "rmse",
                "r2",
                "best_val_loss",
                "training_time",
                "train_val_gap",
                "num_parameters",
            ]
            if column in successful.columns
        ]

        summary = (
            successful
            .groupby("experiment_id")[numeric_columns]
            .agg(["mean", "std"])
        )

        summary_path = Path(
            f"experiments/{task}/hyperparameter_summary.csv"
        )

        summary.to_csv(summary_path)

        print(f"\nResumo de {task}: {summary_path}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":

    start = time.time()

    run_task("classification")
    run_task("regression")

    create_summary()

    elapsed = time.time() - start

    print("\n" + "=" * 70)
    print("TODOS OS EXPERIMENTOS CONCLUÍDOS")
    print(f"Tempo total: {elapsed / 60:.2f} minutos")
    print("=" * 70)