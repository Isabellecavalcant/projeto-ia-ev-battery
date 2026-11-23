from pathlib import Path

import pandas as pd
import torch

from src.pytorch.preprocessing import prepare_data
from src.pytorch.dataset import create_dataloaders
from src.pytorch.models import MLPClassifier, MLPRegressor
from src.pytorch.train import set_seed, train_model
from src.pytorch.evaluate import predict, classification_metrics, regression_metrics


# ============================================================
# CONFIGURAÇÕES GERAIS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

SEEDS = [42, 123, 2026]

EPOCHS = 20
PATIENCE = 5

BATCH_SIZE = 256
LEARNING_RATE = 0.001
OPTIMIZER = "adam"


# ============================================================
# CONFIGURAÇÕES FINAIS DE CLASSIFICAÇÃO
# ============================================================

CLASSIFICATION_CONFIGS = [
    {
        "config_id": "C1",
        "neurons": 32,
        "layers": 2,
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
    },
    {
        "config_id": "C2",
        "neurons": 32,
        "layers": 2,
        "activation": "tanh",
        "dropout": 0.0,
        "weight_decay": 0.001,
    },
    {
        "config_id": "C3",
        "neurons": 32,
        "layers": 2,
        "activation": "gelu",
        "dropout": 0.0,
        "weight_decay": 0.001,
    },
]


# ============================================================
# CONFIGURAÇÕES FINAIS DE REGRESSÃO
# ============================================================

REGRESSION_CONFIGS = [
    {
        "config_id": "R1",
        "neurons": 32,
        "layers": 2,
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
    },
    {
        "config_id": "R2",
        "neurons": 64,
        "layers": 2,
        "activation": "relu",
        "dropout": 0.0,
        "weight_decay": 0.0,
    },
    {
        "config_id": "R3",
        "neurons": 32,
        "layers": 2,
        "activation": "gelu",
        "dropout": 0.0,
        "weight_decay": 0.0,
    },
    {
        "config_id": "R4",
        "neurons": 64,
        "layers": 2,
        "activation": "gelu",
        "dropout": 0.0,
        "weight_decay": 0.0,
    },
]


# ============================================================
# CRIAÇÃO DO MODELO
# ============================================================

def create_model(task, data, config):

    hidden_dims = [
        config["neurons"]
    ] * config["layers"]

    common_args = {
        "input_dim": data.X_train.shape[1],
        "hidden_dims": hidden_dims,
        "activation": config["activation"],
        "dropout": config["dropout"],
        "batch_norm": False,
    }

    if task == "classification":
        return MLPClassifier(**common_args)

    return MLPRegressor(**common_args)


# ============================================================
# EXECUÇÃO DE UMA CONFIGURAÇÃO
# ============================================================

def run_single_experiment(task, data, config, seed):

    config_id = config["config_id"]

    experiment_id = f"final_{config_id}"

    print("\n" + "=" * 65)
    print(f"TAREFA: {task.upper()}")
    print(f"CONFIGURAÇÃO: {config_id}")
    print(f"SEED: {seed}")
    print("=" * 65)

    # A seed é definida antes da criação do modelo
    set_seed(seed)

    model = create_model(
        task,
        data,
        config,
    )

    loaders = create_dataloaders(
        data,
        batch_size=BATCH_SIZE,
        seed=seed,
    )

    model_config = {
        **config,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "optimizer": OPTIMIZER,
    }

    training_result = train_model(
        model=model,
        loaders=loaders,
        task=task,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer_name=OPTIMIZER,
        weight_decay=config["weight_decay"],
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        min_delta=0.0001,
        seed=seed,
        loss_name="default",
        pos_weight=None,
        gradient_clip=None,
        model_config=model_config,
        experiment_id=experiment_id,
    )

    predictions, targets = predict(
        training_result["model"],
        loaders["validation"],
    )

    result = {
        "task": task,
        "config_id": config_id,
        "seed": seed,
        "neurons": config["neurons"],
        "layers": config["layers"],
        "activation": config["activation"],
        "dropout": config["dropout"],
        "weight_decay": config["weight_decay"],
        "best_epoch": training_result["best_epoch"],
        "best_val_loss": training_result["best_val_loss"],
        "training_time": training_result["training_time"],
        "num_parameters": training_result["num_parameters"],
        "train_val_gap": training_result["train_val_gap"],
        "status": "success",
    }

    if task == "classification":

        probabilities = torch.sigmoid(
            torch.as_tensor(predictions)
        ).numpy()

        metrics = classification_metrics(
            targets,
            probabilities,
            threshold=0.5,
        )

        result.update({
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        })

        print(f"\nAccuracy: {result['accuracy']:.6f}")
        print(f"Precision: {result['precision']:.6f}")
        print(f"Recall: {result['recall']:.6f}")
        print(f"F1: {result['f1']:.6f}")
        print(f"ROC-AUC: {result['roc_auc']:.6f}")
        print(f"PR-AUC: {result['pr_auc']:.6f}")

    else:

        metrics = regression_metrics(
            targets,
            predictions,
            target_scaler=data.target_scaler,
        )

        result.update({
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
        })

        print(f"\nMAE: {result['mae']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"R²: {result['r2']:.6f}")

    print(f"Melhor época: {result['best_epoch']}")
    print(f"Tempo: {result['training_time']:.2f} segundos")

    return result


# ============================================================
# EXECUÇÃO DA BUSCA DE UMA TAREFA
# ============================================================

def run_task(task, configs):

    print("\n" + "#" * 70)
    print(f"INICIANDO BUSCA FINAL: {task.upper()}")
    print("#" * 70)

    print("\nPreparando dados...")

    data = prepare_data(task)

    task_dir = EXPERIMENTS_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    results_path = task_dir / "final_search_results.csv"

    results = []

    total = len(configs) * len(SEEDS)
    progress = 0

    for config in configs:

        for seed in SEEDS:

            progress += 1

            print(f"\nPROGRESSO {task}: {progress}/{total}")

            result = run_single_experiment(
                task=task,
                data=data,
                config=config,
                seed=seed,
            )

            results.append(result)

            pd.DataFrame(results).to_csv(
                results_path,
                index=False,
            )

    results_df = pd.DataFrame(results)

    metric_columns = (
        ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
         "best_val_loss", "training_time", "train_val_gap"]
        if task == "classification"
        else ["mae", "rmse", "r2", "best_val_loss",
              "training_time", "train_val_gap"]
    )

    summary = (
        results_df
        .groupby("config_id")[metric_columns]
        .agg(["mean", "std"])
    )

    summary_path = task_dir / "final_search_summary.csv"

    summary.to_csv(summary_path)

    print("\n" + "=" * 70)
    print(f"RESUMO FINAL: {task.upper()}")
    print("=" * 70)

    print(summary.to_string())

    print(f"\nResultados salvos em: {results_path}")
    print(f"Resumo salvo em: {summary_path}")

    return summary


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    print("=" * 70)
    print("BUSCA FINAL DE HIPERPARÂMETROS")
    print("=" * 70)

    print(f"Seeds: {SEEDS}")
    print(f"Épocas máximas: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Otimizador: {OPTIMIZER}")

    run_task(
        task="classification",
        configs=CLASSIFICATION_CONFIGS,
    )

    run_task(
        task="regression",
        configs=REGRESSION_CONFIGS,
    )

    print("\n" + "=" * 70)
    print("BUSCA FINAL CONCLUÍDA!")
    print("=" * 70)


if __name__ == "__main__":
    main()