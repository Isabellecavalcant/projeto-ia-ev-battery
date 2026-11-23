from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.pytorch.preprocessing import prepare_data
from src.pytorch.dataset import create_dataloaders
from src.pytorch.models import MLPRegressor
from src.pytorch.train import set_seed, train_model
from src.pytorch.evaluate import predict, regression_metrics


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "regression"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "regression"

SEEDS = [42, 123, 2026]

EPOCHS = 20
PATIENCE = 5

NEURONS_VALUES = [32, 64]
DROPOUT_VALUES = [0.0, 0.3]


# ============================================================
# EXPERIMENTO INDIVIDUAL
# ============================================================

def run_single_experiment(data, neurons, dropout, seed):

    experiment_id = (
        f"interaction_neurons_{neurons}_dropout_{str(dropout).replace('.', '_')}"
    )

    print("\n" + "=" * 65)
    print(f"EXPERIMENTO: {experiment_id}")
    print(f"SEED: {seed}")
    print("=" * 65)

    set_seed(seed)

    model_config = {
        "neurons": neurons,
        "layers": 2,
        "activation": "relu",
        "dropout": dropout,
        "batch_norm": False,
    }

    model = MLPRegressor(
        input_dim=data.X_train.shape[1],
        hidden_dims=[neurons, neurons],
        activation="relu",
        dropout=dropout,
        batch_norm=False,
    )

    loaders = create_dataloaders(
        data,
        batch_size=256,
        seed=seed,
    )

    training_result = train_model(
        model=model,
        loaders=loaders,
        task="regression",
        epochs=EPOCHS,
        learning_rate=0.001,
        optimizer_name="adam",
        weight_decay=0.0,
        batch_size=256,
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

    metrics = regression_metrics(
        targets,
        predictions,
        target_scaler=data.target_scaler,
    )

    result = {
        "experiment_id": experiment_id,
        "neurons": neurons,
        "dropout": dropout,
        "seed": seed,
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "best_epoch": training_result["best_epoch"],
        "training_time": training_result["training_time"],
        "num_parameters": training_result["num_parameters"],
        "train_val_gap": training_result["train_val_gap"],
        "status": "success",
    }

    print("\nRESULTADO:")
    print(f"MAE: {result['mae']:.4f}")
    print(f"RMSE: {result['rmse']:.4f}")
    print(f"R²: {result['r2']:.6f}")

    return result


# ============================================================
# RESUMO ESTATÍSTICO
# ============================================================

def create_summary(results_df):

    summary = (
        results_df
        .groupby(["neurons", "dropout"])
        .agg({
            "mae": ["mean", "std"],
            "rmse": ["mean", "std"],
            "r2": ["mean", "std"],
            "training_time": ["mean", "std"],
        })
    )

    output_path = RESULTS_DIR / "interaction_dropout_neurons_summary.csv"

    summary.to_csv(output_path)

    print(f"\nResumo salvo em: {output_path}")

    print("\nRESUMO DOS EXPERIMENTOS:")
    print(summary.to_string())

    return summary


# ============================================================
# GRÁFICO DE INTERAÇÃO
# ============================================================

def create_interaction_plot(summary):

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics = [
        ("mae", "MAE (ciclos)", "menor é melhor"),
        ("rmse", "RMSE (ciclos)", "menor é melhor"),
        ("r2", "R²", "maior é melhor"),
    ]

    for metric, label, direction in metrics:

        values = summary[(metric, "mean")].unstack(level="dropout")

        fig, ax = plt.subplots(figsize=(8, 5))

        for dropout in values.columns:

            ax.plot(
                values.index,
                values[dropout],
                marker="o",
                linewidth=2,
                label=f"Dropout = {dropout}",
            )

        ax.set_xlabel("Neurônios por camada")
        ax.set_ylabel(label)

        ax.set_title(
            f"Interação entre neurônios e dropout — {label}"
        )

        ax.set_xticks(NEURONS_VALUES)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        output_path = FIGURES_DIR / f"interaction_dropout_neurons_{metric}.png"

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(f"Gráfico salvo: {output_path}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("EXPERIMENTO DE INTERAÇÃO: DROPOUT × NEURÔNIOS")
    print("=" * 65)

    print("\nPreparando dados de regressão...")

    data = prepare_data("regression")

    results = []

    total = len(NEURONS_VALUES) * len(DROPOUT_VALUES) * len(SEEDS)
    progress = 0

    for neurons in NEURONS_VALUES:

        for dropout in DROPOUT_VALUES:

            for seed in SEEDS:

                progress += 1

                print(f"\nPROGRESSO: {progress}/{total}")

                result = run_single_experiment(
                    data=data,
                    neurons=neurons,
                    dropout=dropout,
                    seed=seed,
                )

                results.append(result)

                results_df = pd.DataFrame(results)

                results_path = (
                    RESULTS_DIR / "interaction_dropout_neurons_results.csv"
                )

                results_df.to_csv(
                    results_path,
                    index=False,
                )

    results_df = pd.DataFrame(results)

    summary = create_summary(results_df)

    create_interaction_plot(summary)

    print("\n" + "=" * 65)
    print("EXPERIMENTO DE INTERAÇÃO CONCLUÍDO!")
    print("=" * 65)


if __name__ == "__main__":
    main()