from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TASKS = {
    "classification": {
        "metric": "f1",
        "metric_label": "F1-score",
        "title": "Classificação",
        "direction": "max",
    },
    "regression": {
        "metric": "mae",
        "metric_label": "MAE (ciclos)",
        "title": "Regressão",
        "direction": "min",
    },
}

HYPERPARAMETERS = {
    "neurons": "Neurônios por camada",
    "layers": "Número de camadas",
    "activation": "Função de ativação",
    "optimizer": "Otimizador",
    "learning_rate": "Taxa de aprendizado",
    "batch_size": "Tamanho do batch",
    "dropout": "Dropout",
    "weight_decay": "Weight decay",
}


# ============================================================
# LEITURA DOS RESULTADOS
# ============================================================

def load_summary(task):
    path = EXPERIMENTS_DIR / task / "hyperparameter_summary.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}"
        )

    df = pd.read_csv(
        path,
        header=[0, 1],
        index_col=0,
    )

    return df


# ============================================================
# IDENTIFICAÇÃO DAS CONFIGURAÇÕES
# ============================================================

def get_experiment_rows(df, hyperparameter):
    prefix = hyperparameter + "_"

    rows = df.loc[
        df.index.to_series().str.startswith(prefix)
    ].copy()

    if rows.empty:
        raise ValueError(
            f"Nenhum experimento encontrado para {hyperparameter}"
        )

    return rows


def format_label(experiment_id, hyperparameter):
    value = experiment_id.replace(
        hyperparameter + "_",
        "",
        1,
    )

    if hyperparameter == "learning_rate":
        return value.replace("_", ".")

    if hyperparameter == "weight_decay":
        return value.replace("_", ".")

    if hyperparameter == "dropout":
        return value.replace("_", ".")

    return value


# ============================================================
# GERAÇÃO DOS GRÁFICOS INDIVIDUAIS
# ============================================================

def plot_hyperparameter(
    df,
    task,
    hyperparameter,
    configuration,
):
    metric = configuration["metric"]
    metric_label = configuration["metric_label"]

    baseline_mean = df.loc["baseline", (metric, "mean")]

    baseline_std = df.loc["baseline", (metric, "std")]

    experiments = get_experiment_rows(
        df,
        hyperparameter,
    )

    means = experiments[(metric, "mean")]
    stds = experiments[(metric, "std")]

    labels = [
        format_label(index, hyperparameter)
        for index in experiments.index
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    positions = range(len(labels))

    ax.bar(
        positions,
        means.values,
        yerr=stds.values,
        capsize=5,
        label="Experimentos",
    )

    ax.axhline(
        baseline_mean,
        linestyle="--",
        label=f"Baseline ({baseline_mean:.4f})",
    )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels)

    ax.set_xlabel(HYPERPARAMETERS[hyperparameter])
    ax.set_ylabel(metric_label)

    ax.set_title(
        f"{configuration['title']} — "
        f"Influência de {HYPERPARAMETERS[hyperparameter]}"
    )

    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_dir = FIGURES_DIR / task
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_dir / f"{hyperparameter}.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Gráfico salvo: {output_path}")


# ============================================================
# GERAÇÃO DE TODOS OS GRÁFICOS
# ============================================================

def plot_task(task, configuration):
    print("\n" + "=" * 65)
    print(f"GERANDO GRÁFICOS: {task.upper()}")
    print("=" * 65)

    df = load_summary(task)

    for hyperparameter in HYPERPARAMETERS:
        plot_hyperparameter(
            df=df,
            task=task,
            hyperparameter=hyperparameter,
            configuration=configuration,
        )

    print(
        f"\nTodos os gráficos de {task} foram gerados."
    )


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

def main():
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for task, configuration in TASKS.items():
        plot_task(
            task,
            configuration,
        )

    print("\n" + "=" * 65)
    print("TODOS OS GRÁFICOS FORAM GERADOS COM SUCESSO!")
    print("=" * 65)


if __name__ == "__main__":
    main()