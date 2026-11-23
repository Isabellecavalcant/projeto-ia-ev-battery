from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {
        "task": "classification",
        "config": "C3",
        "seed": 123,
        "history": ROOT / "experiments/classification/final_C3_seed_123.csv",
        "best_epoch": 9,
        "title": "Curva de aprendizado — Classificação (C3)",
        "filename": "curva_aprendizado_classificacao.png",
    },
    {
        "task": "regression",
        "config": "R1",
        "seed": 42,
        "history": ROOT / "experiments/regression/final_R1_seed_42.csv",
        "best_epoch": 15,
        "title": "Curva de aprendizado — Regressão (R1)",
        "filename": "curva_aprendizado_regressao.png",
    },
]

for exp in EXPERIMENTS:
    df = pd.read_csv(exp["history"])

    if not {"epoch", "train_loss", "val_loss"}.issubset(df.columns):
        raise ValueError(f"Colunas inesperadas em {exp['history']}")

    best_row = df.loc[df["val_loss"].idxmin()]
    if int(best_row["epoch"]) != exp["best_epoch"]:
        raise ValueError(
            f"Época mínima divergente em {exp['history']}: "
            f"esperada {exp['best_epoch']}, encontrada {int(best_row['epoch'])}"
        )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(df["epoch"], df["train_loss"], marker="o",
            markersize=3, label="Perda de treinamento")
    ax.plot(df["epoch"], df["val_loss"], marker="o",
            markersize=3, label="Perda de validação")

    ax.axvline(
        exp["best_epoch"],
        linestyle="--",
        label=f"Melhor época ({exp['best_epoch']})"
    )
    ax.axvline(
        int(df["epoch"].iloc[-1]),
        linestyle=":",
        label=f"Última época ({int(df['epoch'].iloc[-1])})"
    )

    ax.set_title(exp["title"])
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.set_xticks(df["epoch"])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output = FIGURES / exp["filename"]
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura salva: {output.relative_to(ROOT)}")
    print(
        f"  Melhor época: {int(best_row['epoch'])} | "
        f"Train loss: {best_row['train_loss']:.8f} | "
        f"Val loss: {best_row['val_loss']:.8f} | "
        f"Última época: {int(df['epoch'].iloc[-1])}"
    )

# Tabela com os gaps das três seeds das configurações selecionadas.
gap_rows = []

for task, config in [("classification", "C3"), ("regression", "R1")]:
    results_path = ROOT / f"experiments/{task}/final_search_results.csv"
    results = pd.read_csv(results_path)
    selected = results[results["config_id"] == config].copy()

    for _, row in selected.iterrows():
        gap_rows.append({
            "task": task,
            "config_id": config,
            "seed": int(row["seed"]),
            "best_epoch": int(row["best_epoch"]),
            "train_val_gap": row["train_val_gap"],
        })

gaps = pd.DataFrame(gap_rows)
summary = (
    gaps.groupby(["task", "config_id"])["train_val_gap"]
    .agg(["mean", "std"])
    .reset_index()
)

gaps.to_csv(FIGURES / "gaps_generalizacao_por_seed.csv", index=False)
summary.to_csv(FIGURES / "resumo_gaps_generalizacao.csv", index=False)

print("\nGaps por seed:")
print(gaps.to_string(index=False))

print("\nResumo dos gaps:")
print(summary.to_string(index=False))

print("\nArquivos gerados em:", FIGURES)
