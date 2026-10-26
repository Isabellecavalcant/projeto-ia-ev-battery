"""Gera gráficos de avaliação a partir das previsões já salvas."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
PREDS = ROOT / "reports" / "predictions"
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def carregar(nome):
    caminho = PREDS / f"{nome}.npz"
    with np.load(caminho) as dados:
        return {chave: dados[chave].copy() for chave in dados.files}


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(FIGS / nome, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Gráfico salvo: reports/figures/{nome}")


def graficos_classificacao():
    modelos = [
        "tree_unrestricted",
        "tree_tuned",
        "random_forest",
        "gradient_boosting",
        "expert_system",
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    roc_count = 0

    for modelo in modelos:
        d = carregar(f"classification_{modelo}")
        y_true = d["y_true"].astype(int)
        score = d["y_score"].ravel()

        if len(score) != len(y_true) or not np.isfinite(score).all():
            print(f"Pontuação inválida; ROC ignorada: {modelo}")
            continue

        # O sistema especialista pode não fornecer probabilidades calibradas.
        # Sua pontuação é utilizada apenas se houver variação útil.
        if np.unique(score).size < 2:
            print(f"Sem variação de pontuação; ROC ignorada: {modelo}")
            continue

        RocCurveDisplay.from_predictions(
            y_true, score, name=modelo, ax=ax
        )
        roc_count += 1

    if roc_count:
        ax.plot([0, 1], [0, 1], linestyle="--", label="Referência aleatória")
        ax.set_title("Curvas ROC — classificação")
        salvar(fig, "curvas_roc_classification.png")
    else:
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    pr_count = 0

    for modelo in modelos:
        d = carregar(f"classification_{modelo}")
        y_true = d["y_true"].astype(int)
        score = d["y_score"].ravel()

        if len(score) != len(y_true) or not np.isfinite(score).all():
            continue
        if np.unique(score).size < 2:
            continue

        PrecisionRecallDisplay.from_predictions(
            y_true, score, name=modelo, ax=ax
        )
        pr_count += 1

    if pr_count:
        prevalencia = np.mean(carregar("classification_gradient_boosting")["y_true"])
        ax.axhline(
            prevalencia,
            linestyle="--",
            label=f"Prevalência positiva ({prevalencia:.3f})",
        )
        ax.set_title("Precisão–Revocação — classificação")
        salvar(fig, "curvas_pr_classification.png")
    else:
        plt.close(fig)

    # Matrizes de confusão dos modelos, incluindo o sistema especialista.
    for modelo in modelos:
        d = carregar(f"classification_{modelo}")
        y_true = d["y_true"].astype(int)
        y_pred = d["y_pred"].astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Sem falha", "Falha"],
        ).plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"Matriz de confusão — {modelo}")
        salvar(fig, f"matriz_confusao_{modelo}.png")


def graficos_regressao():
    modelos = [
        "tree_unrestricted",
        "tree_tuned",
        "random_forest",
        "gradient_boosting",
        "expert_system",
    ]

    # Resíduos: valores observados versus resíduos.
    for modelo in modelos:
        d = carregar(f"regression_{modelo}")
        y_true = d["y_true"].astype(float).ravel()
        y_pred = d["y_pred"].astype(float).ravel()

        validos = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[validos], y_pred[validos]
        residuos = y_true - y_pred

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(y_pred, residuos, s=8, alpha=0.25)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Valor previsto")
        ax.set_ylabel("Resíduo (observado − previsto)")
        ax.set_title(f"Resíduos — {modelo}")
        salvar(fig, f"residuos_{modelo}.png")

    # Observado versus previsto, com linha ideal y = x.
    for modelo in modelos:
        d = carregar(f"regression_{modelo}")
        y_true = d["y_true"].astype(float).ravel()
        y_pred = d["y_pred"].astype(float).ravel()

        validos = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[validos], y_pred[validos]

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(y_true, y_pred, s=8, alpha=0.25)

        if len(y_true):
            minimo = min(y_true.min(), y_pred.min())
            maximo = max(y_true.max(), y_pred.max())
            ax.plot([minimo, maximo], [minimo, maximo], linestyle="--")

        ax.set_xlabel("Valor observado")
        ax.set_ylabel("Valor previsto")
        ax.set_title(f"Observado versus previsto — {modelo}")
        salvar(fig, f"observado_vs_previsto_{modelo}.png")


if __name__ == "__main__":
    graficos_classificacao()
    graficos_regressao()
    print("\nGeração dos gráficos concluída.")