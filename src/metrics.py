"""Métricas da Aula 08 para regressão e classificação, mais o gap de
generalização (treino - teste)."""
import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, precision_score, r2_score, recall_score,
    roc_auc_score,
)


# ------------------------------------------------------------------- regressão
def regression_metrics(y_true, y_pred) -> dict:
    """MAE, MSE, RMSE, MAPE, R2 e MedAE.

    ATENÇÃO MAPE: o target tem valores iguais a 0 (mínimo = 0), o que torna o
    MAPE indefinido/explosivo. Aqui ele é calculado só onde y_true != 0, e o
    número de linhas excluídas é reportado em 'mape_n_excluidos'.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    nz = y_true != 0
    mape = (np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100
            if nz.any() else np.nan)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE_%": float(mape),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "mape_n_excluidos": int((~nz).sum()),
    }


# --------------------------------------------------------------- classificação
def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    """Métricas para classificação binária (classe positiva = 1 = falha).

    y_score: probabilidade/pontuação da classe 1. Se None (ex.: sistema
    especialista sem escore), ROC-AUC e PR-AUC ficam como NaN.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": np.nan,
        "PR_AUC": np.nan,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }
    if y_score is not None:
        out["ROC_AUC"] = roc_auc_score(y_true, y_score)
        out["PR_AUC"] = average_precision_score(y_true, y_score)
    return out


def get_scores(modelo, X):
    """Pontuação da classe positiva: predict_proba[:, 1] ou decision_function.
    Devolve None se o modelo não produz pontuação."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, 1]
    if hasattr(modelo, "decision_function"):
        return modelo.decision_function(X)
    return None


# --------------------------------------------------------------------- gap
# Para métricas de ERRO (menor é melhor): gap = teste - treino (positivo =
# o erro piora no teste = overfitting). Para métricas de DESEMPENHO (maior é
# melhor): gap = treino - teste (positivo = piora no teste = overfitting).
MENOR_E_MELHOR = {"MAE", "MSE", "RMSE", "MAPE_%", "MedAE"}


def generalization_gap(m_treino: dict, m_teste: dict, chaves=None) -> dict:
    """Gap de generalização por métrica, com sinal tal que POSITIVO = o modelo
    é pior no teste do que no treino."""
    chaves = chaves or [k for k in m_treino
                        if k in m_teste and isinstance(m_treino[k], float)]
    gap = {}
    for k in chaves:
        if k in MENOR_E_MELHOR:
            gap[f"gap_{k}"] = m_teste[k] - m_treino[k]
        else:
            gap[f"gap_{k}"] = m_treino[k] - m_teste[k]
    return gap