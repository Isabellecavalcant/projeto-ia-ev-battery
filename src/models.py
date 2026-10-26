"""Árvore de Decisão e Ensembles (Random Forest + Gradient Boosting).

O ajuste de hiperparâmetros usa SOMENTE o conjunto de treino (validação
cruzada em uma subamostra do treino, para caber no tempo). O teste só entra na
avaliação final.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from data_prep import RANDOM_STATE, make_pipeline, make_preprocessor

# métrica principal usada no ajuste e na validação cruzada
SCORING = {"classification": "f1", "regression": "neg_root_mean_squared_error"}
CRITERIA = {"classification": ["gini", "entropy"],
            "regression": ["squared_error", "friedman_mse"]}

# ccp_alpha tem escala diferente: impureza Gini <= 0,5; variância do alvo ~3e6
TREE_GRID = {
    "classification": {"model__max_depth": [3, 5, 8, 12, 16, None],
                       "model__min_samples_leaf": [1, 10, 50],
                       "model__ccp_alpha": [0.0, 1e-4, 1e-3]},
    "regression": {"model__max_depth": [3, 5, 8, 12, 16, None],
                   "model__min_samples_leaf": [1, 10, 50],
                   "model__ccp_alpha": [0.0, 10.0, 100.0]},
}
RF_DIST = {"model__n_estimators": [100, 200],
           "model__max_depth": [10, 20, None],
           "model__max_features": ["sqrt", 0.3],
           "model__min_samples_leaf": [1, 3, 10]}
HGB_DIST = {"model__learning_rate": [0.03, 0.1, 0.2],
            "model__max_iter": [100, 200, 300],
            "model__max_depth": [3, 6, None],
            "model__min_samples_leaf": [20, 50, 100]}


def _clf(task):
    return task == "classification"


def cv_splitter(task, k):
    cls = StratifiedKFold if _clf(task) else KFold
    return cls(n_splits=k, shuffle=True, random_state=RANDOM_STATE)


def subsample(X, y, n, task):
    """Subamostra do TREINO (estratificada na classificação)."""
    if n is None or n >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(
        X, y, train_size=n, random_state=RANDOM_STATE,
        stratify=y if _clf(task) else None)
    return Xs, ys


# ------------------------------------------------------------------ estimadores
def tree_est(task, **kw):
    cls = DecisionTreeClassifier if _clf(task) else DecisionTreeRegressor
    return cls(random_state=RANDOM_STATE, **kw)


def rf_est(task, **kw):
    cls = RandomForestClassifier if _clf(task) else RandomForestRegressor
    return cls(random_state=RANDOM_STATE, n_jobs=-1, **kw)


def hgb_est(task, **kw):
    cls = HistGradientBoostingClassifier if _clf(task) else HistGradientBoostingRegressor
    return cls(random_state=RANDOM_STATE, **kw)


def _strip(params):
    return {k.replace("model__", ""): v for k, v in params.items()}


# ----------------------------------------------------------------------- árvore
def tune_tree(task, X, y, grid=None, cv_k=3):
    """Compara os critérios de impureza e ajusta max_depth, min_samples_leaf e
    ccp_alpha por GridSearchCV. Retorna (melhores_params, df_comparacao)."""
    grid = grid or TREE_GRID[task]
    linhas, melhor = [], None
    for crit in CRITERIA[task]:
        pipe = make_pipeline(tree_est(task, criterion=crit), X)
        gs = GridSearchCV(pipe, grid, scoring=SCORING[task],
                          cv=cv_splitter(task, cv_k), n_jobs=1)
        gs.fit(X, y)
        linhas.append({"criterio": crit, "cv_score": gs.best_score_,
                       **_strip(gs.best_params_)})
        if melhor is None or gs.best_score_ > melhor[0]:
            melhor = (gs.best_score_, {"criterion": crit, **_strip(gs.best_params_)})
    return melhor[1], pd.DataFrame(linhas)


# -------------------------------------------------------------------- ensembles
def tune_random(task, estimator, dist, X, y, n_iter=8, cv_k=3):
    """RandomizedSearchCV sobre o Pipeline (prep + modelo)."""
    pipe = make_pipeline(estimator, X)
    rs = RandomizedSearchCV(pipe, dist, n_iter=n_iter, scoring=SCORING[task],
                            cv=cv_splitter(task, cv_k), n_jobs=1,
                            random_state=RANDOM_STATE)
    rs.fit(X, y)
    return _strip(rs.best_params_), rs.best_score_


def _erro(task, y_true, y_pred):
    """Erro usado na curva: 1 - F1 (classificação) ou RMSE (regressão)."""
    if _clf(task):
        return 1 - f1_score(y_true, y_pred, zero_division=0)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rf_error_curve(task, best_params, X, y, ns=(1, 10, 50, 100, 200)):
    """Erro da Random Forest em função do número de árvores.

    Usa uma validação interna (20% do TREINO); o conjunto de teste não é
    tocado. Treino alto-viés/baixo-erro vs. validação mostra a variância.
    """
    Xf, Xv, yf, yv = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE,
        stratify=y if _clf(task) else None)
    prep = make_preprocessor(Xf)
    A, V = prep.fit_transform(Xf), prep.transform(Xv)
    base = {k: v for k, v in best_params.items() if k != "n_estimators"}
    rf = rf_est(task, warm_start=True, **base)
    ntr = min(20000, len(A))
    linhas = []
    for n in ns:
        rf.set_params(n_estimators=n)
        rf.fit(A, yf)
        linhas.append({"n_arvores": n,
                       "erro_treino": _erro(task, yf.iloc[:ntr], rf.predict(A[:ntr])),
                       "erro_validacao": _erro(task, yv, rf.predict(V))})
    return pd.DataFrame(linhas)


# ------------------------------------------------------------------ complexidade
def complexidade(modelo) -> dict:
    """Profundidade/nós/folhas (árvore), nº de árvores e nós totais (ensemble)."""
    if hasattr(modelo, "tree_"):
        t = modelo.tree_
        return {"profundidade": int(t.max_depth), "n_nos": int(t.node_count),
                "n_folhas": int(t.n_leaves), "n_arvores": 1}
    if hasattr(modelo, "estimators_"):
        arv = modelo.estimators_
        return {"profundidade": int(max(a.tree_.max_depth for a in arv)),
                "n_nos": int(sum(a.tree_.node_count for a in arv)),
                "n_folhas": int(sum(a.tree_.n_leaves for a in arv)),
                "n_arvores": len(arv)}
    if hasattr(modelo, "_predictors"):  # HistGradientBoosting
        preds = [p for it in modelo._predictors for p in it]
        return {"profundidade": int(max(p.get_max_depth() for p in preds)),
                "n_nos": int(sum(p.get_n_leaf_nodes() * 2 - 1 for p in preds)),
                "n_folhas": int(sum(p.get_n_leaf_nodes() for p in preds)),
                "n_arvores": len(preds)}
    return {}