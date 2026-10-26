#Treina, ajusta, avalia e mede Árvore de Decisão e Ensembles.

#Uso (na raiz do projeto, com o .venv ativo):
#    python src/run_models.py --task classification --fast   # teste rápido
#    python src/run_models.py --task classification          # execução completa
#    python src/run_models.py --task regression
#
#Saídas:
#    reports/tables/resultados_<task>.csv        métricas, gap, CV, eficiência
#    reports/tables/comparacao_criterios_<task>.csv
#    reports/tables/curva_rf_<task>.csv
#    reports/tables/importancia_<modelo>_<task>.csv
#    reports/tables/regras_arvore_<task>.txt
#    reports/figures/arvore_<task>.png
#    reports/predictions/<task>_<modelo>.npz     (y_true, y_pred, y_score do teste)


import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_text, plot_tree

import models as M
from data_prep import PROCESSED_DIR, ROOT, build_clean_datasets, get_split, make_pipeline
from metrics import (classification_metrics, generalization_gap, get_scores,
                     regression_metrics)
from profiling import medir_inferencia, medir_treinamento, tamanho_em_disco_kb

TABLES = ROOT / "reports" / "tables"
FIGS = ROOT / "reports" / "figures"
PREDS = ROOT / "reports" / "predictions"

FULL = dict(
    train_cap=None,
    tune_n=8000,
    tune_cv=2,
    cv_k=3,
    n_rep=5,
    rf_iter=2,
    hgb_iter=2,
    grid={
        "model__max_depth": [5, 10],
        "model__min_samples_leaf": [10, 50],
        "model__ccp_alpha": [0.0, 10.0, 100.0],
    },
    rf_dist={
        **M.RF_DIST,
        "model__n_estimators": [30, 50],
    },
    hgb_dist={
        **M.HGB_DIST,
        "model__max_iter": [50, 100],
    },
    curve_ns=[1, 10, 30, 50],
)

FAST = dict(
    train_cap=15000,
    tune_n=4000,
    tune_cv=2,
    cv_k=2,
    n_rep=2,
    rf_iter=1,
    hgb_iter=1,
    grid={
        "model__max_depth": [3, 8],
        "model__min_samples_leaf": [10],
        "model__ccp_alpha": [0.0],
    },
    rf_dist={
        **M.RF_DIST,
        "model__n_estimators": [20, 40],
    },
    hgb_dist={
        **M.HGB_DIST,
        "model__max_iter": [30, 60],
    },
    curve_ns=[1, 5, 20],
)


def avaliar(task, nome, pipe, Xtr, ytr, Xte, yte, extra, cfg, linhas):
    """Métricas treino/teste, gap, CV, eficiência; salva predições do teste."""
    clf = task == "classification"
    p_tr, p_te = pipe.predict(Xtr), pipe.predict(Xte)
    if clf:
        s_tr, s_te = get_scores(pipe, Xtr), get_scores(pipe, Xte)
        m_tr = classification_metrics(ytr, p_tr, s_tr)
        m_te = classification_metrics(yte, p_te, s_te)
    else:
        s_te = None
        m_tr, m_te = regression_metrics(ytr, p_tr), regression_metrics(yte, p_te)
    gap = generalization_gap(m_tr, m_te)

    cv = M.cv_splitter(task, cfg["cv_k"])
    scores = cross_val_score(clone(pipe), Xtr, ytr, cv=cv, scoring=M.SCORING[task], n_jobs=1)
    if not clf:
        scores = -scores
    inf = medir_inferencia(pipe.predict, Xte, n_rep=cfg["n_rep"])

    linha = {"tarefa": task, "modelo": nome, "n_cpus": os.cpu_count(),
             **{f"treino_{k}": v for k, v in m_tr.items()},
             **{f"teste_{k}": v for k, v in m_te.items()}, **gap,
             "cv_media": float(scores.mean()), "cv_desvio": float(scores.std()),
             "cv_metrica": "F1" if clf else "RMSE", **inf,
             "disco_kb": tamanho_em_disco_kb(pipe),
             **M.complexidade(pipe[-1]), **extra}
    linhas.append(linha)
    PREDS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PREDS / f"{task}_{nome}.npz", y_true=np.asarray(yte),
                        y_pred=p_te, y_score=s_te if s_te is not None else np.nan)
    print(f"  [{nome}] teste:",
          {k: round(v, 4) for k, v in m_te.items() if k in ("F1", "PR_AUC", "RMSE", "R2")},
          "| CV:", round(scores.mean(), 4), "+-", round(scores.std(), 4))


def salvar_importancia(task, nome, pipe):
    modelo = pipe[-1]
    if not hasattr(modelo, "feature_importances_"):
        return
    nomes = pipe[:-1].get_feature_names_out()
    df = (pd.DataFrame({"feature": nomes, "importancia": modelo.feature_importances_})
          .sort_values("importancia", ascending=False))
    df.to_csv(TABLES / f"importancia_{nome}_{task}.csv", index=False)


def run(task, fast):
    cfg = FAST if fast else FULL
    for d in (TABLES, FIGS, PREDS):
        d.mkdir(parents=True, exist_ok=True)
    if not (PROCESSED_DIR / f"dataset_{task}.csv").exists():
        build_clean_datasets()
    Xtr, Xte, ytr, yte = get_split(task)
    if cfg["train_cap"]:
        Xtr, ytr = M.subsample(Xtr, ytr, cfg["train_cap"], task)
        Xte, yte = Xte.iloc[:5000], yte.iloc[:5000]
        print("*** MODO RÁPIDO: apenas teste de funcionamento, não use estes números ***")
    print(f"{task}: treino={Xtr.shape} teste={Xte.shape}")
    Xs, ys = M.subsample(Xtr, ytr, cfg["tune_n"], task)
    linhas, ajuste = [], {}

    # ---------------------------------------------------------- árvore de decisão
    t0 = time.perf_counter()
    best_tree, comp = M.tune_tree(task, Xs, ys, grid=cfg["grid"], cv_k=cfg["tune_cv"])
    t_tune_tree = time.perf_counter() - t0
    comp.to_csv(TABLES / f"comparacao_criterios_{task}.csv", index=False)
    ajuste["tree_tuned"] = best_tree
    print("árvore: melhores params", best_tree)

    # árvore sem restrição (evidencia overfitting)
    free = make_pipeline(M.tree_est(task), Xtr)
    _, tinfo = medir_treinamento(lambda: free.fit(Xtr, ytr))
    avaliar(task, "tree_unrestricted", free, Xtr, ytr, Xte, yte, tinfo, cfg, linhas)

    tuned = make_pipeline(M.tree_est(task, **best_tree), Xtr)
    _, tinfo = medir_treinamento(lambda: tuned.fit(Xtr, ytr))
    avaliar(task, "tree_tuned", tuned, Xtr, ytr, Xte, yte,
            {**tinfo, "tuning_tempo_s": t_tune_tree}, cfg, linhas)
    salvar_importancia(task, "tree_tuned", tuned)
    nomes = tuned[:-1].get_feature_names_out()
    (TABLES / f"regras_arvore_{task}.txt").write_text(
        export_text(tuned[-1], feature_names=list(nomes), max_depth=4))
    plt.figure(figsize=(22, 9))
    plot_tree(tuned[-1], max_depth=3, feature_names=list(nomes), filled=True,
              fontsize=7, proportion=True)
    plt.savefig(FIGS / f"arvore_{task}.png", dpi=130, bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------------- random forest
    t0 = time.perf_counter()
    best_rf, sc = M.tune_random(task, M.rf_est(task), cfg["rf_dist"], Xs, ys,
                                n_iter=cfg["rf_iter"], cv_k=cfg["tune_cv"])
    t_tune_rf = time.perf_counter() - t0
    ajuste["rf"] = best_rf
    print("RF: melhores params", best_rf, "cv", round(sc, 4))
    rf = make_pipeline(M.rf_est(task, **best_rf), Xtr)
    _, tinfo = medir_treinamento(lambda: rf.fit(Xtr, ytr))
    avaliar(task, "random_forest", rf, Xtr, ytr, Xte, yte,
            {**tinfo, "tuning_tempo_s": t_tune_rf}, cfg, linhas)
    salvar_importancia(task, "random_forest", rf)
    M.rf_error_curve(task, best_rf, Xtr, ytr, ns=cfg["curve_ns"]).to_csv(
        TABLES / f"curva_rf_{task}.csv", index=False)

    # ---------------------------------------------- gradient boosting (histograma)
    t0 = time.perf_counter()
    best_gb, sc = M.tune_random(task, M.hgb_est(task), cfg["hgb_dist"], Xs, ys,
                                n_iter=cfg["hgb_iter"], cv_k=cfg["tune_cv"])
    t_tune_gb = time.perf_counter() - t0
    ajuste["gradient_boosting"] = best_gb
    print("GB: melhores params", best_gb, "cv", round(sc, 4))
    gb = make_pipeline(M.hgb_est(task, **best_gb), Xtr)
    _, tinfo = medir_treinamento(lambda: gb.fit(Xtr, ytr))
    avaliar(task, "gradient_boosting", gb, Xtr, ytr, Xte, yte,
            {**tinfo, "tuning_tempo_s": t_tune_gb}, cfg, linhas)

    suf = "_FAST" if fast else ""
    pd.DataFrame(linhas).to_csv(TABLES / f"resultados_{task}{suf}.csv", index=False)
    (TABLES / f"hiperparametros_{task}{suf}.json").write_text(
        json.dumps(ajuste, indent=2, default=str))
    print(f"\nPronto. Resultados em reports/tables/resultados_{task}{suf}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--fast", action="store_true", help="teste rápido (dados reduzidos)")
    a = ap.parse_args()
    run(a.task, a.fast)