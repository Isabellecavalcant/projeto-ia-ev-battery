"""Avalia o sistema especialista: desempenho, gap e eficiência (só inferência).

Uso (na raiz do projeto, com o .venv ativo):

    # enquanto você ainda está ajustando as regras: SÓ o conjunto de TREINO
    python src/run_expert_system.py --task classification --rules rules/regras_classificacao.txt

    # avaliação FINAL (usa o teste; faça UMA vez, com as regras já congeladas)
    python src/run_expert_system.py --task classification --rules rules/regras_classificacao.txt --final

O teste NÃO deve ser consultado durante a elaboração das regras (vazamento).
"""
import argparse
import os

import numpy as np
import pandas as pd

from data_prep import ROOT, get_split
from expert_system import BaseConhecimento, MotorInferencia
from metrics import classification_metrics, generalization_gap, regression_metrics
from profiling import medir_inferencia, tamanho_em_disco_kb

ALVO = {"classification": "battery_failure",
        "regression": "predicted_remaining_life_cycles"}
PROBLEMA_LOG = {"classification": "classificacao", "regression": "regressao"}
TABLES = ROOT / "reports" / "tables"
PREDS = ROOT / "reports" / "predictions"


def metricas(task, y, pred, score):
    if task == "classification":
        return classification_metrics(y, pred.astype(int), score)
    return regression_metrics(y, pred.astype(float))


def main(task, rules, final, n_rep):
    TABLES.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)
    Xtr, Xte, ytr, yte = get_split(task)
    base = BaseConhecimento(rules)
    motor = MotorInferencia(base, ALVO[task])
    print(f"{len(base.regras)} regras carregadas de {rules}")

    p_tr = motor.predict(Xtr)
    s_tr = motor.predict_score(Xtr) if task == "classification" else None
    m_tr = metricas(task, ytr, p_tr, s_tr)
    mostrar = ("Accuracy", "Precision", "Recall", "F1", "RMSE", "MAE", "R2")
    print("TREINO:", {k: round(v, 4) for k, v in m_tr.items() if k in mostrar})
    cob = motor.cobertura()
    cob["pct_treino"] = (100 * cob["disparos"] / len(Xtr)).round(2)
    print(cob.to_string(index=False))

    if not final:
        print("\n(Só treino. Use --final para avaliar no teste quando as regras estiverem congeladas.)")
        return

    motor.disparos.clear()
    p_te = motor.predict(Xte)
    s_te = motor.predict_score(Xte) if task == "classification" else None
    m_te = metricas(task, yte, p_te, s_te)
    gap = generalization_gap(m_tr, m_te)
    print("TESTE :", {k: round(v, 4) for k, v in m_te.items() if k in mostrar})

    inf = medir_inferencia(motor.predict, Xte, n_rep=n_rep)   # apenas inferência
    linha = {"tarefa": task, "modelo": "expert_system", "n_cpus": os.cpu_count(),
             **{f"treino_{k}": v for k, v in m_tr.items()},
             **{f"teste_{k}": v for k, v in m_te.items()}, **gap, **inf,
             "disco_kb": tamanho_em_disco_kb(str(rules)), "n_regras": len(base.regras)}
    # esforço humano (cronômetro): coluna própria, NÃO somada à inferência
    try:
        from cronometro import resumo_por_problema
        r = resumo_por_problema(PROBLEMA_LOG[task]).iloc[0]
        linha.update({"elaboracao_total_min": r["tempo_total_min"],
                      "elaboracao_media_sessao_min": r["tempo_medio_sessao_min"],
                      "elaboracao_desvio_min": r["desvio_sessao_min"],
                      "elaboracao_min_por_regra": r["tempo_medio_por_regra_min"]})
    except Exception as e:
        print("Aviso: sem dados do cronômetro para este problema:", e)

    pd.DataFrame([linha]).to_csv(TABLES / f"resultados_especialista_{task}.csv", index=False)
    cob.to_csv(TABLES / f"cobertura_regras_{task}.csv", index=False)
    np.savez_compressed(PREDS / f"{task}_expert_system.npz", y_true=np.asarray(yte),
                        y_pred=p_te, y_score=s_te if s_te is not None else np.nan)
    print(f"Salvo em reports/tables/resultados_especialista_{task}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression"], required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--n-rep", type=int, default=30)
    a = ap.parse_args()
    main(a.task, a.rules, a.final, a.n_rep)