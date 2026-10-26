"""Cronômetro para medir o esforço humano de elaborar as regras.

Uso (na raiz do projeto, com o .venv ativo):

    python src/cronometro.py sessao --problema classificacao
    python src/cronometro.py sessao --problema regressao
    python src/cronometro.py resumo

Dentro da sessão:  [Enter] pausa/retoma  |  digite f + [Enter] para finalizar.
O tempo em pausa NÃO conta (use a pausa para interrupções). Ao finalizar, o
script pergunta quantas regras você criou e grava em logs/cronometro_regras.csv.
"""
import argparse
import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "cronometro_regras.csv"
RESUMO = ROOT / "reports" / "tables" / "resumo_cronometro.csv"
CAMPOS = ["problema", "sessao", "data", "hora_inicio", "hora_fim", "tempo_ativo_s",
          "tempo_ativo_min", "tempo_pausado_s", "n_interrupcoes", "n_regras", "obs"]


def _proxima_sessao(problema):
    if not LOG.exists():
        return 1
    df = pd.read_csv(LOG)
    return int((df["problema"] == problema).sum()) + 1


def sessao(problema):
    sess = _proxima_sessao(problema)
    print(f"\nSessão {sess} — problema: {problema}")
    input("Pressione [Enter] para INICIAR o cronômetro... ")
    inicio = datetime.now()
    ativo, pausado, n_pausas = 0.0, 0.0, 0
    t_ref, rodando = time.perf_counter(), True
    print("Cronômetro rodando. [Enter] pausa/retoma | f + [Enter] finaliza.")
    while True:
        cmd = input("").strip().lower()
        agora = time.perf_counter()
        if cmd == "f":
            if rodando:
                ativo += agora - t_ref
            else:
                pausado += agora - t_ref
            break
        if rodando:
            ativo += agora - t_ref
            n_pausas += 1
            print(f"  ⏸  PAUSADO (ativo até agora: {ativo / 60:.1f} min)")
        else:
            pausado += agora - t_ref
            print("  ▶  RETOMADO")
        rodando, t_ref = not rodando, agora
    fim = datetime.now()
    print(f"\nTempo ativo: {ativo / 60:.2f} min | pausas: {n_pausas}")
    while True:
        try:
            n_regras = int(input("Quantas regras você criou nesta sessão? "))
            break
        except ValueError:
            print("Digite um número inteiro.")
    obs = input("Observações (interrupções, dificuldades) [opcional]: ").strip()

    LOG.parent.mkdir(parents=True, exist_ok=True)
    novo = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            w.writeheader()
        w.writerow({
            "problema": problema, "sessao": sess,
            "data": inicio.strftime("%d/%m/%Y"),
            "hora_inicio": inicio.strftime("%H:%M:%S"),
            "hora_fim": fim.strftime("%H:%M:%S"),
            "tempo_ativo_s": round(ativo, 1),
            "tempo_ativo_min": round(ativo / 60, 2),
            "tempo_pausado_s": round(pausado, 1),
            "n_interrupcoes": n_pausas, "n_regras": n_regras, "obs": obs})
    print(f"Sessão gravada em {LOG}")


def resumo_por_problema(problema=None):
    """Total por sessão, média, desvio e tempo médio por regra (por problema)."""
    if not LOG.exists():
        raise FileNotFoundError("Nenhuma sessão registrada ainda.")
    df = pd.read_csv(LOG)
    linhas = []
    for prob, g in df.groupby("problema"):
        if problema and prob != problema:
            continue
        tempos = g["tempo_ativo_min"].tolist()
        linhas.append({
            "problema": prob, "n_sessoes": len(g), "n_regras_total": int(g["n_regras"].sum()),
            "tempo_total_min": round(sum(tempos), 2),
            "tempo_medio_sessao_min": round(statistics.mean(tempos), 2),
            "desvio_sessao_min": round(statistics.stdev(tempos), 2) if len(tempos) > 1 else float("nan"),
            "tempo_medio_por_regra_min": round(sum(tempos) / max(g["n_regras"].sum(), 1), 2),
        })
    return pd.DataFrame(linhas)


def resumo():
    df = resumo_por_problema()
    RESUMO.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESUMO, index=False)
    print(df.to_string(index=False))
    for prob, g in pd.read_csv(LOG).groupby("problema"):
        if len(g) < 3:
            print(f"\n⚠ '{prob}': só {len(g)} sessão(ões); o enunciado pede no mínimo 3.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sessao")
    s.add_argument("--problema", choices=["classificacao", "regressao"], required=True)
    sub.add_parser("resumo")
    a = ap.parse_args()
    sessao(a.problema) if a.cmd == "sessao" else resumo()