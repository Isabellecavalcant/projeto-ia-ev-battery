from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "reports" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def carregar(arquivo):
    caminho = TABLES / arquivo
    if not caminho.exists():
        print(f"Aviso: arquivo ausente: {caminho}")
        return pd.DataFrame()
    return pd.read_csv(caminho)


def consolidar(tarefa):
    if tarefa == "classification":
        arquivo_ml = "resultados_classification.csv"
        arquivo_especialista = "resultados_especialista_classification.csv"
        metricas = [
            "teste_Accuracy",
            "teste_Precision",
            "teste_Recall",
            "teste_F1",
            "teste_ROC_AUC",
            "teste_PR_AUC",
        ]
    else:
        arquivo_ml = "resultados_regression.csv"
        arquivo_especialista = "resultados_especialista_regression.csv"
        metricas = [
            "teste_MAE",
            "teste_RMSE",
            "teste_MAPE_%",
            "teste_R2",
            "teste_MedAE",
        ]

    df_ml = carregar(arquivo_ml)
    df_es = carregar(arquivo_especialista)
    partes = []

    for df in [df_ml, df_es]:
        if not df.empty:
            partes.append(df)

    if not partes:
        return pd.DataFrame()

    df = pd.concat(partes, ignore_index=True, sort=False)
    colunas_desejadas = [
        "tarefa",
        "modelo",
        *metricas,
        "cv_media",
        "cv_desvio",
        "cv_metrica",
        "gap_F1" if tarefa == "classification" else "gap_RMSE",
        "inf_tempo_medio_s",
        "inf_tempo_desvio_s",
        "inf_tempo_por_amostra_ms",
        "inf_vazao_amostras_s",
        "inf_pico_memoria_kb",
        "disco_kb",
        "treino_tempo_s",
        "treino_pico_ram_mb",
        "treino_energia_kwh",
        "treino_co2_g",
        "inf_energia_kwh",
        "inf_co2_g",
        "profundidade",
        "n_nos",
        "n_folhas",
        "n_arvores",
        "n_regras",
        "elaboracao_total_min",
        "elaboracao_media_sessao_min",
        "elaboracao_desvio_min",
        "elaboracao_min_por_regra",
    ]

    colunas = [c for c in colunas_desejadas if c in df.columns]
    resultado = df[colunas].copy()

    destino = TABLES / f"resumo_final_{tarefa}.csv"
    resultado.to_csv(destino, index=False)
    print(f"\nTabela salva: {destino.relative_to(ROOT)}")
    print(resultado.to_string(index=False))
    return resultado


if __name__ == "__main__":
    consolidar("classification")
    consolidar("regression")