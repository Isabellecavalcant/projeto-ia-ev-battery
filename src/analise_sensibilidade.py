from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.data_prep import TARGET_REG, PROCESSED_DIR, make_preprocessor


ALVO = TARGET_REG
ARQUIVO = PROCESSED_DIR / "dataset_regression.csv"

SUSPEITAS = [
    "cell_voltage_avg",
    "thermal_runaway_risk",
    "thermal_health_score",
]


def metricas(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def criar_preprocessador(X):
    return make_preprocessor(X)


def main():
    df = pd.read_csv(ARQUIVO).dropna(subset=[ALVO])

    X = df.drop(columns=[ALVO])
    y = df[ALVO]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    removiveis = [col for col in SUSPEITAS if col in X.columns]

    experimentos = [
        ("Todas as variáveis", X_train, X_test),
        (
            "Sem as variáveis suspeitas",
            X_train.drop(columns=removiveis),
            X_test.drop(columns=removiveis),
        ),
    ]

    resultados = []

    for nome, treino, teste in experimentos:
        print(f"\nTreinando: {nome}", flush=True)
        inicio = time.perf_counter()

        modelo = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1,
        )

        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ("preprocessor", criar_preprocessador(treino)),
            ("model", modelo),
        ])

        pipeline.fit(treino, y_train)
        pred = pipeline.predict(teste)

        resultado = {
            "Experimento": nome,
            **metricas(y_test, pred),
            "Tempo_treino_s": round(time.perf_counter() - inicio, 2),
            "Variaveis_removidas": ", ".join(removiveis) if removiveis else "Nenhuma",
        }

        resultados.append(resultado)
        print(pd.Series(resultado).to_string(), flush=True)

    tabela = pd.DataFrame(resultados)
    destino = Path("reports/tables/analise_sensibilidade_regressao.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)

    print("\nComparação final:")
    print(tabela.to_string(index=False))
    print(f"\nArquivo salvo em: {destino}")


if __name__ == "__main__":
    main()