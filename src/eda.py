import pandas as pd


def resumo_estatistico(df):
    return df.describe().T


def valores_ausentes(df):
    quantidade = df.isnull().sum()
    percentual = (quantidade / len(df)) * 100

    resultado = pd.DataFrame({
        "quantidade_ausentes": quantidade,
        "percentual_ausentes": percentual
    })

    return resultado.sort_values(
        "percentual_ausentes",
        ascending=False
    )


def detectar_outliers_iqr(df, coluna):
    q1 = df[coluna].quantile(0.25)
    q3 = df[coluna].quantile(0.75)
    iqr = q3 - q1

    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr

    return df[
        (df[coluna] < limite_inferior) |
        (df[coluna] > limite_superior)
    ]


def matriz_correlacao(df):
    numericas = df.select_dtypes(include="number")
    return numericas.corr()