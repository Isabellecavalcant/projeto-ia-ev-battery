from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def remover_identificadores(df):
    return df.drop(
        columns=["vehicle_id", "battery_serial"],
        errors="ignore"
    )


def criar_preprocessador(X):
    colunas_numericas = X.select_dtypes(include="number").columns
    colunas_categoricas = X.select_dtypes(
        include=["object", "category"]
    ).columns

    pipeline_numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    pipeline_categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    return ColumnTransformer([
        ("numerico", pipeline_numerico, colunas_numericas),
        ("categorico", pipeline_categorico, colunas_categoricas)
    ])