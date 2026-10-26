"""Preparação de dados para a avaliação comparativa de modelos.

Reconstrói os datasets de regressão e classificação a partir do CSV bruto,
seguindo as decisões da atividade anterior, e fornece:
  - split treino/teste FIXO e idêntico para todos os modelos;
  - um ColumnTransformer (imputação + One-Hot + padronização) para uso dentro
    de Pipeline, de modo que nada seja ajustado fora do treino (sem leakage).
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------- configuração
RANDOM_STATE = 42          # semente única do projeto (split, modelos, CV)
TEST_SIZE = 0.20           # 80% treino / 20% teste

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "ev_battery_failure_dataset.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

ID_COLS = ["vehicle_id", "battery_serial"]
TARGET_REG = "predicted_remaining_life_cycles"
TARGET_CLF = "battery_failure"


# --------------------------------------------------------------- datasets limpos
def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def build_clean_datasets(raw_path: Path = RAW_PATH, out_dir: Path = PROCESSED_DIR):
    """Gera dataset_regression.csv e dataset_classification.csv.

    Os arquivos contêm apenas dados LIMPOS (sem imputar, codificar ou
    padronizar). Essas etapas ficam no Pipeline, ajustadas só no treino.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_raw(raw_path).drop(columns=ID_COLS)

    # Regressão: remove linhas sem target; battery_failure não é feature.
    reg = df.dropna(subset=[TARGET_REG]).drop(columns=[TARGET_CLF])
    # Classificação: predicted_remaining_life_cycles não é feature.
    clf = df.dropna(subset=[TARGET_CLF]).drop(columns=[TARGET_REG])
    clf[TARGET_CLF] = clf[TARGET_CLF].astype(int)

    reg.to_csv(out_dir / "dataset_regression.csv", index=False)
    clf.to_csv(out_dir / "dataset_classification.csv", index=False)
    return reg, clf


# ------------------------------------------------------------------------ split
def get_split(task: str, processed_dir: Path = PROCESSED_DIR):
    """Retorna X_train, X_test, y_train, y_test (dados brutos, sem transformar).

    task: "regression" ou "classification".
    O split é determinístico (RANDOM_STATE), logo idêntico em qualquer notebook
    e para os três tipos de modelo. Na classificação é estratificado.
    """
    if task == "regression":
        df, target, stratify = pd.read_csv(processed_dir / "dataset_regression.csv"), TARGET_REG, False
    elif task == "classification":
        df, target, stratify = pd.read_csv(processed_dir / "dataset_classification.csv"), TARGET_CLF, True
    else:
        raise ValueError("task deve ser 'regression' ou 'classification'")

    X, y = df.drop(columns=[target]), df[target]
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if stratify else None,
    )


# ------------------------------------------------------------------ pré-processo
def make_preprocessor(X: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    """Imputação (mediana/moda) + One-Hot nas categóricas + StandardScaler nas
    numéricas. Deve ser usado como primeira etapa de um Pipeline.

    scale=False dispensa a padronização (árvores não precisam dela); mantida
    como True por padrão para manter o mesmo pré-processamento da atividade
    anterior.
    """
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])


def make_pipeline(model, X_train: pd.DataFrame, scale: bool = True) -> Pipeline:
    """Pipeline = pré-processamento + modelo (usar em CV e GridSearch)."""
    return Pipeline([("prep", make_preprocessor(X_train, scale)), ("model", model)])


if __name__ == "__main__":
    reg, clf = build_clean_datasets()
    print("regressão:", reg.shape, "| classificação:", clf.shape)
    for t in ("regression", "classification"):
        Xtr, Xte, ytr, yte = get_split(t)
        print(t, Xtr.shape, Xte.shape)