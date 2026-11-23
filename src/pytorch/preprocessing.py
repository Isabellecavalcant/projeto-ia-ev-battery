from dataclasses import dataclass
from pathlib import Path
import joblib

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# CONFIGURAÇÕES
# ============================================================

RANDOM_STATE = 42

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")

TARGETS = {
    "classification": "battery_failure",
    "regression": "predicted_remaining_life_cycles",
}


# ============================================================
# ESTRUTURA DOS DADOS PREPARADOS
# ============================================================

@dataclass
class PreparedData:
    """
    Armazena os dados preparados para treinamento,
    validação e teste.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    # Valores originais do alvo.
    # São especialmente importantes para avaliar a regressão
    # na escala original do problema.
    y_train_original: np.ndarray
    y_val_original: np.ndarray
    y_test_original: np.ndarray

    preprocessor: ColumnTransformer
    target_scaler: StandardScaler | None

    feature_names: list[str]
    task: str
    target_name: str


# ============================================================
# CARREGAMENTO
# ============================================================

def load_dataset(task: str) -> pd.DataFrame:
    """
    Carrega o dataset correspondente à tarefa.

    task:
        classification ou regression
    """

    if task not in TARGETS:
        raise ValueError(
            "Tarefa inválida. Utilize 'classification' ou 'regression'."
        )

    filename = f"dataset_{task}.csv"
    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {filepath}"
        )

    return pd.read_csv(filepath)


# ============================================================
# CONSTRUÇÃO DO PRÉ-PROCESSADOR
# ============================================================

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constrói o pipeline de pré-processamento.

    Variáveis numéricas:
        - imputação pela mediana;
        - padronização com StandardScaler.

    Variáveis categóricas:
        - imputação pela categoria mais frequente;
        - One-Hot Encoding.
    """

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float32
            )),
        ]
    )

    transformers = []

    if numeric_columns:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.0,
    )

    return preprocessor


# ============================================================
# PREPARAÇÃO DOS DADOS
# ============================================================

def prepare_data(
    task: str,
    scale_regression_target: bool = True,
) -> PreparedData:
    """
    Prepara os dados para uma tarefa de classificação
    ou regressão.

    Divisão:
        60% treinamento
        20% validação
        20% teste

    O conjunto de teste permanece isolado durante
    o treinamento e a seleção de hiperparâmetros.
    """

    if task not in TARGETS:
        raise ValueError(
            "Tarefa inválida. Utilize 'classification' ou 'regression'."
        )

    print(f"\nPreparando dados: {task}")

    # --------------------------------------------------------
    # 1. Carregar dataset
    # --------------------------------------------------------

    df = load_dataset(task)

    target_name = TARGETS[task]

    if target_name not in df.columns:
        raise ValueError(
            f"Coluna alvo '{target_name}' não encontrada."
        )

    # Remover linhas sem valor no alvo.
    df = df.dropna(subset=[target_name]).copy()

    X = df.drop(columns=[target_name])
    y = df[target_name]

    # --------------------------------------------------------
    # 2. Definir estratificação para classificação
    # --------------------------------------------------------

    stratify = y if task == "classification" else None

    # --------------------------------------------------------
    # 3. Separar teste: 20%
    # --------------------------------------------------------

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    # --------------------------------------------------------
    # 4. Separar treinamento e validação
    #
    # Dos 80% restantes:
    # 75% para treinamento = 60% do total.
    # 25% para validação = 20% do total.
    # --------------------------------------------------------

    stratify_train_val = (
        y_train_val if task == "classification" else None
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=stratify_train_val,
    )

    # --------------------------------------------------------
    # 5. Construir e ajustar pré-processador
    # --------------------------------------------------------

    preprocessor = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)

    X_val_processed = preprocessor.transform(X_val)

    X_test_processed = preprocessor.transform(X_test)

    # Converter para float32 para reduzir consumo de memória.
    X_train_processed = np.asarray(
        X_train_processed,
        dtype=np.float32
    )

    X_val_processed = np.asarray(
        X_val_processed,
        dtype=np.float32
    )

    X_test_processed = np.asarray(
        X_test_processed,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # 6. Preparar variável alvo
    # --------------------------------------------------------

    target_scaler = None

    y_train_original = np.asarray(
        y_train,
        dtype=np.float32
    ).reshape(-1, 1)

    y_val_original = np.asarray(
        y_val,
        dtype=np.float32
    ).reshape(-1, 1)

    y_test_original = np.asarray(
        y_test,
        dtype=np.float32
    ).reshape(-1, 1)

    if task == "regression" and scale_regression_target:

        target_scaler = StandardScaler()

        y_train_processed = target_scaler.fit_transform(
            y_train_original
        ).astype(np.float32)

        y_val_processed = target_scaler.transform(
            y_val_original
        ).astype(np.float32)

        y_test_processed = target_scaler.transform(
            y_test_original
        ).astype(np.float32)

    else:

        y_train_processed = y_train_original.copy()
        y_val_processed = y_val_original.copy()
        y_test_processed = y_test_original.copy()

    # --------------------------------------------------------
    # 7. Obter nomes das features após transformação
    # --------------------------------------------------------

    feature_names = preprocessor.get_feature_names_out().tolist()

    # --------------------------------------------------------
    # 8. Salvar artefatos de pré-processamento
    # --------------------------------------------------------

    artifact_dir = MODEL_DIR / task
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "preprocessor": preprocessor,
        "target_scaler": target_scaler,
        "feature_names": feature_names,
        "target_name": target_name,
        "task": task,
    }

    artifact_path = artifact_dir / "preprocessing.joblib"

    joblib.dump(artifact, artifact_path)

    # --------------------------------------------------------
    # 9. Exibir informações
    # --------------------------------------------------------

    print("\nDivisão dos dados:")
    print(f"Treinamento: {X_train_processed.shape}")
    print(f"Validação:   {X_val_processed.shape}")
    print(f"Teste:       {X_test_processed.shape}")

    print(f"\nQuantidade de features: {len(feature_names)}")

    print("\nDistribuição do alvo:")

    if task == "classification":

        print(
            "Treinamento:",
            pd.Series(y_train).value_counts(normalize=True).to_dict()
        )

        print(
            "Validação:",
            pd.Series(y_val).value_counts(normalize=True).to_dict()
        )

        print(
            "Teste:",
            pd.Series(y_test).value_counts(normalize=True).to_dict()
        )

    else:

        print(
            "Média do alvo no treinamento:",
            round(float(y_train_original.mean()), 2)
        )

        print(
            "Desvio padrão do alvo no treinamento:",
            round(float(y_train_original.std()), 2)
        )

    print(f"\nArtefatos salvos em: {artifact_path}")

    # --------------------------------------------------------
    # 10. Retornar estrutura preparada
    # --------------------------------------------------------

    return PreparedData(
        X_train=X_train_processed,
        X_val=X_val_processed,
        X_test=X_test_processed,

        y_train=y_train_processed,
        y_val=y_val_processed,
        y_test=y_test_processed,

        y_train_original=y_train_original,
        y_val_original=y_val_original,
        y_test_original=y_test_original,

        preprocessor=preprocessor,
        target_scaler=target_scaler,

        feature_names=feature_names,
        task=task,
        target_name=target_name,
    )


# ============================================================
# EXECUÇÃO DIRETA PARA TESTES
# ============================================================

if __name__ == "__main__":

    classification_data = prepare_data("classification")

    regression_data = prepare_data("regression")