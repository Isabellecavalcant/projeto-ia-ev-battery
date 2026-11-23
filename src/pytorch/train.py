from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.optim import Adam, AdamW, SGD, RMSprop


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DEVICE = torch.device("cpu")

MODEL_DIR = Path("models")
EXPERIMENT_DIR = Path("experiments")


# ============================================================
# REPRODUTIBILIDADE
# ============================================================

def set_seed(seed: int = 42):
    """
    Define as seeds das bibliotecas utilizadas.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(
        True,
        warn_only=True
    )


# ============================================================
# FUNÇÕES DE PERDA
# ============================================================

def get_loss_function(
    task: str,
    loss_name: str = "default",
    pos_weight: float | None = None,
):
    """
    Seleciona a função de perda conforme a tarefa.

    Classificação:
        BCEWithLogitsLoss

    Regressão:
        MSELoss ou SmoothL1Loss
    """

    if task == "classification":

        weight = None

        if pos_weight is not None:
            weight = torch.tensor(
                [pos_weight],
                dtype=torch.float32,
                device=DEVICE,
            )

        return nn.BCEWithLogitsLoss(
            pos_weight=weight
        )

    if task == "regression":

        if loss_name == "mse":
            return nn.MSELoss()

        if loss_name == "smooth_l1":
            return nn.SmoothL1Loss()

        if loss_name == "default":
            return nn.MSELoss()

    raise ValueError(
        f"Combinação de tarefa e função de perda inválida: "
        f"{task}, {loss_name}"
    )


# ============================================================
# OTIMIZADORES
# ============================================================

def get_optimizer(
    model: nn.Module,
    optimizer_name: str = "adam",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
):
    """
    Cria o otimizador escolhido.
    """

    if optimizer_name == "adam":
        return Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if optimizer_name == "sgd_momentum":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    if optimizer_name == "rmsprop":
        return RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Otimizador inválido: {optimizer_name}"
    )


# ============================================================
# EXECUÇÃO DE UMA ÉPOCA
# ============================================================

def run_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer=None,
    gradient_clip: float | None = None,
):
    """
    Executa uma época de treinamento ou validação.

    Quando optimizer é informado:
        treinamento.

    Quando optimizer é None:
        avaliação.
    """

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            predictions = model(X_batch)

            loss = criterion(
                predictions,
                y_batch
            )

            if training:

                loss.backward()

                if gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=gradient_clip,
                    )

                optimizer.step()

        current_batch_size = X_batch.size(0)

        total_loss += loss.item() * current_batch_size
        total_samples += current_batch_size

    return total_loss / total_samples


# ============================================================
# TREINAMENTO PRINCIPAL
# ============================================================

def train_model(
    model: nn.Module,
    loaders: dict,
    task: str,
    epochs: int = 100,
    learning_rate: float = 0.001,
    optimizer_name: str = "adam",
    weight_decay: float = 0.0,
    batch_size: int = 256,
    patience: int = 12,
    min_delta: float = 0.0001,
    seed: int = 42,
    loss_name: str = "default",
    pos_weight: float | None = None,
    gradient_clip: float | None = None,
    model_config: dict | None = None,
    experiment_id: str = "baseline",
):
    """
    Treina uma rede neural e salva o melhor checkpoint.

    O melhor modelo é escolhido pela menor perda
    de validação.

    O conjunto de teste não é utilizado nesta função.
    """

    if task not in ["classification", "regression"]:
        raise ValueError("Tarefa inválida.")

    if epochs <= 0:
        raise ValueError("epochs precisa ser positivo.")

    if patience <= 0:
        raise ValueError("patience precisa ser positivo.")

    set_seed(seed)

    model = model.to(DEVICE)

    criterion = get_loss_function(
        task=task,
        loss_name=loss_name,
        pos_weight=pos_weight,
    )

    optimizer = get_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Diretórios
    # --------------------------------------------------------

    checkpoint_dir = MODEL_DIR / task
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_dir = EXPERIMENT_DIR / task
    history_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir / f"{experiment_id}_seed_{seed}.pt"
    )

    history_path = (
        history_dir / f"{experiment_id}_seed_{seed}.csv"
    )

    # --------------------------------------------------------
    # Controle de treinamento
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    # Quantidade de parâmetros treináveis.
    num_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print("\n" + "=" * 65)
    print(f"TREINAMENTO: {task.upper()}")
    print("=" * 65)

    print(f"Experimento: {experiment_id}")
    print(f"Dispositivo: {DEVICE}")
    print(f"Seed: {seed}")
    print(f"Épocas máximas: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Otimizador: {optimizer_name}")
    print(f"Weight decay: {weight_decay}")
    print(f"Batch size: {batch_size}")
    print(f"Early stopping patience: {patience}")
    print(f"Parâmetros treináveis: {num_parameters}")

    # --------------------------------------------------------
    # Loop de épocas
    # --------------------------------------------------------

    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):

        train_loss = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            gradient_clip=gradient_clip,
        )

        val_loss = run_epoch(
            model=model,
            loader=loaders["validation"],
            criterion=criterion,
            optimizer=None,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        improved = (
            val_loss < best_val_loss - min_delta
        )

        if improved:

            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            checkpoint = {
                "task": task,
                "seed": seed,
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": model_config or {},
                "training_config": {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "optimizer": optimizer_name,
                    "weight_decay": weight_decay,
                    "batch_size": batch_size,
                    "patience": patience,
                    "min_delta": min_delta,
                    "loss_name": loss_name,
                    "pos_weight": pos_weight,
                    "gradient_clip": gradient_clip,
                    "seed": seed,
                    "experiment_id": experiment_id,
                },
            }

            torch.save(
                checkpoint,
                checkpoint_path
            )

        else:

            epochs_without_improvement += 1

        print(
            f"Época {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        # Early stopping.
        if epochs_without_improvement >= patience:

            print(
                f"\nEarly stopping na época {epoch}."
            )

            break

    # --------------------------------------------------------
    # Tempo total de treinamento
    # --------------------------------------------------------

    training_time = time.perf_counter() - start_time

    # --------------------------------------------------------
    # Restaurar melhor modelo
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # --------------------------------------------------------
    # Salvar histórico
    # --------------------------------------------------------

    history_df = pd.DataFrame(history)

    best_row = history_df.loc[
        history_df["epoch"] == best_epoch
    ].iloc[0]

    history_df.to_csv(
        history_path,
        index=False
    )

    # --------------------------------------------------------
    # Resultados
    # --------------------------------------------------------

    train_val_gap = (
        best_row["val_loss"] - best_row["train_loss"]
    )

    print("\nTreinamento finalizado.")

    print(f"Melhor época: {best_epoch}")
    print(f"Melhor perda de treinamento: {best_row['train_loss']:.6f}")
    print(f"Melhor perda de validação: {best_val_loss:.6f}")
    print(f"Gap treino-validação: {train_val_gap:.6f}")
    print(f"Parâmetros treináveis: {num_parameters}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Histórico: {history_path}")

    # --------------------------------------------------------
    # Retorno dos resultados
    # --------------------------------------------------------

    return {
        "model": model,
        "history": history_df,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "checkpoint_path": checkpoint_path,
        "history_path": history_path,
        "seed": seed,
        "training_time": training_time,
        "num_parameters": num_parameters,
        "best_train_loss": best_row["train_loss"],
        "train_val_gap": train_val_gap,
    }


# ============================================================
# TESTE DO MÓDULO
# ============================================================

if __name__ == "__main__":

    from src.pytorch.preprocessing import prepare_data
    from src.pytorch.dataset import create_dataloaders
    from src.pytorch.models import MLPClassifier, MLPRegressor

    # --------------------------------------------------------
    # Testar classificação
    # --------------------------------------------------------

    data = prepare_data("classification")

    loaders = create_dataloaders(
        data,
        batch_size=256,
        seed=42,
    )

    model_config = {
        "input_dim": len(data.feature_names),
        "hidden_dims": [128, 64],
        "activation": "relu",
        "dropout": 0.2,
        "batch_norm": False,
    }

    # Definir a seed antes da criação do modelo.
    set_seed(42)

    model = MLPClassifier(**model_config)

    result = train_model(
        model=model,
        loaders=loaders,
        task="classification",
        epochs=5,
        learning_rate=0.001,
        optimizer_name="adam",
        batch_size=256,
        patience=3,
        seed=42,
        model_config=model_config,
        experiment_id="teste_classificacao",
    )

    print("\nTeste de treinamento concluído.")

    print(f"Tempo: {result['training_time']:.2f} segundos")
    print(f"Parâmetros: {result['num_parameters']}")