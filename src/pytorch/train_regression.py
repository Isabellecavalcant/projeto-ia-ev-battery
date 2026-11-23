from src.pytorch.preprocessing import prepare_data
from src.pytorch.dataset import create_dataloaders
from src.pytorch.models import MLPRegressor
from src.pytorch.train import train_model, set_seed


def main():

    seed = 42

    # Definir seed antes de inicializar a rede.
    set_seed(seed)

    # --------------------------------------------------------
    # 1. Preparar dados
    # --------------------------------------------------------

    data = prepare_data("regression")

    loaders = create_dataloaders(
        data,
        batch_size=256,
        seed=seed,
    )

    # --------------------------------------------------------
    # 2. Configurar arquitetura
    # --------------------------------------------------------

    model_config = {
        "input_dim": len(data.feature_names),
        "hidden_dims": [128, 64],
        "activation": "relu",
        "dropout": 0.2,
        "batch_norm": False,
    }

    model = MLPRegressor(**model_config)

    # --------------------------------------------------------
    # 3. Treinar
    # --------------------------------------------------------

    result = train_model(
        model=model,
        loaders=loaders,
        task="regression",
        epochs=5,
        learning_rate=0.001,
        optimizer_name="adam",
        weight_decay=0.0,
        batch_size=256,
        patience=3,
        seed=seed,
        loss_name="mse",
        model_config=model_config,
    )

    print("\n" + "=" * 60)
    print("TREINAMENTO DE REGRESSÃO CONCLUÍDO")
    print("=" * 60)

    print("Melhor época:", result["best_epoch"])
    print("Melhor perda de validação:", result["best_val_loss"])
    print("Checkpoint:", result["checkpoint_path"])


if __name__ == "__main__":
    main()
