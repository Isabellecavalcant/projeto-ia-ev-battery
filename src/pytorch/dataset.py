import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from src.pytorch.preprocessing import PreparedData


# ============================================================
# DATASET
# ============================================================

class BatteryDataset(Dataset):
    """
    Dataset de baterias para classificação ou regressão.

    Recebe os dados já processados e os converte
    em tensores PyTorch.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):

        if len(X) != len(y):
            raise ValueError(
                "X e y precisam possuir a mesma quantidade de registros."
            )

        self.X = torch.from_numpy(
            np.ascontiguousarray(X, dtype=np.float32)
        )

        self.y = torch.from_numpy(
            np.ascontiguousarray(y, dtype=np.float32)
        )

    def __len__(self):
        """
        Retorna a quantidade de exemplos.
        """
        return len(self.X)

    def __getitem__(self, index):
        """
        Retorna um exemplo e seu respectivo alvo.
        """
        return self.X[index], self.y[index]


# ============================================================
# DATALOADERS
# ============================================================

def create_dataloaders(
    data: PreparedData,
    batch_size: int = 256,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Cria os DataLoaders de treinamento, validação e teste.

    Treinamento:
        shuffle=True

    Validação e teste:
        shuffle=False

    num_workers=0:
        Evita processos adicionais e reduz a sobrecarga
        de memória neste ambiente.
    """

    if batch_size <= 0:
        raise ValueError("batch_size precisa ser maior que zero.")

    # --------------------------------------------------------
    # 1. Criar datasets
    # --------------------------------------------------------

    train_dataset = BatteryDataset(
        data.X_train,
        data.y_train
    )

    val_dataset = BatteryDataset(
        data.X_val,
        data.y_val
    )

    test_dataset = BatteryDataset(
        data.X_test,
        data.y_test
    )

    # --------------------------------------------------------
    # 2. Gerador reprodutível para embaralhamento
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(seed)

    # --------------------------------------------------------
    # 3. Criar DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader,
    }


# ============================================================
# TESTE DO MÓDULO
# ============================================================

if __name__ == "__main__":

    from src.pytorch.preprocessing import prepare_data

    for task in ["classification", "regression"]:

        print("\n" + "=" * 60)
        print(f"TESTANDO DATASET: {task.upper()}")
        print("=" * 60)

        prepared_data = prepare_data(task)

        loaders = create_dataloaders(
            prepared_data,
            batch_size=256,
            seed=42,
        )

        for name, loader in loaders.items():

            print(f"\n{name.upper()}")
            print(f"Quantidade de exemplos: {len(loader.dataset)}")
            print(f"Quantidade de batches: {len(loader)}")

            # Examinar somente o primeiro batch.
            X_batch, y_batch = next(iter(loader))

            print(f"Formato de X: {tuple(X_batch.shape)}")
            print(f"Formato de y: {tuple(y_batch.shape)}")

            print(f"Tipo de X: {X_batch.dtype}")
            print(f"Tipo de y: {y_batch.dtype}")

            print(f"Dispositivo: {X_batch.device}")