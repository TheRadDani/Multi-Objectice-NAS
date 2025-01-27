from pathlib import Path

import torchx
from torchx import specs
from torchx.components import utils


def trainer(
        log_path: str,
        hidden_size_1: int,
        hidden_size_2: int, 
        learning_rate: float,
        epochs: int,
        dropout: float,
        batch_size: int,
        trial_idx: int = -1,
) -> specs.AppDef:
    
    # define the log path to pass it to TorchX ``AppDef``
    if trial_idx >= 0:
        log_path = Path(log_path).joinpath(str(trial_idx)).absolute().as_posix()
    
    return utils.python(
        # command line arguments to the training script
        "--log_path",
        log_path,
        "--hidden_size_1",
        str(hidden_size_1),
        "--hidden_size_2",
        str(hidden_size_2),
        "--learning_rate",
        str(learning_rate),
        "--epochs",
        str(epochs),
        "--dropout",
        str(dropout),
        "--batch_size",
        str(batch_size),
        name="trainer",
        script="mnist_train_nas.py",
        image=torchx.version.TORCHX_IMAGE,
    )

