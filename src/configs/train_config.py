from pathlib import Path


class TrainConfig:
    """Class defining the train configurations"""

    # path related params
    SOURCE_DIR: str = "/home/jataayu/Indi_Yolo/Indi-Yolo/finetuning"
    """Path to the source directory"""
    DATASET_ROOT: str = "./vehicle_dataset"
    """Path to the custom vehicle dataset"""

    # model related params
    MODEL: str = "yolo26s-cls.pt"
    """Path belonging to the YOLO-MODEL variant"""
    IMG_SIZE: int = 224
    """Size of the image"""

    # HyperParameters
    EPOCHS: int = 50
    """Number of epochs in the training loop"""
    BATCH_SIZE: int = 32
    SPLIT_RATIO: float = 0.8
    """Train-Test split ratio"""
    LR: float = 0.001
    """learning rate for the training epoch"""

    # wandb params
    PROJECT_NAME: str = "Indi-Yolo"
