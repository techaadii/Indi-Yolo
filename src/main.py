from src.configs.train_config import TrainConfig
from src.data.data_manager import VehicleDataManager
from src.yolo.yolo_trainer import VehicleDataTrainer




def main()->None:
    """Main function to initiate the training loop"""
    cfg = TrainConfig()

    # 1. Prepare Folders
    dm = VehicleDataManager(cfg)
    dm.prepare()

    # 2. Train and Log
    trainer = VehicleDataTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
