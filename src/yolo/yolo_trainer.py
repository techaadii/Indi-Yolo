from ultralytics import YOLO
import wandb
from pathlib import Path

class VehicleDataTrainer:
    """This class defines the training setup for Indi-YOLo"""
    def __init__(self,config) -> None:
        self.cfg=config
        self.model : YOLO = YOLO(self.cfg.MODEL)


    def train(self):
        wandb.init(
            project=self.cfg.PROJECT_NAME,
            config={
                'learning_rate':self.cfg.LR,
                'epochs': self.cfg.EPOCHS,
                'batch size': self.cfg.BATCH_SIZE,
            })


        self.model.train(
            data=str(Path(self.cfg.DATASET_ROOT).absolute()),
            epochs=self.cfg.EPOCHS,
            imgsz=self.cfg.IMG_SIZE,
            batch=self.cfg.BATCH_SIZE,
            lr0=self.cfg.LR,
            project=self.cfg.PROJECT_NAME,
            
            device="cuda"  
        )
        wandb.finish()
        
