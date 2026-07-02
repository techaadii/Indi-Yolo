import os
import shutil
import random
import logging
from pathlib import Path
# Assuming your folder structure is src/data_manager.py, 
# then configs is at the same level as data_manager inside src/
from src.configs.train_config import TrainConfig 

class VehicleDataManager:
    """
    Class prepares the IISERB Vehicle Dataset in the YOLO Training Format.
    """

    def __init__(self, config: TrainConfig) -> None:
        self._cfg = config
        # Use the names consistently
        self._source_path = Path(self._cfg.SOURCE_DIR)
        self._dest_path = Path(self._cfg.DATASET_ROOT)

    def extract_labels(self, file_path: Path) -> str | None:
        """
        Extracts the label from the image paths.
        Input: AB4_20260204_132137_frame026910_bus_0.56
        Output: bus
        """
        try:
            # Fix: match the argument name 'file_path'
            return file_path.stem.split("_")[-2]
        except (IndexError, AttributeError):
            return None

    def prepare(self) -> None:
        # Fix: use self._dest_path
        if self._dest_path.exists():
            logging.info(f"Cleaning existing directory: {self._dest_path}")
            shutil.rmtree(self._dest_path)

        # Fix: use self._source_path
        if not self._source_path.exists():
            raise FileNotFoundError(f"Source directory not found: {self._source_path}")

        images = [f for f in self._source_path.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        
        if not images:
            logging.warning("No images found in source directory!")
            return

        random.shuffle(images)

        split_idx = int(len(images) * self._cfg.SPLIT_RATIO)

        dataset_splits = {
            "train": images[:split_idx],
            "val": images[split_idx:],
        }

        for split, files in dataset_splits.items():
            for f in files:
                # Fix: match the method name 'extract_labels'
                label = self.extract_labels(f)
                if label:
                    target_dir = self._dest_path / split / label
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, target_dir / f.name)
        
        logging.info(f"Dataset ready at {self._dest_path}")