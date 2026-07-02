from src.configs.train_config import TrainConfig
from pathlib import Path
import os
import shutil
import random

class VehicleDataManager:
    """
    Class prepares the IISERB Vehicle Dataset in the YOLO Training Format
    Image name : "AB4_20260204_132137_frame026910_bus_0.56"
    where the first part shows the camera of IISER-B it is captured, date and time, frame number, vehilce class and a confidence score.

    This Dataset comes under the ownership of IISERB.

    """

    def __init__(self, config) -> None:
        self._cfg=config
        self._source_path: Path= Path(self.cfg.SOURCE_DIR)
        self._dest_path : Path = Path(self.cfg.DATASET_ROOT)

    def extract_labels(self,file_path)->[str,None]:
        """
        function extracts the label from the image paths.
        Input: AB4_20260204_132137_frame026910_bus_0.56
        Output: bus
        """
        try:
            return filepath.stem.split("_")[-2]

        except IndexError:
            return None

    def prepare(self)->None:
        if self.dest.exists():

            shutil.rmtree(self.dest)

        images = [f for f in self.source.iterdir() if f.suffix.lower() in ['.jpg','.png','.jpeg']]
        random.shuffle(images)

        split_idx = int(len(images)*self._cfg.SPLIT_RATIO)

        dataset_splits = {
            "train":images[:split_idx],
            "val": images[split_idx:],
        }

        for split, files in dataset_splits.items():
            for f in files:
                label = self._extract_label(f)
                if label:
                    target_dir = self.dest / split / label
                    target_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, target_dir / f.name)
        
        logging.info(f"Dataset ready at {self.dest}")




