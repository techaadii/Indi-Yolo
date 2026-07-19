import shutil
import random
import logging
from pathlib import Path


class VehicleDataManager:
    def __init__(self, config) -> None:
        self._cfg = config
        self._source_path = Path(self._cfg.SOURCE_DIR)
        self._dest_path = Path(self._cfg.DATASET_ROOT)

    def extract_label(self, file_path: Path) -> str | None:
        try:
            # Splits: AB4_..._bus_0.56.jpg -> bus
            return file_path.stem.split("_")[-2]
        except IndexError, AttributeError:
            return None

    def prepare(self) -> None:
        if self._dest_path.exists():
            shutil.rmtree(self._dest_path)

        # 1. Group all images by their label first
        all_files = [
            f
            for f in self._source_path.iterdir()
            if f.suffix.lower() in [".jpg", ".png", ".jpeg"]
        ]
        label_groups = {}

        for f in all_files:
            label = self.extract_label(f)
            if label:
                if label not in label_groups:
                    label_groups[label] = []
                label_groups[label].append(f)

        logging.info(f"Found {len(label_groups)} unique classes.")

        # 2. Split each class individually (Stratified Split)
        for label, files in label_groups.items():
            random.shuffle(files)

            # Ensure at least 1 image goes to validation if class is tiny
            split_idx = int(len(files) * self._cfg.SPLIT_RATIO)
            if split_idx == len(files) and len(files) > 1:
                split_idx -= 1

            train_files = files[:split_idx]
            val_files = files[split_idx:]

            # If val_files is empty, move one from train to val
            if not val_files and len(train_files) > 0:
                val_files.append(train_files.pop())

            # 3. Copy files
            self._copy_set(train_files, "train", label)
            self._copy_set(val_files, "val", label)

        logging.info(f"Stratified dataset prepared at {self._dest_path}")

    def _copy_set(self, files, subset, label):
        target_dir = self._dest_path / subset / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, target_dir / f.name)
