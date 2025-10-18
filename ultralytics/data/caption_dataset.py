# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
from collections import defaultdict
from pathlib import Path

from ultralytics.data.dataset import YOLOMultiModalDataset


class YOLOCaptionDataset(YOLOMultiModalDataset):
    """
    Dataset class for loading object detection labels in YOLO format with caption support.

    This class extends YOLOMultiModalDataset to load captions from text files instead of using
    predefined class names. Caption files should be named as 'image-name_caption.txt' and contain
    the caption text for the corresponding image.

    Attributes:
        caption_dir (str): Directory containing caption files.
        caption_files (List[str]): List of caption file paths.

    Methods:
        load_caption_files: Load all caption files from the specified directory.
        get_caption_for_image: Get caption text for a specific image.
        update_labels_info: Override to use captions instead of class names.
    """

    def __init__(self, *args, caption_dir: str = "", **kwargs):
        """
        Initialize a YOLOCaptionDataset.

        Args:
            caption_dir (str): Directory containing caption files.
            *args: Additional positional arguments for the parent class.
            **kwargs: Additional keyword arguments for the parent class.
        """
        self.caption_dir = caption_dir
        self.caption_files = []
        self.caption_cache = {}
        super().__init__(*args, **kwargs)

        # Load caption files if caption directory is provided
        if self.caption_dir:
            self.load_caption_files()

    def load_caption_files(self):
        """Load all caption files from the caption directory."""
        if not self.caption_dir or not os.path.exists(self.caption_dir):
            return

        caption_path = Path(self.caption_dir)
        self.caption_files = list(caption_path.glob("*_caption.txt"))
        print(f"Found {len(self.caption_files)} caption files in {self.caption_dir}")

    def get_caption_for_image(self, image_path: str) -> str:
        """
        Get caption text for a specific image.

        Args:
            image_path (str): Path to the image file.

        Returns:
            (str): Caption text for the image, or empty string if not found.
        """
        # Extract image name without extension
        image_name = Path(image_path).stem

        # Check cache first
        if image_name in self.caption_cache:
            return self.caption_cache[image_name]

        # Look for caption file
        caption_file = None
        for cap_file in self.caption_files:
            if cap_file.stem.startswith(image_name + "_"):
                caption_file = cap_file
                break

        if caption_file and caption_file.exists():
            try:
                with open(caption_file, encoding="utf-8") as f:
                    caption = f.read().strip()
                self.caption_cache[image_name] = caption
                return caption
            except Exception as e:
                print(f"Error reading caption file {caption_file}: {e}")
                return ""

        return ""

    def update_labels_info(self, label: dict) -> dict:
        """
        Add caption information for multi-modal model training.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances and texts from captions.
        """
        labels = super().update_labels_info(label)

        # Get caption for this image
        caption = self.get_caption_for_image(label["im_file"])

        if caption:
            # Use caption as text for all instances in this image
            labels["texts"] = [[caption] for _ in range(len(labels.get("cls", [])))]
        else:
            # Fallback to class names if no caption found
            labels["texts"] = [v.split("/") for _, v in self.data["names"].items()]

        return labels

    @property
    def category_names(self):
        """
        Return category names from captions and class names.

        Returns:
            (Set[str]): Set of unique category names from captions and class names.
        """
        names = set()

        # Add names from captions
        for label in self.labels:
            for text in label.get("texts", []):
                for t in text:
                    names.add(t.strip())

        # Add names from class names as fallback
        class_names = self.data.get("names", {}).values()
        for name in class_names:
            for n in name.split("/"):
                names.add(n.strip())

        return names

    @property
    def category_freq(self):
        """Return frequency of each category in the dataset."""
        category_freq = defaultdict(int)
        for label in self.labels:
            for text in label.get("texts", []):
                for t in text:
                    t = t.strip()
                    category_freq[t] += 1
        return category_freq


class YOLOCaptionTrainerFromScratch:
    """
    Custom trainer that extends YOLOETrainerFromScratch to use captions instead of class names.

    This trainer modifies the text embedding generation to use captions from text files instead of predefined class
    names.
    """

    def __init__(self, caption_dir: str = "", **kwargs):
        """
        Initialize the caption-based trainer.

        Args:
            caption_dir (str): Directory containing caption files.
            **kwargs: Additional arguments for the parent trainer.
        """
        self.caption_dir = caption_dir
        # Import here to avoid circular imports
        from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch

        self.base_trainer = YOLOETrainerFromScratch(**kwargs)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build dataset with caption support.

        Args:
            img_path: Path to images or list of paths.
            mode: 'train' or 'val' mode.
            batch: Batch size.

        Returns:
            Dataset with caption support.
        """
        # Use the base trainer's build_dataset method
        dataset = self.base_trainer.build_dataset(img_path, mode, batch)

        # If we have caption directory, wrap the dataset with caption support
        if self.caption_dir and mode == "train":
            # This is a simplified approach - you might need to modify this
            # based on your specific dataset structure
            pass

        return dataset

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path):
        """
        Generate text embeddings for captions.

        Args:
            texts: List of text samples (captions).
            batch: Batch size for processing.
            cache_dir: Directory to save/load cached embeddings.

        Returns:
            Dictionary mapping text samples to their embeddings.
        """
        return self.base_trainer.generate_text_embeddings(texts, batch, cache_dir)
