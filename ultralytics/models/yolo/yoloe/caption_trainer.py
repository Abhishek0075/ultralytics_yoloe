# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import itertools
import os
from pathlib import Path
from typing import Optional, Union

import torch

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import de_parallel

from ..world.train_world import WorldTrainerFromScratch


class YOLOECaptionDataset:
    """
    Custom dataset class for loading captions from text files.

    This class handles loading captions from files named 'image-name_caption.txt' and provides them for text embedding
    generation.
    """

    def __init__(self, caption_dir: str = ""):
        """
        Initialize the caption dataset.

        Args:
            caption_dir (str): Directory containing caption files.
        """
        self.caption_dir = caption_dir
        self.caption_cache = {}
        self.caption_files = []

        if self.caption_dir and os.path.exists(self.caption_dir):
            self.load_caption_files()

    def load_caption_files(self):
        """Load all caption files from the caption directory."""
        caption_path = Path(self.caption_dir)
        self.caption_files = list(caption_path.glob("*_caption.txt"))
        LOGGER.info(f"Found {len(self.caption_files)} caption files in {self.caption_dir}")

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
                LOGGER.warning(f"Error reading caption file {caption_file}: {e}")
                return ""

        return ""

    def get_all_captions(self) -> list[str]:
        """
        Get all unique captions from the dataset.

        Returns:
            (List[str]): List of unique captions.
        """
        captions = set()
        for cap_file in self.caption_files:
            try:
                with open(cap_file, encoding="utf-8") as f:
                    caption = f.read().strip()
                    if caption:
                        captions.add(caption)
            except Exception as e:
                LOGGER.warning(f"Error reading caption file {cap_file}: {e}")

        return list(captions)


class YOLOECaptionTrainerFromScratch(WorldTrainerFromScratch):
    """
    Train YOLOE models from scratch with caption-based text embeddings.

    This trainer extends YOLOETrainerFromScratch to use captions from text files
    instead of class names for text embedding generation.

    Attributes:
        caption_dataset (YOLOECaptionDataset): Dataset for loading captions.
        caption_dir (str): Directory containing caption files.

    Methods:
        build_dataset: Build datasets with caption support.
        preprocess_batch: Process batches with caption-based text features.
        generate_text_embeddings: Generate text embeddings from captions.
        set_text_embeddings: Set text embeddings using captions.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: Optional[dict] = None, _callbacks=None, caption_dir: str = ""):
        """
        Initialize the YOLOE Caption Trainer.

        Args:
            cfg: Configuration dictionary.
            overrides: Parameter overrides.
            _callbacks: List of callback functions.
            caption_dir (str): Directory containing caption files.
        """
        if overrides is None:
            overrides = {}
        overrides["overlap_mask"] = False

        self.caption_dir = caption_dir
        self.caption_dataset = YOLOECaptionDataset(caption_dir) if caption_dir else None

        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path: Union[list[str], str], mode: str = "train", batch: Optional[int] = None):
        """
        Build YOLO Dataset for training or validation with caption support.

        Args:
            img_path (List[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
            batch (int, optional): Size of batches, used for rectangular training/validation.

        Returns:
            (YOLOConcatDataset | Dataset): The constructed dataset for training or validation.
        """
        # Use the parent's build_dataset method
        dataset = super().build_dataset(img_path, mode, batch)

        # If we have caption support and it's training mode, we'll handle captions in preprocess_batch
        if self.caption_dataset and mode == "train":
            LOGGER.info("Caption-based training enabled. Captions will be loaded from text files.")

        return dataset

    def preprocess_batch(self, batch):
        """
        Process batch for training, using captions for text features.

        Args:
            batch: Batch of data to process.

        Returns:
            Processed batch with caption-based text features.
        """
        batch = DetectionTrainer.preprocess_batch(self, batch)

        if self.caption_dataset:
            # Get captions for all images in the batch
            captions = []
            for img_path in batch.get("im_file", []):
                caption = self.caption_dataset.get_caption_for_image(img_path)
                if not caption:
                    # Fallback to class names if no caption found
                    caption = " ".join(self.data["names"].values())
                captions.append(caption)

            # Generate text embeddings for captions
            if hasattr(self, "text_embeddings") and self.text_embeddings:
                # Use cached embeddings if available
                txt_feats = torch.stack(
                    [
                        self.text_embeddings.get(caption, self.text_embeddings[list(self.text_embeddings.keys())[0]])
                        for caption in captions
                    ]
                ).to(self.device)
            else:
                # Generate embeddings on the fly (less efficient)
                if self.model is not None:
                    txt_feats = de_parallel(self.model).get_text_pe(
                        captions, len(captions), without_reprta=True, cache_clip_model=False
                    )
                    txt_feats = txt_feats.squeeze(0)
                else:
                    # Fallback: use dummy embeddings
                    txt_feats = torch.randn(len(captions), 512).to(self.device)

            batch["txt_feats"] = txt_feats.reshape(1, -1, txt_feats.shape[-1])
        else:
            # Use the original text processing
            texts = list(itertools.chain(*batch["texts"]))
            txt_feats = torch.stack([self.text_embeddings[text] for text in texts]).to(self.device)
            txt_feats = txt_feats.reshape(len(batch["texts"]), -1, txt_feats.shape[-1])
            batch["txt_feats"] = txt_feats

        return batch

    def generate_text_embeddings(self, texts: list[str], batch: int, cache_dir: Path):
        """
        Generate text embeddings for captions.

        Args:
            texts (List[str]): List of text samples (captions).
            batch (int): Batch size for processing.
            cache_dir (Path): Directory to save/load cached embeddings.

        Returns:
            (dict): Dictionary mapping text samples to their embeddings.
        """
        model = "mobileclip:blt"
        cache_path = cache_dir / f"caption_embeddings_{model.replace(':', '_').replace('/', '_')}.pt"

        if cache_path.exists():
            LOGGER.info(f"Reading existing caption cache from '{cache_path}'")
            txt_map = torch.load(cache_path, map_location=self.device)
            if sorted(txt_map.keys()) == sorted(texts):
                return txt_map

        LOGGER.info(f"Caching caption embeddings to '{cache_path}'")
        assert self.model is not None
        txt_feats = de_parallel(self.model).get_text_pe(texts, batch, without_reprta=True, cache_clip_model=False)
        txt_map = dict(zip(texts, txt_feats.squeeze(0)))
        torch.save(txt_map, cache_path)
        return txt_map

    def set_text_embeddings(self, datasets, batch: int):
        """
        Set text embeddings for datasets using captions.

        Args:
            datasets (List[Dataset]): List of datasets containing captions to process.
            batch (int): Batch size for processing text embeddings.
        """
        if not self.caption_dataset:
            # Fallback to original method
            super().set_text_embeddings(datasets, batch)
            return

        # Collect all unique captions from the dataset
        all_captions = set()

        # Get captions from caption files
        if self.caption_dataset.caption_files:
            all_captions.update(self.caption_dataset.get_all_captions())

        # Also collect from dataset labels if available
        for dataset in datasets:
            if hasattr(dataset, "labels"):
                for label in dataset.labels:
                    if "im_file" in label:
                        caption = self.caption_dataset.get_caption_for_image(label["im_file"])
                        if caption:
                            all_captions.add(caption)

        if not all_captions:
            LOGGER.warning("No captions found. Falling back to class names.")
            super().set_text_embeddings(datasets, batch)
            return

        all_captions = list(all_captions)
        LOGGER.info(f"Found {len(all_captions)} unique captions for text embedding generation")

        # Generate cache directory
        cache_dir = Path(self.args.project) / "caption_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Generate text embeddings for captions
        text_embeddings = self.generate_text_embeddings(all_captions, batch, cache_dir)
        self.text_embeddings = text_embeddings

        LOGGER.info(f"Generated text embeddings for {len(text_embeddings)} captions")
