import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json
import pathlib
from PIL import Image
import os

from pytorch_detection.utils import collate_fn
from pytorch_detection import utils
from pytorch_detection import transforms as T

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, models, transforms
import pytorch_detection.transforms as T 
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

device = torch.device("cuda")

class DataHandler:
	def __init__(self, run_config):
		self._training_dataset = None
		self._validation_dataset = None
		self._run_config = run_config
		self.root_path = "./dataset"

		self.load_datasets()
		
	def load_datasets(self):
		masked_faces_paths = list(pathlib.Path(self.root_path + "/masked_faces").glob('*'))
		normal_faces_paths = list(pathlib.Path(self.root_path + "/normal_faces").glob('*'))

		ratio = 90
		length = len(masked_faces_paths)
		training_size = int(0.8 * length)

		training_paths = masked_faces_paths[:training_size] + normal_faces_paths[:training_size]
		validation_paths = masked_faces_paths[training_size:] + normal_faces_paths[training_size:]

		self._training_dataset = CustomDataset(training_paths, root_path="./dataset", run_type="train")
		self._validation_dataset = CustomDataset(validation_paths, root_path="./dataset", run_type="valid")

	def get_data_loaders(self) -> Tuple[DataLoader]:
		return (
			DataLoader(
				self._training_dataset, 
				batch_size=self._run_config.batch_size, 
				shuffle=True, 
				num_workers=self._run_config.workers, 
				pin_memory=False,
				collate_fn=collate_fn
			), 
			DataLoader(
				self._validation_dataset, 
				batch_size=self._run_config.batch_size, 
				shuffle=True, 
				num_workers=self._run_config.workers, 
				pin_memory=False,
				collate_fn=collate_fn
			)
		)

	def get_datasets(self) -> Tuple[Dataset]:
		return self._training_dataset, self._validation_dataset

	def get_datasets_sizes(self) -> Tuple[int]:
		return len(self._training_dataset), len(self._validation_dataset)


class CustomDataset(Dataset):
	def __init__(self, image_paths, root_path = "./dataset", run_type = "train"):
		self.root_path = root_path
		self.run_type = run_type
		with open(self.root_path + "/targets.json") as json_file:
			self.targets = json.load(json_file)

		self.images_paths = image_paths

		self.transformers = {
			'train_transforms' : T.Compose([
				# transforms.RandomHorizontalFlip(0.5),
				T.ToTensor()
			]),
			'valid_transforms' : T.Compose([
				T.ToTensor()
			])
		}


	def __getitem__(self, idx):
		image_path = self.images_paths[idx]
		image_name = str(image_path).split(os.sep)[2]
		image_class = str(image_path).split(os.sep)[1]

		class_to_label = {
			"masked_faces": 1,
			"normal_faces": 2
		}

		label = class_to_label[image_class]
		image = Image.open(image_path).convert("RGB")


		target = {}
		if len(self.targets[image_name]["bbox"]) == 0:
			self.targets[image_name]["bbox"] = [[]]
		target["boxes"] = torch.as_tensor(self.targets[image_name]["bbox"], dtype=torch.float32).to(device)
		target["labels"] = torch.as_tensor([label] * len(self.targets[image_name]["bbox"]), dtype=torch.int64).to(device)
		target["iscrowd"] = torch.tensor([0] * len(self.targets[image_name]["bbox"]), dtype=torch.int64).to(device)
		target["image_id"] = torch.tensor([idx]).to(device)

		area = (
			target["boxes"][:, 3] - \
			target["boxes"][:, 1]) * \
			(target["boxes"][:, 2] - \
			target["boxes"][:, 0]
		)

		target["area"] = torch.tensor(area)

		if self.transformers is not None:
			image, target = self.transformers[self.run_type + "_transforms"](image, target)
		return image, target


	def __len__(self):
		return len(self.images_paths)


def get_aug(aug, min_area=0., min_visibility=0.):
	return Compose(
		aug, 
		bbox_params=BboxParams(
			format='pascal_voc', 
			min_area=min_area, 
			min_visibility=min_visibility, 
			label_fields=['category_id']
		)
	)