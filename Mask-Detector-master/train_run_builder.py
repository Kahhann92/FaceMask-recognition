import numpy as np
import pandas as pd
import argparse
from argparse import Namespace
from collections import OrderedDict
import time
import os
import copy
import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from pytorch_detection.engine import train_one_epoch, evaluate

from run_builder import RunBuilder
from data import DataHandler
from config import Config
import utils
from models import model_rcnn
import cv2

torch.manual_seed(0)
np.random.seed(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


def train(cfg) -> None:
	device = torch.device(cfg.device)
	print(f"Using device {device}")

	runs = None
	if cfg.use_run_setup == True:
		runs = RunBuilder.get_runs(cfg.run_setup)
	else:
		runs = RunBuilder.get_runs(
			OrderedDict({
				"lr": [cfg.lr],
				"num_epochs": [cfg.num_epochs]
			})
		)
	assert runs != None


	data_handler = DataHandler(cfg)
	train_dataset, validation_dataset = data_handler.get_datasets()
	train_loader, validation_loader = data_handler.get_data_loaders()
	training_dataset_size, validation_dataset_size = data_handler.get_datasets_sizes()


	best_model_wts = None
	best_acc = 0.0
	best_config = None

	for run in runs:
		comment = f"Run setup -- {run}"
		print(comment)

		model = model_rcnn.create_model(num_classes=3)
		model.to(device)
		params = [p for p in model.parameters() if p.requires_grad]
		optimizer = optim.Adam(params, lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

		# Check if resume
		if cfg.use_run_setup == False and cfg.resume == True:
			checkpoint = torch.load("./checkpoints/ckp.pt")
			model.load_state_dict(checkpoint["model_state_dict"])
			optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
			run.num_epochs -= checkpoint["epoch"]

		# loss_criterion = ...

		log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
		writer = SummaryWriter(log_dir)

		since = time.time()
		for epoch in range(run.num_epochs):
			print('Epoch {}/{}'.format(epoch, run.num_epochs))
			print('-' * 10)

			# train for one epoch, printing every 10 iterations
			train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)
			evaluate(model, validation_loader, device)

		time_elapsed = time.time() - since
		print('Training complete in {:.0f}m {:.0f}s'.format(
			time_elapsed // 60, time_elapsed % 60
		))

	print(f"Best configuration: {best_config}")
	return model


def main(args: Namespace) -> None:
	trained_model = train(Config)
	torch.save(trained_model.state_dict(), f"model.pt")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	args = parser.parse_args()
	main(args)