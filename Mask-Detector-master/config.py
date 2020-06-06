from collections import OrderedDict

class Config:
	use_run_setup = False
	resume = False
	run_setup = OrderedDict({
		"lr": [0.01, 0.001],
		"num_epochs": [10]
	})

	device = "cuda"
	num_epochs = 4
	lr = 0.001
	batch_size = 2
	test_batch_size = 1
	workers = 0