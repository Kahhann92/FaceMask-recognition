from collections import OrderedDict
from collections import namedtuple
from itertools import product

class RunBuilder():
	@staticmethod
	def get_runs(params: OrderedDict):
		Run = namedtuple("Run", params.keys())

		runs = []
		for values in product(*params.values()):
			runs.append(Run(*values))

		return runs