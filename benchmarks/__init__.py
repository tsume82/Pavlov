import benchmarks.benchmarks as functions
import benchmarks.CEC2017.cec2017.functions as CEC2017

try:
	import cocoex as COCO
	# very heavy to import:
	COCO.bbob = COCO.Suite("bbob", "", "dimensions: 10")
	# COCO.bbob_biobj = COCO.Suite("bbob-biobj", "", "")
	# COCO.bbob_largescale = COCO.Suite("bbob-largescale", "", "")
	# COCO.bbob_mixint = COCO.Suite("bbob-mixint", "", "")
	# COCO.bbob_biobj_mixint = COCO.Suite("bbob-biobj-mixint", "", "")
except ModuleNotFoundError as err:
	print(err)
	print("Have you built COCO library? Try to run:\n> git submodule update --remote\n> python do.py run-python")