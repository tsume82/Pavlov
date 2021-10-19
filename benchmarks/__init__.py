import benchmarks.benchmarks as functions
import benchmarks.CEC2017.cec2017.functions as CEC2017

try:
	import cocoex as COCO
	# info about the functions: https://hal.inria.fr/inria-00362633v2/document
except ModuleNotFoundError as err:
	print(err)
	print("Have you built COCO library? Try to run:\n> git submodule update --remote\n> cd benchmarks/COCO\n> python do.py run-python")

# example using COCO:
# suite = COCO.Suite("bbob", "", "dimensions: 10, function_indices: 2")
# problem = suite[0]
# fitness = problem(x)
# first parameter options:
# 'bbob', 'bbob-biobj', 'bbob-largescale', 'bbob-mixint', 'bbob-biobj-mixint'
# second parameter options:
# year: <year>, instances: <number>
# third parameter act as filter:
# 				   "dimensions: LIST", where LIST is the list of dimensions to
#                                keep in the suite (range-style syntax is not
#                                allowed here),
#                  "dimension_indices: VALUES", where VALUES is a list or a
#                                range of dimension indices (starting from 1)
#                                to keep in the suite,
#                  "function_indices: VALUES", where VALUES is a list or a
#                                range of function indices (starting from 1) to
#                                keep in the suite, and
#                  "instance_indices: VALUES", where VALUES is a list or a
#                                range of instance indices (starting from 1) to
#                                keep in the suite.