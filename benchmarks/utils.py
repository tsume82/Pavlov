from benchmarks import functions
from benchmarks import COCO

# doc: https://hal.inria.fr/inria-00362633v2/document
# TODO probably use better names
COCO_OBJ_FN_NAMES = {
    "sphere": 1,
    "ellipsoid": 2,
    "rastrigin": 3,
    "buche-rastrigin": 4,
    "linear slope": 5,
    "attractive sector": 6,
    "step ellipsoidal": 7,
    "rosenbrock": 8,
    "rotated rosenbrock": 9,
    "ellipsoid 2": 10,
    "discus": 11,
    "bent cigar": 12,
    "sharp ridge": 13,
    "different powers": 14,
    "rastrigin 2": 15,
    "weierstrass": 16,
    "schaffers": 17,
    "schaffers 2": 18,
    "griewank-rosenbrock": 19,
    "schwefel": 20,
    "gallagher": 21,
    "gallagher 2": 22,
    "katsuura": 23,
    "lunacek": 24,
}


def loadFunction(name: str, dim=10, options={}):
	"""
	function to facilitate the configuration of the drivers. dim and options are only necessary for COCO's benchmark
	"""
	name = name.lower()
	if name in functions.all.keys():
		return functions.all[name]
	if name in COCO_OBJ_FN_NAMES.keys():
		# TODO possibility to use other suites ('bbob-biobj', 'bbob-largescale', 'bbob-mixint', 'bbob-biobj-mixint')
		arg1 = "instances: {}".format(options.get("instances", 1))
		arg2 = "function_indices: {}, dimensions: {}".format(COCO_OBJ_FN_NAMES[name], dim)
		# can't do differently because COCO's objects can't be pickled
		return lambda x: COCO.Suite("bbob", arg1, arg2)[0](x)
	# TODO use CEC2017, for now COCO have all we need
	return None