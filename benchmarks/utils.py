from benchmarks import functions
from benchmarks import COCO

try:
    from cma import bbobbenchmarks
except ModuleNotFoundError as err:
    bbobbenchmarks = None

# doc: https://hal.inria.fr/inria-00362633v2/document
# TODO probably use better names
BBOB_OBJ_FN_NAMES = {
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


def loadFunction(name: str, dim=10, options={}, lib: str = None):
    """
    function to facilitate the configuration of the drivers. Load an object function from its name

    name: name of the object function to load
    dim: dimension of the input space (necessary only for COCO's benchmark)
    options: options for selecting functions e.g. the instance of the function (necessary only for COCO's benchmark)
    lib: what library to use. "local", "COCO", "cma", default search on all
    """
    name = name.lower()
    if lib:
        lib = lib.lower()
        assert lib in ["local", "coco", "cma"]
    if (not lib or lib == "local") and name in functions.all.keys():
        return functions.all[name]
    if (not lib or lib == "coco") and name in BBOB_OBJ_FN_NAMES.keys():
            # TODO possibility to use other suites ('bbob-biobj', 'bbob-largescale', 'bbob-mixint', 'bbob-biobj-mixint')
            arg1 = "instances: {}".format(options.get("instances", 1))
            arg2 = "function_indices: {}, dimensions: {}".format(BBOB_OBJ_FN_NAMES[name], dim)
            # can't do differently because COCO's objects can't be pickled
            return lambda x: COCO.Suite("bbob", arg1, arg2)[0](x)
    if (not lib or lib == "cma") and name in BBOB_OBJ_FN_NAMES.keys():
        if bbobbenchmarks:
            return bbobbenchmarks.instantiate(BBOB_OBJ_FN_NAMES[name])[0]
        else:
            raise ModuleNotFoundError("cma module not found")
    raise AttributeError("'{}' object function doesn't exists".format(name))
    # TODO use CEC2017, for now COCO have all we need
