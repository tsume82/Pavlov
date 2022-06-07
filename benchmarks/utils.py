from benchmarks import functions

try:
    from cma import bbobbenchmarks
except ModuleNotFoundError as err:
    print(f"\033[91m{err}\033[00m")
    bbobbenchmarks = None

# doc: https://hal.inria.fr/inria-00362633v2/document
# in the paper used ids for train: [12, 11, 2, 23, 15, 8, 17, 20, 1, 16]
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


def loadFunction(function, dim=10, options={}, lib: str = None):
    """
    function to facilitate the configuration of the drivers. Load an object function from its name

    function: name or id of the object function to load
    dim: dimension of the input space (necessary only for COCO's benchmark)
    options: options for selecting functions e.g. the instance of the function (necessary only for COCO's benchmark)
    lib: what library to use. "local", "COCO", "cma", default search on all
    """
    if isinstance(function, str):
        function = int(function) if function.isdigit() else function.lower()
    if lib:
        lib = lib.lower()
        assert lib in ["local", "cma"]

    if (not lib or lib == "local"):
        if isinstance(function, str) and function in functions.all.keys():
            return functions.all[function]
        elif isinstance(function, int) and 0 <= function < len(functions.all):
            return list(functions.all.values())[function]

    if (not lib or lib == "cma"):
        if bbobbenchmarks:
            iinstance = options.get("instance", 0) # id instance
            ifun = BBOB_OBJ_FN_NAMES.get(function, function) # id function
            if not (isinstance(ifun, int) and 1 <= ifun <= 24):
                raise ValueError(f"Unknown function: {function}")
            return bbobbenchmarks.instantiate(ifun, iinstance)[0]
        elif lib == "cma":
            raise ModuleNotFoundError("cma module not found")

    raise AttributeError("'{}' object function doesn't exists".format(function))
