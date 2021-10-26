import os
import sys
sys.path.insert(1, "./")
from examples.configurations import update_and_return, CSA_configuration
from main import main

all_csa_configurations = [
    update_and_return(CSA_configuration, {"env.env_config": {"solver_driver_args": [10, 10, fun, sigma_init]}})
    for fun, sigma_init in zip([12, 11, 2, 23, 15, 8, 17, 20, 1, 16],[1.28, 0.38, 1.54, 1.18, 0.1, 1.66, 0.33, 0.1, 1.63, 0.1])
]
funcs = ["BentCigar", "Discus", "Ellipsoid", "Katsuura", "Rastrigin", "Rosenbrock", "Schaffers", "Schwefel", "Sphere", "Weierstrass"]

kwargs = {}
kwargs["num_runs"] = 50
kwargs["save_exp"] = True
kwargs["plot"] = False

for conf, fun in zip(all_csa_configurations, funcs):
	dir = "./experiments/CSA/"+fun
	os.mkdir(dir)
	main(conf, False, dir, **kwargs)