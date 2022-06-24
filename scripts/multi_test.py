import os
import sys

sys.path.insert(1, "./")
from examples.configurations import update_and_return, de_adapt_configuration, ids_46_functions, dims_46_functions
from main import main

all_ide_configurations = [
    update_and_return(de_adapt_configuration, 
	{
		"env.env_config": {
			"solver_driver_args": [dim, 10, {"id": fun, "instance": 42}, "best1bin"],
			"state_metrics_config": [(dim, 2)],
			"action_space_config": {"F": {"max": [2] * dim, "min": [0] * dim}, "CR": {"max": [1] * dim, "min": [0] * dim}},
			},
		"agent.algorithm.adapt_strategy": "jDE",
	})
    for fun, dim in zip(ids_46_functions,dims_46_functions)
]

funcs = ['BentCigar','Discus','Ellipsoid','Katsuura','Rastrigin','Rosenbrock','Schaffers','Schwefel','Sphere','Weierstrass',
		'AttractiveSector_5D', 'AttractiveSector_10D', 'AttractiveSector_20D', 'BuecheRastrigin_5D', 'BuecheRastrigin_10D', 'BuecheRastrigin_20D',
		'CompositeGR_5D', 'CompositeGR_10D', 'CompositeGR_20D', 'DifferentPowers_5D', 'DifferentPowers_10D', 'DifferentPowers_20D',
		'LinearSlope_5D', 'LinearSlope_10D', 'LinearSlope_20D', 'SharpRidge_5D', 'SharpRidge_10D', 'SharpRidge_20D',
		'StepEllipsoidal_5D', 'StepEllipsoidal_10D', 'StepEllipsoidal_20D', 'RosenbrockRotated_5D', 'RosenbrockRotated_10D', 'RosenbrockRotated_20D',
		'SchaffersIllConditioned_5D', 'SchaffersIllConditioned_10D', 'SchaffersIllConditioned_20D', 'LunacekBiR_5D', 'LunacekBiR_10D', 'LunacekBiR_20D',
		'GG101me_5D', 'GG101me_10D', 'GG101me_20D', 'GG21hi_5D', 'GG21hi_10D', 'GG21hi_20D']

kwargs = {}
kwargs["num_runs"] = 50
kwargs["checkpoint"] = False
kwargs["save_exp"] = True
kwargs["plot"] = False

for conf, fun in zip(all_ide_configurations, funcs):
    dir = f"./experiments/jDE/inst_42_{fun}"
    os.mkdir(dir)
    main(conf, train=False, folder=dir, **kwargs)
