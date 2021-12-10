import os
import sys

sys.path.insert(1, "./")
from examples.configurations import update_and_return, de_adapt_configuration
from main import main

all_ide_configurations = [
    update_and_return(de_adapt_configuration, 
	{
		"env.env_config": {"solver_driver_args": [10, 10, fun, "best1bin"]},
		"agent.algorithm.adapt_strategy": "jDE",
	})
    for fun, dim in zip(
		[6, 6, 6, 4, 4, 4, 19, 19, 19, 14, 14, 14, 5, 5, 5, 13, 13, 13, 7, 7, 7, 9, 9, 9, 18, 18, 18, 24, 24, 24, 21, 21, 21, 22, 22, 22],
		[5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20, 5, 10, 20]
		)
]

funcs = ['AttractiveSector_5D', 'AttractiveSector_10D', 'AttractiveSector_20D', 'BuecheRastrigin_5D', 'BuecheRastrigin_10D', 'BuecheRastrigin_20D',
		'CompositeGR_5D', 'CompositeGR_10D', 'CompositeGR_20D', 'DifferentPowers_5D', 'DifferentPowers_10D', 'DifferentPowers_20D',
		'LinearSlope_5D', 'LinearSlope_10D', 'LinearSlope_20D', 'SharpRidge_5D', 'SharpRidge_10D', 'SharpRidge_20D',
		'StepEllipsoidal_5D', 'StepEllipsoidal_10D', 'StepEllipsoidal_20D', 'RosenbrockRotated_5D', 'RosenbrockRotated_10D', 'RosenbrockRotated_20D',
		'SchaffersIllConditioned_5D', 'SchaffersIllConditioned_10D', 'SchaffersIllConditioned_20D', 'LunacekBiR_5D', 'LunacekBiR_10D', 'LunacekBiR_20D',
		'GG101me_5D', 'GG101me_10D', 'GG101me_20D', 'GG21hi_5D', 'GG21hi_10D', 'GG21hi_20D']

# all_ide_configurations = [
#     update_and_return(
#         de_adapt_configuration,
#         {
#             "env.env_config": {"solver_driver_args": [10, 10, fun, "best1bin"]},
#             "agent.algorithm.adapt_strategy": "jDE",
#         },
#     )
#     for fun in [12, 11, 2, 23, 15, 8, 17, 20, 1, 16]
# ]
# funcs = [
#     "BentCigar",
#     "Discus",
#     "Ellipsoid",
#     "Katsuura",
#     "Rastrigin",
#     "Rosenbrock",
#     "Schaffers",
#     "Schwefel",
#     "Sphere",
#     "Weierstrass",
# ]

kwargs = {}
kwargs["num_runs"] = 50
kwargs["save_exp"] = True
kwargs["plot"] = False

for conf, fun in zip(all_ide_configurations, funcs):
    dir = "./experiments/jDE/" + fun
    os.mkdir(dir)
    main(conf, train=False, folder=dir, **kwargs)
