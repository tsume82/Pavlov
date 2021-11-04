import os
import sys
sys.path.insert(1, "./")
from examples.configurations import update_and_return, CSA_configuration
from main import main

all_csa_configurations = [
    update_and_return(CSA_configuration, {"env.env_config": {"solver_driver_args": [dim, 10, fun, 0.5]}})
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

kwargs = {}
kwargs["num_runs"] = 50
kwargs["save_exp"] = True
kwargs["plot"] = False

for conf, fun in zip(all_csa_configurations, funcs):
	dir = "./experiments/CSA/"+fun
	os.mkdir(dir)
	main(conf, False, dir, **kwargs)