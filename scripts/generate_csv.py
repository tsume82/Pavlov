import sys

sys.path.insert(1, "./")
from utils.plot_utils import plot_experiment, compare_experiments
from utils.config_utils import loadConfiguration
from os.path import isdir, join, normpath

def write_csv(funcs, dir1, dirlist, namelist, filename, multienv=False):
	dir1 = normpath(dir1)
	with open(filename, "w+") as f:

		# write comparison tables
		f.write(f"Function")
		for name in namelist:
			f.write(f",AUC {name},Best {name}")
		f.write("\n")
		
		best_on_auc = best_on_final = tot = 0
		
		for fun in funcs:
			print(f"{fun}")
			f.write(f"{fun}")

			for dir, name in zip(dirlist, namelist):
				dir2 = normpath(dir)
				fun1_path = join(dir1, fun)
				fun2_path = join(dir2, fun)

				if not multienv and not (isdir(fun2_path) and isdir(fun1_path)):
					f.write(f",,")
					continue

				tot += 1

				if multienv:
					experimnet_path = join(dir1, f"{fun}_experiment.bin")
				else:
					experimnet_path = fun1_path

				auc, final_best = compare_experiments(
					[experimnet_path, fun2_path],
					["name1", "name2"],
					logyscale=True,
					logxscale=False,
					ylim=None,
					plotMode=None,
					save=None,
				)

				if auc > 0.5: best_on_auc += 1
				if final_best > 0.5: best_on_final += 1

				f.write(f",{auc},{final_best}")
			f.write("\n")

if __name__ == "__main__":

	funcs = [
		"inst_42_AttractiveSector_5D",
		"inst_42_AttractiveSector_10D",
		"inst_42_AttractiveSector_20D",
		"inst_42_BuecheRastrigin_5D",
		"inst_42_BuecheRastrigin_10D",
		"inst_42_BuecheRastrigin_20D",
		"inst_42_CompositeGR_5D",
		"inst_42_CompositeGR_10D",
		"inst_42_CompositeGR_20D",
		"inst_42_DifferentPowers_5D",
		"inst_42_DifferentPowers_10D",
		"inst_42_DifferentPowers_20D",
		"inst_42_LinearSlope_5D",
		"inst_42_LinearSlope_10D",
		"inst_42_LinearSlope_20D",
		"inst_42_SharpRidge_5D",
		"inst_42_SharpRidge_10D",
		"inst_42_SharpRidge_20D",
		"inst_42_StepEllipsoidal_5D",
		"inst_42_StepEllipsoidal_10D",
		"inst_42_StepEllipsoidal_20D",
		"inst_42_RosenbrockRotated_5D",
		"inst_42_RosenbrockRotated_10D",
		"inst_42_RosenbrockRotated_20D",
		"inst_42_SchaffersIllConditioned_5D",
		"inst_42_SchaffersIllConditioned_10D",
		"inst_42_SchaffersIllConditioned_20D",
		"inst_42_LunacekBiR_5D",
		"inst_42_LunacekBiR_10D",
		"inst_42_LunacekBiR_20D",
		"inst_42_GG101me_5D",
		"inst_42_GG101me_10D",
		"inst_42_GG101me_20D",
		"inst_42_GG21hi_5D",
		"inst_42_GG21hi_10D",
		"inst_42_GG21hi_20D",
	]

	funcs = [
		"inst_42_BentCigar",
		"inst_42_Discus",
		"inst_42_Ellipsoid",
		"inst_42_Katsuura",
		"inst_42_Rastrigin",
		"inst_42_Rosenbrock",
		"inst_42_Schaffers",
		"inst_42_Schwefel",
		"inst_42_Sphere",
		"inst_42_Weierstrass",
	] + funcs
	
	# dir1 = "./experiments/multifunctions/CMA ppo 46 functions"
	# dir2 = ["./experiments/CSA"]
	# namelist = ["CSA"]
	dir2 = ["./experiments/jDE", "./experiments/iDE"]
	namelist = ["jDE", "iDE"]
	multienv = True # flag for multienv directory setup

	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX (100,50,10; train x2)"
	filename = "./graph_generation/inst_42_DE_multi_gauss_wIntraDeltaF_IntraDeltaX_big_moretrain.csv"
	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

	dir1 = "./experiments/multifunctions/DE uniform ppo wIntraDeltaF 46 functions"
	filename = "./graph_generation/inst_42_DE_multi_unif_wIntraDeltaF.csv"
	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

	dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaF + IntraDeltaX + InterDeltaX (more training)"
	filename = "./graph_generation/inst_42_DE_multi_unif_wIntraDeltaF_IntraDeltaX_InterDeltaX_moretrain.csv"
	write_csv(funcs, dir1, dir2, namelist, filename, multienv)
