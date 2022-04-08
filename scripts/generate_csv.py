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
		"AttractiveSector_5D",
		"AttractiveSector_10D",
		"AttractiveSector_20D",
		"BuecheRastrigin_5D",
		"BuecheRastrigin_10D",
		"BuecheRastrigin_20D",
		"CompositeGR_5D",
		"CompositeGR_10D",
		"CompositeGR_20D",
		"DifferentPowers_5D",
		"DifferentPowers_10D",
		"DifferentPowers_20D",
		"LinearSlope_5D",
		"LinearSlope_10D",
		"LinearSlope_20D",
		"SharpRidge_5D",
		"SharpRidge_10D",
		"SharpRidge_20D",
		"StepEllipsoidal_5D",
		"StepEllipsoidal_10D",
		"StepEllipsoidal_20D",
		"RosenbrockRotated_5D",
		"RosenbrockRotated_10D",
		"RosenbrockRotated_20D",
		"SchaffersIllConditioned_5D",
		"SchaffersIllConditioned_10D",
		"SchaffersIllConditioned_20D",
		"LunacekBiR_5D",
		"LunacekBiR_10D",
		"LunacekBiR_20D",
		"GG101me_5D",
		"GG101me_10D",
		"GG101me_20D",
		"GG21hi_5D",
		"GG21hi_10D",
		"GG21hi_20D",
	]

	funcs = [
		"BentCigar",
		"Discus",
		"Ellipsoid",
		"Katsuura",
		"Rastrigin",
		"Rosenbrock",
		"Schaffers",
		"Schwefel",
		"Sphere",
		"Weierstrass",
	] + funcs
	
	# dir1 = "./experiments/multifunctions/CMA ppo 46 functions"
	# dir2 = ["./experiments/CSA"]
	# namelist = ["CSA"]
	dir2 = ["./experiments/jDE", "./experiments/iDE"]
	namelist = ["jDE", "iDE"]
	multienv = False # flag for multienv directory setup

	dir1 = "./experiments/jDE"
	dir2 = ["./experiments/iDE"]
	namelist = ["iDE"]
	filename = "../dataviz/iDE_vs_jDE.csv"
	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 functions"
# 	filename = "../dataviz/DE_multi_gauss.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo wIntraDeltaF 46 functions"
# 	filename = "../dataviz/DE_multi_gauss_wIntraDeltaF.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaX"
# 	filename = "../dataviz/DE_multi_gauss_wIntraDeltaX.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX"
# 	filename = "../dataviz/DE_multi_gauss_wIntraDeltaF_IntraDeltaX.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX (more training)"
# 	filename = "../dataviz/DE_multi_gauss_wIntraDeltaF_IntraDeltaX_moretrain.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX (100,50,10; train x2)"
# 	filename = "../dataviz/DE_multi_gauss_wIntraDeltaF_IntraDeltaX_big_moretrain.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX + InterDeltaX"
# 	filename = "../dataviz/DE_multi_gauss_wIntraDeltaF_IntraDeltaX_InterDeltaX.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX + InterDeltaX (more training)"
# 	filename = "../dataviz/DE_multi_gauss_wIntraDeltaF_IntraDeltaX_InterDeltaX_moretrain.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX + InterDeltaX (100,50,10; train x2)"
# 	filename = "../dataviz/DE_multi_gauss_wIntraDeltaF_IntraDeltaX_InterDeltaX_big_moretrain.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)
# #####
# 	dir1 = "./experiments/multifunctions/DE uniform ppo 46 functions"
# 	filename = "../dataviz/DE_multi_unif.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE uniform ppo wIntraDeltaF 46 functions"
# 	filename = "../dataviz/DE_multi_unif_wIntraDeltaF.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaX"
# 	filename = "../dataviz/DE_multi_unif_wIntraDeltaX.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaF + IntraDeltaX"
# 	filename = "../dataviz/DE_multi_unif_wIntraDeltaF_IntraDeltaX.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaF + IntraDeltaX (more training)"
# 	filename = "../dataviz/DE_multi_unif_wIntraDeltaF_IntraDeltaX_moretrain.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaF + IntraDeltaX (100,50,10; train x2)"
# 	filename = "../dataviz/DE_multi_unif_wIntraDeltaF_IntraDeltaX_big_moretrain.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaF + IntraDeltaX + InterDeltaX"
# 	filename = "../dataviz/DE_multi_unif_wIntraDeltaF_IntraDeltaX_InterDeltaX.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaF + IntraDeltaX + InterDeltaX (more training)"
# 	filename = "../dataviz/DE_multi_unif_wIntraDeltaF_IntraDeltaX_InterDeltaX_moretrain.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)

# 	dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaF + IntraDeltaX + InterDeltaX (100,50,10; train x2)"
# 	filename = "../dataviz/DE_multi_unif_wIntraDeltaF_IntraDeltaX_InterDeltaX_big_moretrain.csv"
# 	write_csv(funcs, dir1, dir2, namelist, filename, multienv)