import sys

sys.path.insert(1, "./")
from utils.plot_utils import plot_experiment, compare_experiments

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
]

dir1 = "./experiments/DE ppo deltabest/"
dir2 = "./experiments/iDE/"
name1 = "PPO deltabest"
name2 = "iDE"

with open(dir1+"results_iDE.md", "w+") as f:
	f.write("## Comparison Table\n\nProbability of PPO trained policy outperforming CSA using 2 different metrics: Area under the curve and the absolute best of the run.\n")
	f.write("| Function    | p({0} < {1}) with AUC metric | p({0} < {1}) with best of the run metric |\n".format(name1,name2))
	f.write("| :---------- | ------------------------------ | ------------------------------- |\n")

	titles = []
	for fun in funcs:
		title = "iDE {} comparison".format(fun)
		titles.append(title.replace(" ", "_"))
		auc, final_best = compare_experiments(
			[dir1+fun,dir2+fun],
			[name1, name2],
			title=title,
			logyscale=True,
			logxscale=False,
			ylim=None,
			plotMode="std",
			save=dir1+fun+"/{}.png".format(title.replace(" ", "_")),
		)
		f.write("| {} | {} | {} |\n".format(
			fun, 
			auc if auc < 0.5 else "**{}**".format(auc), 
			final_best if final_best < 0.5 else "**{}**".format(final_best), 
			))
	
	f.write("\n## Plots\n\n")

	for fun, title in zip(funcs, titles):
		f.write("##### {}\n\n".format(fun))
		f.write("![]({}/{}.png)\n\n".format(fun, title))


