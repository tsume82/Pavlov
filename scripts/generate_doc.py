import sys

sys.path.insert(1, "./")
from utils.plot_utils import plot_experiment, compare_experiments
from utils.config_utils import loadConfiguration
from os.path import isdir

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

#▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚
#region Document parameters:
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

dir1 = "./experiments/CMA ppo deltabest/" # the document will be saved in this folder and the configuration will be taken from here
dir2 = "./experiments/CMA ppo paper model/"
name1 = "PPO with deltabest"
name2 = "PPO without deltabest"
# endregion
#▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚



















#▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚
# region writing the markdown
filename = "results_{0}_vs_{1}.md".format(name1, name2).replace(" ","_")
with open(dir1+filename, "w+") as f:

	# write comparison tables
	f.write("## Comparison Table\n")
	f.write("| Function    | p({0} < {1}) with AUC metric | p({0} < {1}) with best of the run metric |\n".format(name1,name2))
	f.write("| :---------- | ------------------------------ | ------------------------------- |\n")

	titles = []
	for fun in funcs:
		title = "{0} vs {1}: {2} comparison".format(name1, name2, fun)
		titles.append(title.replace(" ", "_"))
		if isdir(dir2+fun) and isdir(dir1+fun):
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
			auc = auc if auc < 0.5 else "**{}**".format(auc)
			final_best = final_best if final_best < 0.5 else "**{}**".format(final_best)
			f.write("| {} | {} | {} |\n".format(fun, auc, final_best))
		else:
			f.write("| {} | {} | {} |\n".format(fun, "---", "---"))


	# write plots
	f.write("\n## Plots\n\n")

	for fun, title in zip(funcs, titles):
		f.write("##### {}\n\n".format(fun))
		f.write("![]({}/{}.png)\n\n".format(fun, title))


	# write configuration (if it exists)
	config = None
	try:
		config = loadConfiguration(dir1+funcs[0])
	except:
		print("no configuration loaded")

	if config is not None:
		import json
		f.write("\n## Configuration\n\n")
		f.write("```json\n")
		config_string = json.dumps(config, indent=4)
		f.write(config_string)
		f.write("\n```")
# endregion
#▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚