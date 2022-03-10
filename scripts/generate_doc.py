import sys

sys.path.insert(1, "./")
from utils.plot_utils import plot_experiment, compare_experiments
from utils.config_utils import loadConfiguration
from os.path import isdir, join, normpath

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
dir1 = "./experiments/multifunctions/CMA ppo" # the document will be saved in this folder and the configuration will be taken from here
dir2 = "./experiments/CSA"
name1 = "PPO multifunction"
name2 = "CSA"
filename = f"results_CMA_{name1}_vs_{name2}.md".replace(" ","_")
multienv = True # flag for multienv directory setup
# endregion
#▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚



















#▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚
# region writing the markdown
dir1 = normpath(dir1)
dir2 = normpath(dir2)
with open(join(dir1, filename), "w+") as f:

	# write comparison tables
	f.write("## Comparison Table\n")
	f.write("| Function    | p({0} < {1}) with AUC metric | p({0} < {1}) with best of the run metric |\n".format(name1,name2))
	f.write("| :---------- | ------------------------------ | ------------------------------- |\n")

	img_paths = []
	best_on_auc = best_on_final = tot = 0
	for fun in funcs:
		title = f"{name1} vs {name2}: {fun} comparison"
		img_name = f"{title.replace(' ', '_').replace(':', '_')}.png"
		

		fun1_path = join(dir1, fun)
		fun2_path = join(dir2, fun)

		if not multienv and not (isdir(fun2_path) and isdir(fun1_path)):
			f.write("| {} | {} | {} |\n".format(fun, "---", "---"))
			continue

		tot += 1

		if multienv:
			experimnet_path = join(dir1, f"{fun}_experiment.bin")
			image_path = join(dir1, "imgs", img_name)
			image_path_rel = join("imgs", img_name)
		else:
			experimnet_path = fun1_path
			image_path = join(fun1_path, img_name)
			image_path_rel = join(fun, img_name)

		img_paths.append(image_path_rel)

		auc, final_best = compare_experiments(
			[experimnet_path, fun2_path],
			[name1, name2],
			title=title,
			logyscale=True,
			logxscale=False,
			ylim=None,
			plotMode="std",
			save= image_path,
		)

		if auc > 0.5: best_on_auc += 1
		if final_best > 0.5: best_on_final += 1
		auc = auc if auc < 0.5 else "**{}**".format(auc)
		final_best = final_best if final_best < 0.5 else "**{}**".format(final_best)
		f.write("| {} | {} | {} |\n".format(fun, auc, final_best))

	auc_tot = f"{(best_on_auc/tot*100):.1f}" if best_on_auc/tot < 0.5 else f"**{best_on_auc/tot*100:.1f}**"
	fin_tot = f"{(best_on_final/tot*100):.1f}" if best_on_final/tot < 0.5 else f"**{best_on_final/tot*100:.1f}**"
	f.write(f"| **Total p({name1} < {name2})** | {auc_tot}% ({best_on_auc}/{tot}) | {fin_tot}% ({best_on_final}/{tot}) |\n")

	# write plots
	f.write("\n## Plots\n\n")

	for fun, img in zip(funcs, img_paths):
		f.write(f"##### {fun}\n\n")
		f.write(f"![]({img})\n\n")


	# write configuration (if it exists)
	config = None
	try:
		conf_path = dir1 if multienv else join(dir1, funcs[0])
		config = loadConfiguration(conf_path)
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