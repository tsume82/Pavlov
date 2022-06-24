import sys

sys.path.insert(1, "./")
from utils.plot_utils import plot_experiment, compare_experiments
from utils.config_utils import loadConfiguration
from os.path import isdir, join, normpath

def write_doc(funcs, dir1, dir2, name1, name2, filename, multienv=False, generate_plots=True):
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
			print(f"{fun}")
			title = f"{name1} vs {name2}: {fun} comparison"
			img_name = f"{title.replace(' ', '_').replace(':', '_')}.png"
			

			fun1_path = join(dir1, fun)
			fun2_path = join(dir2, fun)

			if not multienv and not (isdir(fun2_path) and isdir(fun1_path)):
				f.write("| {} | {} | {} |\n".format(fun, "---", "---"))
				continue

			tot += 1

			if multienv:
				experiment_path = join(dir1, f"{fun}_experiment.bin")
				image_path = join(dir1, "imgs", img_name)
				image_path_rel = join("imgs", img_name)
			else:
				experiment_path = fun1_path
				image_path = join(fun1_path, img_name)
				image_path_rel = join(fun, img_name)

			img_paths.append(image_path_rel)

			auc, final_best = compare_experiments(
				[experiment_path, fun2_path],
				[name1, name2],
				title=title,
				logyscale=True,
				logxscale=False,
				ylim=None,
				plotMode="std" if generate_plots else None,
				save=image_path,
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
		if generate_plots:
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
	
	dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX (100,50,10; train x2)" # the document will be saved in this folder and the configuration will be taken from here
	dir2 = "./experiments/iDE"
	name1 = "policy"
	name2 = "iDE"
	filename = f"_results_DE_{name1}_vs_{name2}.md".replace(" ","_")
	multienv = True # flag for multienv directory setup
	generate_plots = True


	write_doc(funcs, dir1, dir2, name1, name2, filename, multienv, generate_plots)

	# dir2 = ["./experiments/jDE", "./experiments/iDE"]
	# namelist = ["jDE", "iDE"]
	# multienv = True # flag for multienv directory setup

	# dir1 = "./experiments/multifunctions/DE gaussian ppo 46 wIntraDeltaF + IntraDeltaX (100,50,10; train x2)"
	# filename = "./graph_generation/inst_42_DE_multi_gauss_wIntraDeltaF_IntraDeltaX_big_moretrain.csv"
	# write_csv(funcs, dir1, dir2, namelist, filename, multienv)

	# dir1 = "./experiments/multifunctions/DE uniform ppo wIntraDeltaF 46 functions"
	# filename = "./graph_generation/inst_42_DE_multi_unif_wIntraDeltaF.csv"
	# write_csv(funcs, dir1, dir2, namelist, filename, multienv)

	# dir1 = "./experiments/multifunctions/DE uniform ppo 46 wIntraDeltaF + IntraDeltaX + InterDeltaX (more training)"
	# filename = "./graph_generation/inst_42_DE_multi_unif_wIntraDeltaF_IntraDeltaX_InterDeltaX_moretrain.csv"
	# write_csv(funcs, dir1, dir2, namelist, filename, multienv)
