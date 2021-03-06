import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, StrMethodFormatter, ScalarFormatter, LinearLocator, LogLocator
import numpy as np
from os.path import dirname, splitext, dirname, isdir, join, normpath
from os import makedirs
import json
import pickle

rcParams["font.size"] = 16


class plot_episodes:
	def __init__(self) -> None:
		self.lines = []
		self.data = []
		self.num_episodes = 0

	def plot(self, cumulative_reward):
		if isinstance(cumulative_reward, list):
			self.__plot_multi_episodes(cumulative_reward)
		else:
			self.__plot_single_episode(cumulative_reward)

	def show(self):
		plt.ioff()
		plt.show()

	def close(self):
		plt.close()

	def save(self, path, infos, info_file_name=None):
		splitted_path = splitext(path)
		# i = 1
		# path = splitted_path[0] + "_" + str(i) + splitted_path[1]
		# while exists(path):
		#     i += 1
		#     path = splitted_path[0] + "_" + str(i) + splitted_path[1]

		formatted_infos = json.dumps(infos, skipkeys=True)
		if info_file_name and isinstance(info_file_name, str):
			with open(dirname(path) + "/" + info_file_name, "w") as f:
				f.write(formatted_infos)

		if splitted_path[1].lower() in [".png", ".svg"]:  # Description keyword works only in SVG and PNG
			self.fig.savefig(path, metadata={"Description": formatted_infos})
		else:
			self.fig.savefig(path)

	def __plot_multi_episodes(self, cumulative_rewards):
		colors = ["blue"]
		labels = ["cumulative reward"]
		if self.num_episodes == 0:
			self.__init_plot()
			self.data = [[*range(len(cumulative_rewards))], cumulative_rewards]
			for i in range(1):
				(line,) = self.ax.plot(self.data[0], self.data[i + 1], color=colors[i], label=labels[i])
				self.lines.append(line)
		else:
			self.data[0].extend([*range(self.num_episodes, self.num_episodes + len(cumulative_rewards))])
			self.data[1].extend(cumulative_rewards)

			for i, line in enumerate(self.lines):
				line.set_xdata(np.array(self.data[0]))
				line.set_ydata(np.array(self.data[i + 1]))

		ymin = min([min(d) for d in self.data[1:]])
		ymax = max([max(d) for d in self.data[1:]])
		yrange = ymax - ymin
		self.num_episodes += len(cumulative_rewards)
		self.ax.set_xlim((0, self.num_episodes))
		self.ax.set_ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
		self.ax.grid(True)
		self.ax.legend()
		plt.draw()
		plt.pause(0.001)

	def __plot_single_episode(self, cumulative_reward):
		colors = ["blue"]
		labels = ["cumulative_reward"]
		if self.num_episodes == 0:
			self.__init_plot()
			self.data = [[self.num_episodes], [cumulative_reward]]
			for i in range(1):
				(line,) = self.ax.plot(self.data[0], self.data[i + 1], color=colors[i], label=labels[i])
				self.lines.append(line)
		else:
			self.data[0].append(self.num_episodes)
			self.data[1].append(cumulative_reward)

			for i, line in enumerate(self.lines):
				line.set_xdata(np.array(self.data[0]))
				line.set_ydata(np.array(self.data[i + 1]))

		ymin = min([min(d) for d in self.data[1:]])
		ymax = max([max(d) for d in self.data[1:]])
		yrange = ymax - ymin
		self.ax.set_xlim((0, self.num_episodes))
		self.ax.set_ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
		self.num_episodes += 1
		self.ax.grid(True)
		self.ax.legend()
		plt.draw()
		plt.pause(0.001)

	def __init_plot(self):
		self.fig = plt.figure("plot data")
		plt.ion()
		self.ax = self.fig.subplots()
		self.ax.set_xlabel("Episodes")
		self.ax.set_ylabel("Cumulative Reward")
		# self.button = Button(plt.axes([0.81, 0.0, 0.15, 0.075]), "Log Scale")
		# def toggle_fun(val):
		# 	newscale = "log" if self.ax.get_yaxis().get_scale() != "log" else "linear"
		# 	print(newscale)
		# 	self.ax.set_yscale(newscale)
		# self.button.on_clicked(toggle_fun)


def plot_experiment(
	experiment, title="Plot Experiment", logyscale=True, logxscale=True, ylim=None
):
	if isinstance(experiment, str):
		experiment = load_experiment(experiment)

	fig, axs = plt.subplots(2, sharex=True, figsize=(12, 6))
	fig.tight_layout(rect=(0.06, 0, 1, 0.98), h_pad=0)
	fig.canvas.manager.set_window_title(title)
	cmap = cm.get_cmap("tab20c")

	evolution_length = len(experiment[0]["fitness"])
	popLength = len(experiment[0]["fitness"][0])
	actions_per_step = len(experiment[0]["actions"][1])
	action_labels = []
	avg = np.empty(shape=[0, evolution_length])
	min_fit = np.inf
	max_fit = -np.inf
	for i, traj in enumerate(experiment):
		actions = []
		for step in traj["actions"]:
			if step:
				actions.append([np.average(v) if isinstance(v, (list, np.ndarray)) else v for v in step.values()])
				if not action_labels:
					action_labels = list(step.keys())
			else:
				actions.append([np.nan]*actions_per_step)
		popAvg = np.average(traj["fitness"], axis=1)
		min_fit = min(np.min(traj["fitness"]), min_fit)
		max_fit = max(np.max(traj["fitness"]), max_fit)
		popmin = np.min(traj["fitness"], axis=1)
		popmax = np.max(traj["fitness"], axis=1)
		avg = np.vstack([avg, popAvg])
		x = np.array([*range(0, evolution_length)]) * popLength
		axs[0].fill_between(x, popmin, popmax, color="lightsteelblue", alpha=0.6)  # darkorange alpha=0.15
		axs[0].plot(x, popAvg, color="blue", alpha=0.4)
		actions = np.array(actions)
		for j, label in enumerate(action_labels):
			axs[1].plot(x, actions[:,j], color=cmap(j * 4), alpha=0.4, label=label if i == 0 else None)
	axs[0].plot(x, np.average(avg, axis=0), color="red", alpha=0.8)

	print("max value: {}".format(max_fit))
	print("min value: {}".format(min_fit))

	# Top
	axs[0].title.set_text(title)
	if logyscale:
		if min_fit < 0:
			shift = 1e-9
			shifted_min_fit = min_fit - shift  # to avoid log(0)
			exp = lambda x: (2) ** (x) + shifted_min_fit
			log = lambda x: np.log(x - shifted_min_fit) / np.log(2)
			axs[0].set_yscale("function", functions=(log, exp))
			axs[0].set_yticks(np.geomspace(shift, max_fit - min_fit + shift, num=10) + min_fit - shift)
			axs[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.3f}"))
		else:
			axs[0].set_yscale("log", subs=[2, 4, 6, 8])
	else:
		axs[0].set_yscale("linear")
		axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
	ticks = list(axs[0].get_yticks()) + [min_fit]
	if logxscale:
		axs[0].set_xscale("log")
		axs[0].xaxis.set_major_formatter(ScalarFormatter())

	axs[0].set_ylim((min_fit, max_fit))
	axs[0].grid(True, which="both")
	# axs[0].set_xticklabels(np.array([*range(0,50)])*10)
	axs[0].set_ylabel("fitness", labelpad=0)
	axs[0].tick_params(axis="y", which="minor", grid_alpha=0.3)
	# axs[0].set_yticks(ticks)
	# axs[0].set_yticklabels([np.format_float_scientific(t, precision = 3, unique=True) for t in  ticks])

	# customization to match the paper's plots
	axs[0].set_xticks([10, 100, 200, 300, 400, 500])
	# axs[0].set_yticks([-50,-60,-70,-80,-90])
	if ylim:
		axs[0].set_ylim(ylim)

	# Bottom
	axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))
	axs[1].set_xlabel("function evaluations", labelpad=0)
	axs[1].set_ylabel("actions", labelpad=0)
	axs[1].grid(True, which="both")
	axs[1].tick_params(axis="y", which="minor", grid_alpha=0.3)
	plt.legend()

	plt.show()


"""
	compare 2 or more experiments (if there are exactly 2, it computes different statistic metrics)
	plotMode: "std", "adv", "raw"
"""
def compare_experiments(
	exp_list,
	exp_names,
	title="Compare Experiments",
	save=False,
	logyscale=True,
	logxscale=True,
	ylim=None,
	plotMode="std",
	show_actions=True,
):
	assert plotMode in ["std", "adv", "raw", None]

	for i, experiment in enumerate(exp_list):
		if isinstance(experiment, str):
			exp_list[i] = load_experiment(experiment)

	if plotMode is not None:
		evolution_length = len(exp_list[0][0]["fitness"])
		popLength = len(exp_list[0][0]["fitness"][0])
		actions_set = list(set([a for traj in exp_list for a in list(traj[0]["actions"][1].keys())]))
		actions_set.sort()
		min_fit = min_plot = np.inf
		max_fit = max_plot = -np.inf

		if show_actions:
			actions_per_step = len(actions_set)
			fig, axs = plt.subplots(1+actions_per_step, sharex=True, figsize=(12, 3+actions_per_step*3))
		else:
			fig, axs = plt.subplots(1, figsize=(6, 3))
			axs = [axs]
		actions_map = {k: i for i, k in enumerate(actions_set)}
		getActionAxis = lambda key: axs[1+actions_map[key]]
		fig.tight_layout(rect=(0.06, 0, 1, 0.98), h_pad=0)
		fig.canvas.manager.set_window_title(title)
		data = [None for _ in exp_list]
		cmap = cm.get_cmap("tab20c")
		cmap2 = cm.get_cmap("tab20b")

		# compute data to plot
		for i, experiment in enumerate(exp_list):
			action_labels = list(exp_list[i][0]["actions"][1].keys())
			avg = np.empty(shape=[0, evolution_length])
			all_actions = {k:np.empty(shape=(0, evolution_length)) for k in action_labels}
			all_best_during_run = np.empty(shape=[0, evolution_length])
			allPopAvg = np.empty(shape=[0, evolution_length])
			min_traj = np.ones(shape=[evolution_length]) * np.inf
			max_traj = np.ones(shape=[evolution_length]) * -np.inf
			first_col = cmap(i * 4)
			second_col = cmap(i * 4 + 1)
			third_col = cmap(i * 4 + 3)
			for j, traj in enumerate(experiment):
				actions = {k:[] for k in action_labels}
				for step in traj["actions"]:
					if step:
						for k, v in step.items():
							actions[k].append(np.average(v))
					else:
						for k in action_labels:
							actions[k].append(np.nan)

				min_fit = min(np.min(traj["fitness"]), min_fit)
				max_fit = max(np.max(traj["fitness"]), max_fit)
				for k in action_labels:
					all_actions[k] = np.vstack([all_actions[k], np.array(actions[k])])
				x = np.array([*range(1, evolution_length+1)]) * popLength

				if plotMode == "std":
					best_during_run = getBestDuringRun(traj["fitness"])
					all_best_during_run = np.vstack([all_best_during_run, best_during_run])
					data[i] = all_best_during_run
					continue

				popAvg = np.average(traj["fitness"], axis=1)
				avg = np.vstack([avg, popAvg])

				if plotMode == "adv":
					allPopAvg = np.vstack([allPopAvg, popAvg])
					min_traj = np.min([np.min(traj["fitness"], axis=1), min_traj], axis=0)
					max_traj = np.max([np.max(traj["fitness"], axis=1), max_traj], axis=0)
					continue
				if plotMode == "raw":
					popmin = np.min(traj["fitness"], axis=1)
					popmax = np.max(traj["fitness"], axis=1)
					axs[0].fill_between(x, popmin, popmax, color=third_col, alpha=0.6)
					axs[0].plot(x, popAvg, color=first_col, alpha=0.4)
					if show_actions:
						for act_label in action_labels:
							getActionAxis(act_label).plot(x, actions[act_label], color=first_col, alpha=0.4)
					continue
			
			

			# plot data
			if plotMode == "adv":
				min_plot = min(min_plot, min_fit)
				max_plot = max(max_plot, max_fit)
				axs[0].plot(x, np.average(avg, axis=0), color=first_col, label=exp_names[i], lw=2, zorder=3 * (i + 1) + 2)
				axs[0].fill_between(
					x,
					np.min(allPopAvg, axis=0),
					np.max(allPopAvg, axis=0),
					color=second_col,
					alpha=1 / (1 + i),
					zorder=3 * (i + 1) + 1,
				)
				axs[0].fill_between(x, min_traj, max_traj, color=third_col, alpha=1 / (1 + i), zorder=3 * (i + 1))
				if show_actions:
					for act_label in action_labels:
						getActionAxis(act_label).plot(x, np.average(all_actions[act_label], axis=0), color=first_col, label=exp_names[i], lw=2)
						getActionAxis(act_label).fill_between(
							x, np.min(all_actions[act_label], axis=0), np.max(all_actions[act_label], axis=0), color=second_col, alpha=2**(-i)
						)
			elif plotMode == "raw":
				min_plot = min(min_plot, min_fit)
				max_plot = max(max_plot, max_fit)
				axs[0].plot(x, np.average(avg, axis=0), c="red", alpha=0.8)
			elif plotMode == "std":
				all_best_avg = np.average(all_best_during_run, axis=0)
				all_best_std = np.std(all_best_during_run, axis=0)
				min_plot = min(min_plot, np.min(all_best_avg - all_best_std))
				max_plot = max(max_plot, np.max(all_best_avg + all_best_std))
				if show_actions:
					for act_label in action_labels:
						actions_avg =  np.average(all_actions[act_label], axis=0)
						actions_std = np.std(all_actions[act_label], axis=0)
						getActionAxis(act_label).plot(x, actions_avg, color=first_col, lw=2, zorder=3 * (i + 1) + 2)
						getActionAxis(act_label).fill_between(
							x,
							(actions_avg - actions_std),
							(actions_avg + actions_std),
							color=second_col,
							alpha=1 / (1 + i),
							zorder=3 * (i + 1),
						)
		
		if plotMode == "std":
			shift = 1
			min_avg = np.inf
			use_shift = logyscale and min_plot < shift
			for i, d in enumerate(data):
				first_col = cmap(i * 4)
				second_col = cmap(i * 4 + 1)
				all_best_avg = np.average(d, axis=0)
				all_best_std = np.std(d, axis=0)
				if use_shift: 
					all_best_avg=all_best_avg+abs(min_plot)+shift
					min_avg = min(min_avg, np.min(all_best_avg))
				axs[0].plot(x, all_best_avg, color=first_col, lw=2, zorder=3 * (i + 1) + 2, label=exp_names[i])
				axs[0].fill_between(
					x,
					all_best_avg - all_best_std,
					all_best_avg + all_best_std,
					color=second_col,
					alpha=2**(-i),
					zorder=3 * (i + 1),
				)
			if use_shift:
				max_plot+=abs(min_plot)+shift
				min_plot = (min_avg+shift)/2

		# configure plots

		# Top
		if logyscale:
			axs[0].set_yscale("log", subs=[2, 4, 6, 8], base=10)
			if abs(min_plot-max_plot) < 1e3:
				axs[0].yaxis.set_minor_formatter(ScalarFormatter())
				axs[0].tick_params(axis="y", which="minor", labelsize=12)
		else:
			axs[0].set_yscale("linear")
			axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
		if logxscale:
			axs[0].set_xscale("log")
			axs[0].xaxis.set_major_formatter(ScalarFormatter())

		# ticks = list(axs[0].get_yticks()) + [min_fit]
		axs[0].set_title(title)
		axs[0].set_ylim((min_plot, max_plot))
		axs[0].grid(True, which="both")
		# axs[0].set_xticklabels(np.array([*range(0,50)])*10)
		axs[0].set_ylabel("fitness", labelpad=0)
		axs[0].yaxis.label.set_color("forestgreen")
		axs[0].tick_params(axis="y", which="minor", grid_alpha=0.3)
		# axs[0].set_yticks(ticks)
		# axs[0].set_yticklabels([np.format_float_scientific(t, precision = 3, unique=True) for t in  ticks])

		ticks = [10]+[*range(0, evolution_length*popLength, int(np.floor(evolution_length*popLength/100/5))*100)][1:]
		axs[0].set_xticks(ticks)
		if ylim:
			axs[0].set_ylim(ylim)

		# Bottom
		# axs[-1].set_xlabel("function evaluations", labelpad=0)

		axs[0].legend([Line2D([0],[0],color=cmap(i * 4)) for i, _ in enumerate(exp_names)], exp_names, prop = {"size": 12})

		if show_actions:
			for action in actions_set:
				getActionAxis(action).legend([Line2D([0],[0],color=cmap(i * 4)) for i, l in enumerate(exp_names)], exp_names, prop = {"size": 12})
				getActionAxis(action).yaxis.set_minor_locator(AutoMinorLocator(2))
				getActionAxis(action).set_ylabel(action, labelpad=0)
				getActionAxis(action).yaxis.label.set_color("forestgreen")
				getActionAxis(action).grid(True, which="both")
				getActionAxis(action).tick_params(axis="y", which="minor", grid_alpha=0.3)
			
		if save:
			makedirs(dirname(save), exist_ok=True)
			fig.savefig(save)
		else:
			plt.show()

	# compute statistical metrics
	if len(exp_list) == 2:
		auc = compute_metrics_comparison(exp_list, "AUC")
		final_best = compute_metrics_comparison(exp_list, "final_best")
		return auc, final_best
	return 0,0


def getBestDuringRun(fitness):
	curr_min = np.inf
	out = []
	for pop_fitness in fitness:
		curr_min = min(np.min(pop_fitness), curr_min)
		out.append(curr_min)
	return np.array(out)


"""
	compute the metric for each run and it compare between 2 experiments
	metric: "AUC" (Area under the curve),
			"final_best"
"""
def compute_metrics_comparison(exp_list, metric="AUC"):
	assert len(exp_list) == 2
	for i, experiment in enumerate(exp_list):
		if isinstance(experiment, str) and isdir(experiment):
			exp_list[i] = load_experiment(experiment)

	def compute_area_metrics(exp_list):
		num_run = len(exp_list[0])
		# metrics of each run for both the experiments
		metrics = np.zeros(shape=[2, num_run])

		for i, experiment in enumerate(exp_list):
			for j, run in enumerate(experiment):
				min_fit_run = getBestDuringRun(run["fitness"])

				# compute sum of trapezoid areas
				area = np.trapz(min_fit_run, dx=1)
				metrics[i, j] = area

		return metrics

	def compute_final_best_metrics(exp_list):
		num_run = len(exp_list[0])
		# metrics of each run for both the experiments
		metrics = np.zeros(shape=[2, num_run])

		for i, experiment in enumerate(exp_list):
			for j, run in enumerate(experiment):
				min_run = np.min(run["fitness"])
				metrics[i, j] = min_run
		return metrics

	metric = metric.lower()
	if metric == "auc":
		metrics = compute_area_metrics(exp_list)
	elif metric == "final_best":
		metrics = compute_final_best_metrics(exp_list)

	# create a matrix of all pairs between the runs of the 2 experiments
	mesh = np.array(np.meshgrid(metrics[0], metrics[1]))
	combinations = mesh.T.reshape(-1, 2)

	# create an array with 1 if the metric in experiment 1 is less than experiment 2, 0 otherwise
	# sum the array and divide it for its evolution_length to get a probablity
	# see Equation 1 in "Learning Step-Size Adaptation in CMA-ES"
	prob = np.sum([1 if c[0] < c[1] else 0 for c in combinations]) / combinations.shape[0]
	return prob


def save_experiment(experiment, folder, name="experiment"):
	name = name if isinstance(name, str) else "experiment"
	file_path = join(folder, f"{normpath(name)}.bin")
	makedirs(dirname(file_path), exist_ok=True)
	with open(file_path, "wb") as f:
		pickle.dump(experiment, f)


def load_experiment(dir_or_file):
	if isdir(dir_or_file):
		dir_or_file = join(dir_or_file, "experiment.bin")
	with open(dir_or_file, "rb") as f:
		return pickle.load(f)
