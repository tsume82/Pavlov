import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from matplotlib.widgets import Button
from matplotlib.ticker import AutoMinorLocator, StrMethodFormatter, ScalarFormatter, LinearLocator, LogLocator
import numpy as np
from numpy.matrixlib.defmatrix import matrix
from utils.array_utils import getScalar
from os.path import exists, splitext, dirname, isdir
import json
import pickle

from numpy.lib.arraysetops import unique


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
	experiment, title="Plot Experiment", title_act="step size", logyscale=True, logxscale=True, ylim=None
):
	if isinstance(experiment, str):
		experiment = load_experiment(experiment)

	fig, axs = plt.subplots(2, sharex=True, figsize=(12, 6))
	fig.tight_layout(rect=(0.06, 0, 1, 0.98), h_pad=0)
	fig.canvas.manager.set_window_title(title)

	length = len(experiment[0]["fitness"])
	popLength = len(experiment[0]["fitness"][0])
	avg = np.empty(shape=[0, length])
	min_fit = np.inf
	max_fit = -np.inf
	for traj in experiment:
		actions = []
		for step in traj["actions"]:
			actions.append(step.get("step_size", None))  # TODO handle different actions spaces
		popAvg = np.average(traj["fitness"], axis=1)
		min_fit = min(np.min(traj["fitness"]), min_fit)
		max_fit = max(np.max(traj["fitness"]), max_fit)
		popmin = np.min(traj["fitness"], axis=1)
		popmax = np.max(traj["fitness"], axis=1)
		avg = np.vstack([avg, popAvg])
		x = np.array([*range(0, length)]) * popLength
		axs[0].fill_between(x, popmin, popmax, color="lightsteelblue", alpha=0.6)  # darkorange alpha=0.15
		axs[0].plot(x, popAvg, color="blue", alpha=0.4)
		axs[1].plot(x, actions, color="black", alpha=0.4)
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
	axs[1].set_ylabel(title_act, labelpad=0)
	axs[1].grid(True, which="both")
	axs[1].tick_params(axis="y", which="minor", grid_alpha=0.3)

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
):
	assert plotMode in ["std", "adv", "raw"]
	fig, axs = plt.subplots(2, sharex=True, figsize=(12, 6))
	fig.tight_layout(rect=(0.06, 0, 1, 0.98), h_pad=0)
	fig.canvas.manager.set_window_title(title)
	action_axis_label = "step size"
	cmap = cm.get_cmap("tab20c")

	for i, experiment in enumerate(exp_list):
		if isinstance(experiment, str):
			exp_list[i] = load_experiment(experiment)

	length = len(exp_list[0][0]["fitness"])
	popLength = len(exp_list[0][0]["fitness"][0])
	min_fit = min_plot = np.inf
	max_fit = max_plot = -np.inf

	# compute data to plot
	for i, experiment in enumerate(exp_list):
		avg = np.empty(shape=[0, length])
		all_actions = np.empty(shape=[0, length])
		all_best_during_run = np.empty(shape=[0, length])
		allPopAvg = np.empty(shape=[0, length])
		min_traj = np.ones(shape=[length]) * np.inf
		max_traj = np.ones(shape=[length]) * -np.inf
		first_col = cmap(i * 4)
		second_col = cmap(i * 4 + 1)
		third_col = cmap(i * 4 + 3)
		for j, traj in enumerate(experiment):
			actions = []
			for step in traj["actions"]:
				action = getScalar(step.get("step_size", None))
				actions.append(action)  # TODO handle different actions spaces

			min_fit = min(np.min(traj["fitness"]), min_fit)
			max_fit = max(np.max(traj["fitness"]), max_fit)
			all_actions = np.vstack([all_actions, actions])
			x = np.array([*range(1, length+1)]) * popLength

			if plotMode == "std":
				best_during_run = getBestDuringRun(traj["fitness"])
				all_best_during_run = np.vstack([all_best_during_run, best_during_run])
				continue

			popAvg = np.average(traj["fitness"], axis=1)
			avg = np.vstack([avg, popAvg])

			if plotMode == "adv":
				allPopAvg = np.vstack([allPopAvg, popAvg])
				min_traj = np.min([np.min(traj["fitness"], axis=1), min_traj], axis=0)
				max_traj = np.max([np.max(traj["fitness"], axis=1), max_traj], axis=0)
				continue
			if plotMode == "raw":
				label = exp_names[i] if j == 0 else None
				popmin = np.min(traj["fitness"], axis=1)
				popmax = np.max(traj["fitness"], axis=1)
				axs[0].fill_between(x, popmin, popmax, color=third_col, alpha=0.6)
				axs[0].plot(x, popAvg, color=first_col, alpha=0.4, label=label)
				axs[1].plot(x, actions, color=first_col, alpha=0.4, label=label)
				continue

		# plot data

		if plotMode == "adv":
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
			axs[1].plot(x, np.average(all_actions, axis=0), color=first_col, label=exp_names[i], lw=2)
			axs[1].fill_between(
				x, np.min(all_actions, axis=0), np.max(all_actions, axis=0), color=second_col, alpha=1 / (1 + i)
			)
			min_plot = min(min_plot, min_fit)
			max_plot = max(max_plot, max_fit)
		elif plotMode == "raw":
			axs[0].plot(x, np.average(avg, axis=0), c="red", alpha=0.8)
			min_plot = min(min_plot, min_fit)
			max_plot = max(max_plot, max_fit)
		elif plotMode == "std":
			all_best_avg = np.average(all_best_during_run, axis=0)
			all_best_std = np.std(all_best_during_run, axis=0)
			actions_avg =  np.average(all_actions, axis=0)
			actions_std = np.std(all_actions, axis=0)
			axs[0].plot(x, all_best_avg, color=first_col, lw=2, zorder=3 * (i + 1) + 2, label=exp_names[i])
			axs[0].fill_between(
				x,
				all_best_avg - all_best_std,
				all_best_avg + all_best_std,
				color=second_col,
				alpha=1 / (1 + i),
				zorder=3 * (i + 1),
			)
			axs[1].plot(x, actions_avg, color=first_col, lw=2, zorder=3 * (i + 1) + 2, label=exp_names[i])
			axs[1].fill_between(
				x,
				actions_avg - actions_std,
				actions_avg + actions_std,
				color=second_col,
				alpha=1 / (1 + i),
				zorder=3 * (i + 1),
			)
			min_plot = min(min_plot, np.min(all_best_avg - all_best_std))
			max_plot = max(max_plot, np.max(all_best_avg + all_best_std))
			
	print("max value: {}".format(max_fit))
	print("min value: {}".format(min_fit))

	# configure plots

	# Top
	if logyscale:
		if min_fit < 0:
			shift = 1e-2
			shifted_min_plot = min_plot - shift  # to avoid log(0)
			exp = lambda x: (2) ** (x) + shifted_min_plot
			log = lambda x: np.log(x - shifted_min_plot) / np.log(2)
			axs[0].set_yscale("function", functions=(log, exp))
			axs[0].set_yticks(np.geomspace(shift, max_plot - min_fit + shift, num=10) + min_fit - shift)
			axs[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.3f}"))
		else:
			axs[0].set_yscale("log", subs=[2, 4, 6, 8])
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
	axs[1].set_ylabel(action_axis_label, labelpad=0)
	axs[1].grid(True, which="both")
	axs[1].tick_params(axis="y", which="minor", grid_alpha=0.3)


	# compute statistical metrics
	if len(exp_list) == 2:
		auc = compute_metrics_comparison(exp_list, "AUC")
		final_best = compute_metrics_comparison(exp_list, "final_best")
		print("with AUC metric: p({} < {}) = {}".format(exp_names[0], exp_names[1], auc))
		print("with final best metric: p({} < {}) = {}".format(exp_names[0], exp_names[1], final_best))

	plt.legend()
	if save:
		fig.savefig(save + "/{}.png".format(title))
	else:
		plt.show()


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

		min_fit = np.inf
		for i, experiment in enumerate(exp_list):
			for j, run in enumerate(experiment):
				min_fit = min(np.min(run["fitness"]), min_fit)
		# raise the values with the -minimum to bring them above zero
		# so the area can't be decreased with negative values
		shift = -min_fit + 1e-5 if min_fit < 0 else 0

		for i, experiment in enumerate(exp_list):
			for j, run in enumerate(experiment):
				# better to use the minimums or the averages of the populations?
				min_fit_run = getBestDuringRun(run["fitness"]) + shift
				# avg_fit_run = np.average(run["fitness"], axis=1)

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

	# create an array with 1 if the metric in experiment 0 is less than experiment 2, 0 otherwise
	# sum the array and divide it for its length to get a probablity
	# see Equation 1 in "Learning Step-Size Adaptation in CMA-ES"
	prob = np.sum([1 if c[0] < c[1] else 0 for c in combinations]) / combinations.shape[0]
	return prob


def save_experiment(experiment, folder, name="experiment"):
	name = name if isinstance(name, str) else "experiment"
	with open(folder + "/{}.bin".format(name), "wb") as f:
		pickle.dump(experiment, f)


def load_experiment(dir_or_file):
	if isdir(dir_or_file):
		dir_or_file = dir_or_file + "/experiment.bin"
	with open(dir_or_file, "rb") as f:
		return pickle.load(f)
