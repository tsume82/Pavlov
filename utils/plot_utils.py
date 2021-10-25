import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.widgets import Button
from matplotlib.ticker import AutoMinorLocator, StrMethodFormatter, ScalarFormatter, LinearLocator, LogLocator
import numpy as np
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

	def save(self, path, infos, info_file_name=None):
		i = 1
		splitted_path = splitext(path)
		path = splitted_path[0] + "_" + str(i) + splitted_path[1]
		while exists(path):
			i+=1
			path = splitted_path[0] + "_" + str(i) + splitted_path[1]

		formatted_infos = json.dumps(infos, skipkeys=True)
		if info_file_name and isinstance(info_file_name, str):
			with open(dirname(path)+"/"+info_file_name, "w") as f:
				f.write(formatted_infos)

		if splitted_path[1].lower() in [".png", ".svg"]: # Description keyword works only in SVG and PNG
			self.fig.savefig(path, metadata={"Description": formatted_infos }) 
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

def plot_experiment(experiment, title="Plot Experiment", title_act="step size", logyscale=False, logxscale=True):
	if isdir(experiment):
		experiment = load_experiment(experiment)
		
	fig, axs = plt.subplots(2, sharex=True, figsize=(12,6))
	fig.tight_layout()
	fig.canvas.set_window_title(title)

	length = len(experiment[0]["fitness"])
	avg = np.empty(shape=[0,length])
	min_fit = np.inf
	max_fit = -np.inf
	for traj in experiment:
		actions = []
		for step in traj["actions"]:
			actions.append(step.get("step_size", None)) # TODO handle different actions spaces
		popAvg = np.average(traj["fitness"], axis=1)
		min_fit = min(np.min(traj["fitness"]), min_fit)
		max_fit = max(np.max(traj["fitness"]), max_fit)
		popmin = np.min(traj["fitness"], axis=1)
		popmax = np.max(traj["fitness"], axis=1)
		avg = np.vstack([avg,popAvg])
		axs[0].fill_between([*range(length)], popmin, popmax, color="darkorange", alpha=0.15)
		axs[0].plot([*range(length)], popAvg, color="blue", alpha=0.4)
		axs[1].plot([*range(1,length)], actions, color="black", alpha=0.4)
	axs[0].plot([*range(length)], np.average(avg, axis=0), color="red", alpha=0.5)

	print("minimum value: {}".format(min_fit))

	# Top
	axs[0].title.set_text(title)
	if logyscale:
		if min_fit < 0:
			shift = 1e-9
			shifted_min_fit = min_fit - shift # to avoid log(0)
			exp = lambda x: (2)**(x)+shifted_min_fit
			log = lambda x: np.log(x-shifted_min_fit)/np.log(2)
			axs[0].set_yscale('function', functions=(log, exp))
			axs[0].set_yticks(np.geomspace(shift, max_fit-min_fit+shift, num=10)+min_fit-shift)
			axs[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
		else:
			axs[0].set_yscale("log", subs=[2,4,6,8])
	else:
		axs[0].set_yscale("linear")
		axs[0].yaxis.set_minor_locator(AutoMinorLocator(2))
	ticks = list(axs[0].get_yticks()) + [min_fit]
	if logxscale:
		axs[0].set_xscale("log")
		axs[0].xaxis.set_major_formatter(ScalarFormatter())

	# axs[0].set_yticks(ticks)
	# axs[0].set_yticklabels([np.format_float_scientific(t, precision = 3, unique=True) for t in  ticks])
	axs[0].grid(True, which="both")
	axs[0].set_ylabel("fitness", labelpad=0)
	axs[0].tick_params(axis='y', which="minor", grid_alpha=0.3)

	# customization to match the paper's plots
	axs[0].set_xticks([1,10,50,100,200,300,400,500])
	axs[0].set_ylim((-5e5, 7e6))
	axs[0].set_yticks([6000000,4000000,2000000,0])

	# Bottom
	axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))
	axs[1].set_xlabel("function evaluations", labelpad=0)
	axs[1].set_ylabel(title_act, labelpad=0)
	axs[1].grid(True, which="both")
	axs[1].tick_params(axis='y', which="minor", grid_alpha=0.3)

	plt.show()

def save_experiment(experiment, folder):
	with open(folder + '/experiment.bin', 'wb') as f:
		pickle.dump(experiment, f)

def load_experiment(folder):
	with open(folder+"/experiment.bin", 'rb') as f:
		return pickle.load(f)