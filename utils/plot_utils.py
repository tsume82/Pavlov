import matplotlib.pyplot as plt
from matplotlib.widgets import Button 
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import numpy as np
from os.path import exists, splitext, dirname
import json

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

def plot_experiment(experiment, title="", title_act="step size"):
	fig, axs = plt.subplots(2, sharex=True)
	# Top
	axs[0].title.set_text(title)
	axs[0].set_yscale("log", subs=[2,4,6,8])
	axs[0].grid(True, which="both")
	axs[0].tick_params(axis='y', which="minor", grid_alpha=0.3)
	# Bottom
	axs[1].title.set_text(title_act)
	axs[1].yaxis.set_minor_locator(AutoMinorLocator(2))
	axs[1].grid(True, which="both")
	axs[1].tick_params(axis='y', which="minor", grid_alpha=0.3)
	length = len(experiment[0]["fitness"])
	avg = np.empty(shape=[0,length])
	for traj in experiment:
		actions = []
		for step in traj["actions"]:
			actions.append(step.get("step_size", None)) # TODO handle different actions spaces
		popAvg = np.average(traj["fitness"], axis=1)
		avg = np.vstack([avg,popAvg])
		axs[0].plot([*range(length)], popAvg, color="blue", alpha=0.4)
		axs[1].plot([*range(1,length)], actions, color="black", alpha=0.4)
	axs[0].plot([*range(length)], np.average(avg, axis=0), color="red", alpha=0.5)
	plt.show()