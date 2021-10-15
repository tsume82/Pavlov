import matplotlib.pyplot as plt
import numpy as np
from os.path import exists, splitext, dirname
from pprint import pformat

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
		while exists(path):
			path = splitted_path[0] + "_" + str(i) + splitted_path[1]
			i+=1

		formatted_infos = pformat(infos)
		if info_file_name and isinstance(info_file_name, str):
			with open(dirname(path)+"/"+info_file_name, "w") as f:
				f.write(formatted_infos)

		if splitted_path[1].lower() in ["png", "svg"]: # Description keyword works only in SVG and PNG
			plt.savefig(path, metadata={"Description": formatted_infos }) 
		else:
			plt.savefig(path)

	def __plot_multi_episodes(self, cumulative_rewards):
		colors = ["blue"]
		labels = ["cumulative reward"]
		if self.num_episodes == 0:
			plt.figure("plot data")
			plt.ion()
			self.data = [[*range(len(cumulative_rewards))], cumulative_rewards]
			for i in range(1):
				(line,) = plt.plot(self.data[0], self.data[i + 1], color=colors[i], label=labels[i])
				self.lines.append(line)
			plt.xlabel("Episodes")
			plt.ylabel("Cumulative Reward")
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
		plt.xlim((0, self.num_episodes))
		plt.ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
		plt.draw()
		plt.pause(0.001)
		plt.legend()
		# plt.show()

	def __plot_single_episode(self, cumulative_reward):
		colors = ["blue"]
		labels = ["cumulative_reward"]
		if self.num_episodes == 0:
			plt.figure("plot data")
			plt.ion()
			self.data = [[self.num_episodes], [cumulative_reward]]
			for i in range(1):
				(line,) = plt.plot(self.data[0], self.data[i + 1], color=colors[i], label=labels[i])
				self.lines.append(line)
			plt.xlabel("Episodes")
			plt.ylabel("Cumulative Reward")
		else:
			self.data[0].append(self.num_episodes)
			self.data[1].append(cumulative_reward)

			for i, line in enumerate(self.lines):
				line.set_xdata(np.array(self.data[0]))
				line.set_ydata(np.array(self.data[i + 1]))

		ymin = min([min(d) for d in self.data[1:]])
		ymax = max([max(d) for d in self.data[1:]])
		yrange = ymax - ymin
		plt.xlim((0, self.num_episodes))
		plt.ylim((ymin - 0.1 * yrange, ymax + 0.1 * yrange))
		self.num_episodes += 1
		plt.draw()
		plt.pause(0.00001)
		plt.legend()
		plt.show()
