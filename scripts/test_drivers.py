import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(1, "./")
from drivers import RastriginGADriver, CMAdriver

class plot_helper:
	def __init__(self) -> None:
		self.lines = []
		self.data = []

	def plot(self, num_evaluations, fitness):
		max_fitness = np.max(fitness)
		min_fitness = np.min(fitness)
		median_fitness = np.median(fitness)
		average_fitness = np.mean(fitness)
		colors = ['black', 'blue', 'green', 'red']
		labels = ['average', 'median', 'max', 'min']
		if num_evaluations == 0:
			plt.figure("plot data")
			plt.ion()
			self.data = [[num_evaluations], [average_fitness], [median_fitness], [max_fitness], [min_fitness]]
			for i in range(4):
				line, = plt.plot(self.data[0], self.data[i+1], color=colors[i], label=labels[i])
				self.lines.append(line)
			plt.xlabel('Evaluations')
			plt.ylabel('Fitness')
		else:
			self.data[0].append(num_evaluations)
			self.data[1].append(average_fitness)
			self.data[2].append(median_fitness)
			self.data[3].append(max_fitness)
			self.data[4].append(min_fitness)
			for i, line in enumerate(self.lines):
				line.set_xdata(np.array(self.data[0]))
				line.set_ydata(np.array(self.data[i+1]))
		ymin = min([min(d) for d in self.data[1:]])
		ymax = max([max(d) for d in self.data[1:]])
		yrange = ymax - ymin
		plt.xlim((0, num_evaluations))
		plt.ylim((ymin - 0.1*yrange, ymax + 0.1*yrange))
		plt.draw()
		plt.pause(0.00001)
		plt.legend()
		plt.show()
		return max_fitness, min_fitness, average_fitness

if __name__ == "__main__":
	driver = CMAdriver(10, 10, init_sigma=1.63, object_function=1, max_steps=None, seed=42)
	driver.initialize()
	driver.reset()
	p=plot_helper()
	for i in range(100):
		pop, fit, _ = driver.step({"step_size": 0.5})
		maximum, minimum, avg = p.plot(i, fit)
		# driver.render(i==99)

	print("Max:\t",maximum)
	print("Min:\t",minimum)
	print("Avg:\t",avg)
	plt.ioff()
	plt.show()
