import numpy as np
from pprint import pprint
import sys
sys.path.insert(1, "./")
from metrics import *

def testMetric(metric: Metric, iterations, pop=10, dim=5):
	for i in range(iterations):
		solutions = np.random.uniform(low=-5, high=5, size=(pop,dim))
		fitness = np.random.normal(loc=0, scale=i, size=(pop,))
		ret = metric.compute(solutions, fitness)
		print()
		print("best:", metric.best)
		print(ret)


if __name__ == "__main__":
	# m0 = RecentGradients(10, 6, 1, None, 3)
	# m1 = DifferenceOfBest(4)
	m2 = DeltaBest(False, True, False)
	testMetric(m2, 10)