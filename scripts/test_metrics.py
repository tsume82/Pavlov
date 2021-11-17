import numpy as np
from pprint import pprint
import sys
sys.path.insert(1, "./")
from metrics import *

def testMetric(metric: Metric, iterations, pop=10, dim=5):
	for i in range(iterations):
		solutions = np.random.uniform(low=-5, high=5, size=(pop,dim))
		fitness = np.random.normal(loc=0, scale=i+1, size=(pop,))
		ret = metric.compute(solutions, fitness)
		print()
		print("best:", metric.prec_best)
		print(ret)


if __name__ == "__main__":
	# m = RecentGradients(10, 6, 1, None, 3)
	m = DifferenceOfBest(4, False, 1, True, True)
	# m = DeltaBest(False, True, False)
	testMetric(m, 10)