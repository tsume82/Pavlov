from numpy import isscalar, ndarray
"""
	convert an array representing one value to a number
"""
def getScalar(x):
	if isscalar(x):
		return x
	if isinstance(x, ndarray):
		if len(x.shape) == 0:
			return x[()]
		if x.shape[0] == 1:
			return x[0]
	return x