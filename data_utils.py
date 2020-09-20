import numpy as np

def load_data(filename, isSupervised=False, isLabelled=True ):
	
	data = np.loadtxt(filename, delimiter=',', skiprows=1)
	
	if isLabelled: 
		x = data[:, 1:-1]
		if isSupervised: 
			y = data[:, -1]
			return {'x': x, 'y': y}
	return {'x': x, 'y': None}


def prepare_data(data, threshold=127):
	return (data >= 127).astype(int)


def get_batch(x, y=None, batch_size=64):
	n = x.shape[0]
	for i in range(0, n, batch_size):
		start_idx, end_idx = i, min(i+batch_size, n)
		if y is not None:
			yield(x[start_idx:end_idx, :], y[start_idx:end_idx, :])
		else:
			yield(x[start_idx:end_idx, :])


