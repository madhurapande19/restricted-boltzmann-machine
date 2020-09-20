import numpy as np
import random
import matplotlib.pyplot as plt
from data_utils import *
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import seaborn as sns
import argparse as ap
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

def sig(x):
	return 1/(1+np.exp(-x))


def generate_sample(distbn):
	sample_size = distbn.shape[1]
	batch_size = distbn.shape[0]
	rand_nos = np.random.uniform(0, 1, (batch_size, sample_size))
	sample = (rand_nos < distbn).astype(int)
	return sample

def visualize_image(x, idx):
	plt.imshow(np.reshape(x, (28,28)), cmap='gray')
	plt.savefig('image'+str(idx)+'.jpg')
	plt.show()

"""
Reference for t-SNE plot: 
https://www.datacamp.com/community/tutorials/introduction-t-sne
"""
def scatter_tsne(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    plt.savefig('tsne_plot.jpg')
    plt.show()
    return f, ax, sc, txts

class RBM():

	def __init__(self, input_size, num_hidden=128):

		self.input_size = input_size
		self.num_hidden = num_hidden
		self.W = np.random.normal(scale=0.1, size=(num_hidden, input_size))
		self.b = np.zeros(input_size)
		self.c = np.zeros(num_hidden)

	
	def forward(self, x):
		y = np.dot(x, self.W.T) + self.c
		return sig(y)

	def backward(self, x):
		y = np.dot(x, self.W) + self.b
		return sig(y)

	def reconstruct(self, x):
		h = self.forward(x)
		x_reconstructed = self.backward(h)
		return x_reconstructed

	def update_w(self, v_d, v_tilde):
		temp1 = self.forward(v_d)
		update1 = np.dot(temp1.T, v_d)

		temp2 = self.forward(v_tilde)
		update2 = np.dot(temp2.T, v_tilde)

		return (update1 - update2)

	def update_b(self, v_d, v_tilde):
		return np.sum((v_d - v_tilde), axis=0)

	def update_c(self, v_d, v_tilde):
		temp1 = self.forward(v_d)
		temp2 = self.forward(v_tilde)
		return np.sum((temp1 - temp2), axis=0)

		

	def train(self, x, learning_rate, epochs, k, batch_size):

		training_errors = []
		rec_error_per_sample = []

		for itr in range(epochs):
			batch_errors = []
			print('Starting epoch '+str(itr))
			for batch in get_batch(x, batch_size=batch_size):

				for sampling_iters in range(k):
					
					#sample from P(H|V)
					h_distbn = self.forward(batch)
					h_sample = generate_sample(h_distbn)
				
					#sample from P(V|H)
					v_distbn = self.backward(h_sample)
					v_sample = generate_sample(v_distbn)



				# Check reconstruction error before weight update
				if itr <= 100:
					batch_errors.append(np.mean((batch - v_sample) ** 2))

				#update w,b,c
				self.W += learning_rate * self.update_w(batch, v_sample)
				self.b += learning_rate * self.update_b(batch, v_sample)
				self.c += learning_rate * self.update_c(batch, v_sample) 


			training_errors.append(np.mean(batch_errors))
				
		return training_errors, batch_errors



parser = ap.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--epochs")
parser.add_argument("--k")
args = parser.parse_args()
learning_rate = float(args.lr)
epochs = int(args.epochs)
k = int(args.k)
# Load the data
data = load_data('train.csv')
x_train = data['x']
if data['y'] is not None:
	y_train = data['y']
print(x_train.shape)

# Threshold data
x_train = prepare_data(x_train)


rbm = RBM(x_train.shape[1])
# Batch size = 1 (Stochastic Gradient Descent)
training_errors, batch_errors = rbm.train(x_train, learning_rate, epochs, k, 1)
print('training_errors')

# Saving reconstruction errors in file
np.savetxt('reconstruction_error.txt', np.array(training_errors))
np.savetxt('batch_errors.txt', np.array(batch_errors), fmt='%.4f')


#Loading test data
data = load_data('test.csv', isSupervised=True, isLabelled=True)
x_test = data['x']
y_test = data['y']

x_test = prepare_data(x_test)

h_test = rbm.forward(x_test)

# t-SNE plot
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_output = tsne.fit_transform(h_test)
print('tsne done')
scatter_tsne(tsne_output, y_test.astype(int))

# Check reconstruction on trained RBM
# print('Original images')
# visualize_image(x_test[0, :],1)
# visualize_image(x_test[1, :],2)
# visualize_image(x_test[2, :],3)
# visualize_image(x_test[3, :],4)
# h1 = rbm.forward(x_test[0, :])
# x1 = rbm.backward(h1)

# h2 = rbm.forward(x_test[1, :])
# x2 = rbm.backward(h2)

# h3 = rbm.forward(x_test[2, :])
# x3 = rbm.backward(h3)

# h4 = rbm.forward(x_test[3, :])
# x4 = rbm.backward(h4)

# print('Reconstructed images')
# visualize_image(x1, 5)
# visualize_image(x2, 6)
# visualize_image(x3, 7)
# visualize_image(x4, 8)