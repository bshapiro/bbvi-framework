from autograd.numpy.random import choice
from black_box_vi_base import BaseModel
from collections import defaultdict
from vi_helpers import softplus, visualize_clusters, generate_clusters
import autograd.numpy as np
import autograd.scipy.stats.multivariate_normal as mvn
import matplotlib.pyplot as plt


class Model(BaseModel):

    def __init__(self, param_location=None, data=None, K=4):
        data = generate_clusters(100, 0.3)
        super(Model, self).__init__(param_location=param_location, data=data)
        self.K = K
        self.D = data.shape[1]
        self.N = data.shape[0]
        self.add_parameter('pi', self.K, -1, -5, 0.1, 0.1)
        for cluster in range(self.K):
            self.add_parameter('cluster_mean' + str(cluster), self.D, data[choice(range(self.N)), :], -5, 1, 0.1)
            self.add_parameter('cluster_sigma' + str(cluster), self.D, -1, -5, 0.1, 0.1)
        self.z = choice(range(self.K), self.N)

    def log_density(self, param_samples, t):
        param_dict = self.unpack_param_samples(param_samples)

        sample_pis = param_dict['pi']
        sample_pis = softplus(param_samples[:, 0:self.K])
        sample_pis = (sample_pis.T / np.sum(sample_pis, axis=1)).T

        likelihoods = 0
        cluster_samples = defaultdict(list)
        for sample_index in range(len(self.z)):
            cluster_samples[self.z[sample_index]].append(sample_index)

        for cluster in range(self.K):
            sample_indices = cluster_samples[cluster]
            likelihood = np.array([np.log(sample_pis[i, cluster]) * len(sample_indices) +
                                   np.sum(mvn.logpdf(self.data[sample_indices, :],
                                                     param_dict['cluster_mean' + str(cluster)][i, :],
                                                     np.diag(softplus(param_dict['cluster_sigma' + str(cluster)][i, :]))))
                                   for i in range(len(param_samples))])

            likelihoods = likelihoods + likelihood

        return np.array(likelihoods)

    def callback(self, params, t, g):
        print(g)
        param_dict = self.unpack_params(params)
        cluster_probabilities = []
        for cluster in range(self.K):
            mean = param_dict['cluster_mean' + str(cluster)][0]
            sigma = param_dict['cluster_sigma' + str(cluster)][0]
            probabilities = mvn.logpdf(self.data, mean, np.diag(softplus(sigma)))
            cluster_probabilities.append(probabilities)
        pi = softplus(param_dict['pi'][0])
        pi = pi / np.sum(pi)
        total_probabilities = np.array(cluster_probabilities).T + np.log(pi)
        self.z = np.argmax(total_probabilities, axis=1)

    def visualize(self, ax, params, iter):
        plt.cla()
        param_dict = self.unpack_params(params)
        means = []
        sigmas = []
        for cluster in range(self.K):
            mean = param_dict['cluster_mean' + str(cluster)][0]
            sigma = np.diag(softplus(param_dict['cluster_sigma' + str(cluster)][0]))
            print(mean, sigma)
            means.append(mean)
            sigmas.append(sigma)

        visualize_clusters(ax, self.data, self.z, means, sigmas)
        plt.draw()
        plt.pause(1.0 / 30.0)
