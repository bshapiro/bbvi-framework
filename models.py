from black_box_vi_base import *
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm
from autograd.numpy.random import choice
import autograd.numpy as np
from vi_helpers import *
import matplotlib.pyplot as plt
from collections import Counter
from collections import defaultdict


class bbviExampleModel(Model):
    """
    Original example black box model from
    the paper on bbvi in 5 lines.
    """

    def __init__(self, data=None):
        super(bbviExampleModel, self).__init__()
        self.D = 2
        self.distributions = [('test', 'gaussian', self.D, 0)]
        self.variational_params = [-1 * np.ones(self.D), -5 * np.ones(self.D)]

    def log_density(self, param_samples, t):
        mu, log_sigma = param_samples[:, 0], param_samples[:, 1]
        sigma_density = norm.logpdf(log_sigma, 0, 1.35)
        mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))
        return sigma_density + mu_density

    def callback(self, variational_params, t, g):
        pass


class mvnExampleModelData1D(Model):
    """
    Random multivariate normal model.
    """
    def __init__(self, data=None):
        super(mvnExampleModelData1D, self).__init__(data)
        self.D = 1
        self.distributions = [('mu', 'gaussian', self.D, 0), ('sigma', 'gaussian', self.D, 2)]
        self.variational_params = np.concatenate([[-1.], [-5.], [-1.], [-5.]])

    def log_density(self, param_samples, t):
        bootstrap = np.array(self.data)
        total_log_prob = np.array([np.sum(norm.logpdf(bootstrap, mu, np.sqrt(softplus(sigma)))) for mu, sigma in param_samples])
        return total_log_prob

    def log_var(self, param_samples, params):
        log_prob = norm.logpdf(param_samples[:, 0], params[0], np.sqrt(softplus(params[1]))) + \
                   norm.logpdf(param_samples[:, 1], params[2], np.sqrt(softplus(params[3])))
        return log_prob

    def callback(self, params, t, g):
        print("GRADIENTS:", g)
        print("mu", params[0:1], "std", np.sqrt(softplus(params[2:3])))
        sigma_inputs = np.arange(-10, 100, 0.1)
        sigma_outputs = softplus(sigma_inputs)
        plt.scatter(sigma_inputs, sigma_outputs)
        plt.show()
        import pdb; pdb.set_trace()


class mvnExampleModelData2D(Model):
    """
    Random multivariate normal model.
    """

    def __init__(self, data=None):
        super(mvnExampleModelData2D, self).__init__(data)
        self.D = 2
        self.distributions = [('mu', 'gaussian', self.D, 0), ('sigma', 'gaussian', self.D, 4)]
        self.variational_params = np.concatenate([-1 * np.ones(self.D), -5 * np.ones(self.D), -1 * np.ones(self.D), -5 * np.ones(self.D)])
        print(self.variational_params)

    def log_density(self, param_samples, t):
        bootstrap = np.array(self.data)
        total_log_prob = np.array([np.sum(mvn.logpdf(bootstrap, sample[0:2], np.diag(softplus(sample[2:4])))) for sample in param_samples])
        return total_log_prob

    def log_var(self, param_samples, params):
        log_prob = mvn.logpdf(param_samples[:, 0:2], params[0:2], np.diag(softplus(params[2:4]))) + \
                   mvn.logpdf(param_samples[:, 2:4], params[4:6], np.diag(softplus(params[6:8])))
        return log_prob

    def callback(self, params, t, g):
        print("GRADIENTS:", g)
        print("mu", params[0:2], \
              "sigma", np.diag(softplus(params[4:6])))


class gmmExampleModel(Model):

    def __init__(self, data=None, K=5):
        super(gmmExampleModel, self).__init__(data)
        self.K = K
        self.D = data.shape[1]
        self.N = data.shape[0]
        self.distributions = [('pi', 'gaussian', self.K, 0)]
        self.variational_params = [-1 * np.ones(self.K), -5 * np.ones(self.K)]
        for cluster in range(self.K):
            self.distributions.append(('mu' + str(cluster), 'gaussian', self.D, self.K*2 + cluster*self.D*4))
            self.distributions.append(('sigma' + str(cluster), 'gaussian', self.D, self.K*2 + cluster*self.D*4 + self.D*2))
            self.variational_params.extend([self.data[choice(range(self.N))], -5 * np.ones(self.D)])
            self.variational_params.extend([-1 * np.ones(self.D), -5 * np.ones(self.D)])
        self.z = choice(range(self.K), self.N)
        # print self.distributions

    def log_density(self, param_samples, t):
        random_indices = range(self.N)
        # random_indices = choice(range(self.N), 1000)
        sample_pis = softplus(param_samples[:, 0:self.K])
        sample_pis = (sample_pis.T / np.sum(sample_pis, axis=1)).T

        num_param_samples = param_samples.shape[0]
        likelihoods = 0

        cluster_samples = defaultdict(list)
        for sample_index in range(len(self.z)):
            cluster_samples[self.z[sample_index]].append(sample_index)

        for cluster in range(self.K):
            cpi = self.K + cluster*self.D*2  # cluster param sample index
            sample_indices = cluster_samples[cluster]
            likelihood = np.array([np.log(sample_pis[i, cluster]) * len(sample_indices) +
                                   np.sum(mvn.logpdf(self.data[sample_indices],
                                                     param_samples[i, cpi:cpi + self.D],
                                                     np.diag(softplus(param_samples[i, cpi + self.D:cpi+2*self.D]))))
                                   for i in range(len(param_samples))])
            likelihoods += likelihood
        return np.array(likelihoods)

    def log_var(self, param_samples, params):
        log_prob = mvn.logpdf(param_samples[:, 0:self.K], params[0:self.K], np.diag(softplus(params[self.K:self.K*2])))
        sample_index = self.K
        param_index = self.K * 2
        for cluster_param in range(self.K*2):
            log_prob += mvn.logpdf(param_samples[:, sample_index:sample_index+self.D],
                                   params[param_index:param_index+self.D],
                                   np.diag(softplus(params[param_index+self.D: param_index+self.D*2])))
            sample_index += self.D
            param_index += self.D * 2
        return log_prob

    def callback(self, params, t, g):
        cluster_probabilities = []
        for cluster in range(self.K):
            cpi = self.K + cluster*self.D*4  # cluster param sample index
            probabilities = mvn.logpdf(self.data, params[cpi:cpi + self.D], np.diag(softplus(params[cpi + self.D*2:cpi+3*self.D])))
            cluster_probabilities.append(probabilities)
        pi = softplus(params[:self.K])
        pi = pi / np.sum(pi)
        total_probabilities = np.array(cluster_probabilities).T + np.log(pi)
        self.z = np.argmax(total_probabilities, axis=1)
        print(Counter(self.z))
        print(pi)
