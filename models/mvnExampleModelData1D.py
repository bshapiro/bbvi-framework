from black_box_vi_base import BaseModel
from scipy import stats
from vi_helpers import softplus
import autograd.numpy as np
import autograd.scipy.stats.norm as norm
import matplotlib.pyplot as plt


class Model(BaseModel):
    """
    Random multivariate normal model.
    """
    def __init__(self, param_location=None, data=None):
        super(Model, self).__init__(param_location=param_location, data=data)
        self.data = np.random.normal(2, 0.5, size=500)

    def log_density(self, param_samples, t):
        param_sample_dict = self.unpack_param_samples(param_samples)
        mu = param_sample_dict['mu']
        sigma = param_sample_dict['sigma']

        total_log_prob = np.array([np.sum(norm.logpdf(self.data, mu[i, :], np.sqrt(softplus(sigma[i, :])))) for i in range(mu.shape[0])])
        return total_log_prob

    def visualize(self, ax, params, i):
        param_dict = self.unpack_params(params)
        mu = param_dict['mu'][0]
        std = np.sqrt(softplus(param_dict['sigma'][0]))
        x = np.linspace(-5, 5, 100)
        plt.plot(x, stats.norm.pdf(x, mu, std))
        plt.draw()
        plt.pause(1.0 / 30.0)
