from black_box_vi_base import BaseModel
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.numpy as np
from vi_helpers import softplus


class Model(BaseModel):
    """
    Random multivariate normal model.
    """

    def __init__(self, param_location=None, data=None):
        super(Model, self).__init__(param_location=param_location, data=data)
        self.D = 2
        self.distributions = [('mu', 'gaussian', self.D, 0), ('sigma', 'gaussian', self.D, 4)]
        self.variational_params = np.concatenate([-1 * np.ones(self.D), -5 * np.ones(self.D), -1 * np.ones(self.D), -5 * np.ones(self.D)])
        print(self.variational_params)

    def log_density(self, param_samples, t):
        total_log_prob = np.array([np.sum(mvn.logpdf(self.data, sample[0:2], np.diag(softplus(sample[2:4])))) for sample in param_samples])
        return total_log_prob

    def callback(self, params, t, g):
        print("GRADIENTS:", g)
        print("mu", params[0:2], \
              "sigma", np.diag(softplus(params[4:6])))
