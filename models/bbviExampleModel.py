from black_box_vi_base import BaseModel
import autograd.scipy.stats.norm as norm
import autograd.numpy as np
import matplotlib.pyplot as plt
from vi_helpers import plot_isocontours, softplus
import autograd.scipy.stats.multivariate_normal as mvn


class Model(BaseModel):
    """
    Original example black box model from
    the paper on bbvi in 5 lines.
    """

    def __init__(self, param_location=None, data=None):
        super(Model, self).__init__(param_location=param_location, data=data)

    def log_density(self, param_samples, t):
        param_sample_dict = self.unpack_param_samples(param_samples)
        gaussian1_samples = param_sample_dict['gaussian1']

        dim1 = gaussian1_samples[:, 0]
        dim2 = gaussian1_samples[:, 1]
        sigma_density = norm.logpdf(dim2, 0, 1.35)
        mu_density = norm.logpdf(dim1, 0, np.exp(dim2))

        return sigma_density + mu_density

    def visualize(self, ax, params, i):
        param_dict = self.unpack_params(params)
        mean, std = param_dict['gaussian1'][0], param_dict['gaussian1'][1]

        plt.cla()
        target_distribution = lambda x: np.exp(self.log_density(x, i))
        plot_isocontours(ax, target_distribution)

        variational_contour = lambda x: mvn.pdf(x, mean, np.diag(2 * softplus(std)))
        plot_isocontours(ax, variational_contour)
        plt.draw()
        plt.pause(1.0 / 30.0)
