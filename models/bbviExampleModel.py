from black_box_vi_base import BaseModel
import autograd.scipy.stats.norm as norm
import autograd.numpy as np


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

    def callback(self, variational_params, t, g):
        pass
