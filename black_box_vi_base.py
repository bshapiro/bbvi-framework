from autograd import grad
from autograd.misc import flatten
from copy import copy
from vi_helpers import regularize, softplus, size
import autograd.builtins as btn
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import json


class BaseModel(object):
    """
    Base model class. Sets up parameter handling.
    Every new model should extend the model class and override
    the log_density method.
    """

    def __init__(self, param_location=None, data=None):
        """
        Make sure to extend this class so that the below initialization
        logic still occurs in your model. Then define any additional variables
        needed to compute the log density.
        Optionally, include any additional initialization logic.
        """
        self.data = data
        self.variational_params = []
        self.param_info = {}
        self.param_index = 0
        self.param_sample_index = 0
        self.step_size = []

        if param_location is not None:
            param_file = open(param_location)
            params = json.load(param_file)

            for param in params['params']:
                self.add_parameter(param['name'],
                                   param['length'],
                                   param['mean_init'],
                                   param['sigma_init'],
                                   param['mean_step'],
                                   param['sigma_step'])

    def _config(self):
        """
        Sets the model up for the VI run. Internal use.
        """
        self.variational_params = np.hstack(self.variational_params)

    def add_parameter(self, name, length, mean_init, sigma_init, mean_step, sigma_step):
        """
        Use this method to add any parameters dynamically when you initialize your model.
        Then access your parameters from log_density using the dictionary produced by
        unpack_param_samples. All bookkeeping is done for you!
        """
        if size(mean_init) == 1:
            self.variational_params.extend(mean_init * np.ones(length))
        elif len(mean_init) == length:
            self.variational_params.extend(np.array(mean_init))
        else:
            print("Invalid length for mean initialization of parameter", name)
        if size(sigma_init) == 1:
            self.variational_params.extend(sigma_init * np.ones(length))
        elif len(sigma_init) == length:
            self.variational_params.extend(np.array(sigma_init))
        else:
            print("Invalid length for sigma initialization of parameter", name)
        self.step_size.extend([mean_step] * length)
        self.step_size.extend([sigma_step] * length)
        self.param_info[name] = (copy(self.param_sample_index), copy(self.param_index), length)
        self.param_index += length * 2
        self.param_sample_index += length

    def unpack_param_samples(self, param_samples):
        """
        Processes parameter samples into dictionary indexed by parameter name.
        """
        param_dict = {}
        for param in self.param_info.keys():
            sample_index, param_index, length = self.param_info[param]
            param_dict[param] = param_samples[:, sample_index:sample_index + length]
        return param_dict

    def log_density(self, param_samples, t):
        """
        This method should be overriden. It should return
        the joint log density of the data and each set of parameter
        samples using the names defined in the dictionary returned by
        a call to unpack_param_samples(param_samples).
        """
        pass

    def callback(self, params, iter, gradient):
        """
        Override this method to execute any changes to the model that need to
        be accomplished between iterations, for example to update discrete
        variables or to perform EM steps.
        """
        pass

    def visualize(self, ax, params, iter):
        """
        Override this method to execute any visualization that you'd like to occur
        between iterations.
        'ax' is a pointer to the matptlotlib figure Axes object.
        """
        pass

    def update(self, params):
        """
        OVerride this method to update the parameters between inference steps.
        Changes to the parameters will NOT be included in gradient calculations.
        """
        return params

    def unpack_params(self, params):
        """
        Unpacks parameters into dictionary. For each parameter,
        the dictionary contains a tuple for the variational distribution
        (mean, sigma).
        """
        param_dict = {}
        for param in self.param_info.keys():
            sample_index, param_index, length = self.param_info[param]
            param_dict[param] = (params[param_index:param_index + length], params[param_index + length:param_index + length * 2])
        return param_dict

    def _log_var(self, param_samples, variational_params):
        """
        Automatically computes the log density of the param samples according to
        the variational parameters (e.g. log q(z | lambda)).
        Used in ELBO estimation (internal use).
        """
        log_prob = 0
        param_dict = self.unpack_params(param_samples)
        for key in param_dict.keys():
            sample_index, param_index, length = self.param_info[key]
            log_prob = log_prob + mvn.logpdf(param_samples[:, sample_index:sample_index + length],
                                             variational_params[param_index:param_index + length],
                                             np.diag(softplus(variational_params[param_index + length:param_index + length*2])))

        return log_prob


def black_box_variational_inference(logprob, logvar, param_info, num_samples, random_state=None):
    """Implements http://arxiv.org/abs/1401.0118, and uses the
    local reparameterization trick from http://arxiv.org/abs/1506.02557"""

    def unpack_params(params):
        return flatten(params)[0]

    def gaussian_entropy(log_std):
        return 0.5 * len(log_std) * (1.0 + np.log(2 * np.pi)) + np.sum(log_std)

    rs = npr.RandomState(random_state)

    def variational_objective(params, t):
        variational_params = unpack_params(params)
        samples = btn.list([])
        for key in param_info.keys():
            sample_index, param_index, length = param_info[key]
            mean = np.array(variational_params[param_index:param_index + length])
            cov = regularize(softplus(np.array(variational_params[param_index + length:param_index + length * 2])))
            noise_samples = np.array(rs.randn(num_samples, len(mean)))
            cur_samples = noise_samples * np.sqrt(cov) + mean
            samples.append(cur_samples)
        total_samples = np.hstack(samples)
        if len(param_info.keys()) == 1:  # if only one distribution, then we can calculate the entropy exactly
            lower_bound = gaussian_entropy(np.log(np.sqrt(cov))) + np.mean(logprob(total_samples, t))
        else:
            lower_bound = np.mean(-logvar(total_samples, variational_params) + logprob(total_samples, t))
        return -lower_bound  # make it negative so that we can do gradient descent

    gradient = grad(variational_objective)

    return variational_objective, gradient, unpack_params
