from black_box_vi_base import black_box_variational_inference
from copy import copy
from optparse import OptionParser
import autograd.numpy as np
import importlib
import matplotlib.pyplot as plt

########################## SAMPLE RUN #############################
###################################################################
# python run_vi.py -p configs/bbvi_example.json -i 1000 -m models.bbviExampleModel -v
###################################################################

parser = OptionParser()

################### REQUIRED PARAMETERS ################### 
parser.add_option("-i", "--iters", dest="iters",
                  help="The number of iterations of variational inference to run", metavar="ITERS")
parser.add_option("-p", "--params", dest="params",
                  help="The location of the variational parameter descriptor file", metavar="PARAMS")
parser.add_option("-m", "--model",
                  help="The name of the model to be learned", metavar="MODEL")

################### OPTIONAL PARAMETERS ################### 
parser.add_option("-s", "--samples", default=10, dest="samples",
                  help="The number of variational samples to draw at every iteration", metavar="SAMPLES")
parser.add_option("-g", "--gradient", dest="estimate",
                  help="The gradient estimate to use: 'reparamaterization' (default), 'standard' or 'cv'",
                  default="reparameterization", metavar="OBJECTIVE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="Don't print status messages to stdout")
parser.add_option("-v", "--visualize", dest="visualize",
                  action="store_true", help="Whether to visualize the fit of the distribution (requires user implemented visualization)", metavar='VISUALIZE')

(options, args) = parser.parse_args()


def run_vi(model, step_size=1, num_samples=10, num_iters=1000, visualize=False, random_state=None):
    """
    Runs the variational inference gradient optimization procedure. When finished, returns the model
    and the parameters.
    """

    objective, gradient, unpack_params = \
        black_box_variational_inference(model.log_density, model._log_var, model.param_info, num_samples=num_samples, random_state=random_state)

    if visualize:
        fig = plt.figure(figsize=(8, 8), facecolor='white')
        ax = fig.add_subplot(111, frameon=False)
        plt.ion()
        plt.show(block=False)

    def callback(params, i, g):
        print("Iteration {} lower bound {}".format(i, -objective(params, i)))
        params = unpack_params(params)
        model.callback(params, i, g)
        if visualize:
            model.visualize(ax, params, i)

    print("Optimizing variational parameters...")
    x = copy(model.variational_params)
    b1 = 0.9
    b2 = 0.999
    eps = 10**-8
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):  # Currently uses ADAM
        g = gradient(x, i)
        if callback:
            callback(x, i, g)
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size * mhat / (np.sqrt(vhat) + eps)
        x = model.update(x)

    return model, x


if __name__ == "__main__":
    model_class = importlib.import_module(options.model)
    model = model_class.Model(param_location=options.params)

    run_vi(model,
           step_size=model.step_size,
           num_samples=int(options.samples),
           num_iters=int(options.iters),
           visualize=bool(options.visualize),
           random_state=None)
