from optparse import OptionParser
import importlib

parser = OptionParser()

parser.add_option("-i", "--iters", dest="iters",
                  help="The number of iterations of variational inference to run", metavar="PARAMS")
parser.add_option("-s", "--samples", dest="samples",
                  help="The number of variational samples to draw at every iteration", metavar="SAMPLES")
parser.add_option("-o", "--gradient", dest="estimate",
                  help="The gradient estimate to use: 'reparamaterization' (default), 'standard' or 'cv'", metavar="OBJECTIVE")
parser.add_option("-p", "--params", dest="params",
                  help="The location of the variational parameter descriptor file", metavar="PARAMS")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
parser.add_option("-m", "--model",
                  help="The name of the model to be learned", metavar="MODEL")

(options, args) = parser.parse_args()

module = importlib.import_module(options.model)

