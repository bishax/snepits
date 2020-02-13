import logging

import numpy as np
import theano.tensor as tt

import snepits

logger = logging.getLogger(__name__)


class PymcWrapper(tt.Op):
    """ Custom theano Op to wrap household model log likelihood calculation """

    # Inputs
    itypes = [tt.dvector]
    # Outputs
    otypes = [tt.dscalar]

    def __init__(self, model):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        model (snepits.models.models_meta.Population): model
        """

        self.model = model
        self.logpgrad = PymcWrapperGrad(self.model)
        self.call_counter = {'ll': 0, 'grad': 0, 'll_nocache': 0}

    def perform(self, node, inputs, outputs):
        """ Performs log likelihood calculation

        Args:
            node (theano.gof.graph.Apply): Graph node
            inputs (list): List of op inputs
            outputs (list): List of op outputs
        """

        theta, = inputs  # Model parameters
        self.call_counter['ll'] += 1

        if np.all(self.model.params == theta):
            outputs[0][0] = np.array(self.model.LL)
        else:
            self.model.param_update(theta)
            outputs[0][0] = np.array(self.model.calc_LL())
            self.call_counter['ll_nocache'] += 1

    def grad(self, inputs, g):
        """ Log grad operator

        Returns:
            list
                Vector-Jacobian product
        """
        # g[0] is a vector of parameter values?
        theta, = inputs  # our parameters
        self.call_counter['grad'] += 1
        return [g[0] * self.logpgrad(theta)]


class PymcWrapperGrad(tt.Op):
    """ Custom theano Op to wrap household model gradient calculation """

    # Inputs
    itypes = [tt.dvector]
    # Outputs
    otypes = [tt.dvector]

    def __init__(self, model, step=1e-5):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        model (snepits.models.models_meta.Population): model
        step (float, optional): Finite differences step length
        """

        # add inputs as class attributes
        self.model = model
        self.call_counter = {'grad': 0, 'll_grad': 0}
        self.step = step

    def perform(self, node, inputs, outputs):
        """ Perform gradient of log-likelihood calculation

        Args:
            node (theano.gof.graph.Apply): Graph node
            inputs (list): List of op inputs
            outputs (list): List of op outputs
        """
        theta, = inputs

        self.call_counter['grad'] += 1
        if not np.all(self.model.params == theta):
            msg = (
                f"Calculating gradients for {theta}"
                f"but current params are {self.model.params}"
            )
            # logger.warning(msg)
            # raise ValueError(msg)
            self.model.param_update(theta)
            self.model.calc_LL()
            self.call_counter['ll_grad'] += 1
        if self.model._unsolved:
            logger.warning('Model not solved')

        grad = np.zeros(self.model.dim)
        LL = self.model.LL
        for i in range(self.model.dim):
            params = self.model.params
            tmp = np.zeros_like(params)
            tmp[i] += self.step
            self.model.param_update(params + tmp)
            self.model.calc_LL()
            self.model.param_update(params)
            grad[i] = (self.model.LL - LL)/self.step
            if np.isinf(grad[i]) or np.isnan(grad[i]):
                grad[i] = - 1e9#np.inf
                # grad *= -np.inf
                # outputs[0][0] = grad
                # return
        self.model._unsolved = False

        if np.any(np.isnan(grad)):
            print('NAN', grad, 10*'-')
            breakpoint()

        if np.any(np.isinf(grad)):
            print('inf', grad, 10*'-')
            breakpoint()

        outputs[0][0] = grad  # self.model.calc_grad(self.step)
