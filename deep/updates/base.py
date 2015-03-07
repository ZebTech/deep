import theano
import theano.tensor as T
from abc import abstractmethod


class Update(object):

    @abstractmethod
    def __call__(self, cost, param, learning_rate):
        return

    def __repr__(self):
        return self.__class__.__name__


class GradientDescent(Update):

    def __call__(self, cost, param, learning_rate):
        return [(param, param - learning_rate * T.grad(cost, param))]


class Momentum(Update):

    def __init__(self, momentum=.9):
        self.momentum = momentum

    def __call__(self, cost, param, learning_rate):

        #: clean this up

        lr_scalers = dict()

        scaled_lr = learning_rate * lr_scalers.get(param, 1.)

        grad = T.grad(cost, param)
        vel = theano.shared(param.get_value() * 0.)

        inc = self.momentum * vel - scaled_lr * grad

        return [(param, param + inc), (vel, inc)]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.momentum)


class NesterovMomentum(Momentum):

    def __call__(self, cost, param, learning_rate):

        #: clean this up

        lr_scalers = dict()

        scaled_lr = learning_rate * lr_scalers.get(param, 1.)

        grad = T.grad(cost, param)
        vel = theano.shared(param.get_value() * 0.)

        inc = self.momentum * vel - scaled_lr * grad
        inc = self.momentum * inc - scaled_lr * grad

        return [(param, param + inc), (vel, inc)]


class AdaDelta(Update):

    def __init__(self, decay=0.95):
        self.decay=decay

    def __call__(self, cost, param, learning_rate):

        #: clean this up

        lr_scalers = dict()

        grad = T.grad(cost, param)

        mean_square_grad = theano.shared(param.get_value() * .0)
        mean_square_dx = theano.shared(param.get_value() * 0.)

        new_mean_squared_grad = (
            self.decay * mean_square_grad +
            (1 - self.decay) * T.sqr(grad)
        )

        epsilon = lr_scalers.get(param, 1.) * learning_rate
        rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
        rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
        delta_x_t = - rms_dx_tm1 / rms_grad_t * grad

        new_mean_square_dx = (
            self.decay * mean_square_dx +
            (1 - self.decay) * T.sqr(delta_x_t)
        )

        return [(mean_square_grad, new_mean_squared_grad),
                (mean_square_dx, new_mean_square_dx),
                (param, param + delta_x_t)]


class RMSProp(Update):

    def __init__(self, decay=0.1, max_scaling=1e5):
        self.decay = decay
        self.max_scaling = max_scaling
        self.epsilon = 1. / max_scaling

    def __call__(self, cost, param, learning_rate):

        #: clean this up

        lr_scalers = dict()

        grad = T.grad(cost, param)

        mean_square_grad = theano.shared(param.get_value() * 0.)

        new_mean_squared_grad = (
            self.decay * mean_square_grad +
            (1 - self.decay) * T.sqr(grad)
        )

        scaled_lr = lr_scalers.get(param, 1.) * learning_rate
        rms_grad_t = T.sqrt(new_mean_squared_grad)
        rms_grad_t = T.maximum(rms_grad_t, self.epsilon)
        delta_x_t = -scaled_lr * grad / rms_grad_t

        return [(mean_square_grad, new_mean_squared_grad),
                (param, param + delta_x_t)]

