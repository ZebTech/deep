from nn import NN


class GNN(NN):

    def __init__(self, global_nn, glimplse_nn, glimpse_size=12, glimpse_loc=(.5, .5), downscale_factor=2):
        self.global_nn = global_nn
        self.glimpse_nn = glimplse_nn
        self.glimpse_size = glimpse_size
        self.glimpse_loc = glimpse_loc

    @property
    def params(self):
        return self.global_nn.params + self.glimpse_nn.params

    def glimpse(self, X, location, size):
        raise NotImplementedError
