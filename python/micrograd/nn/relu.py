from .module import Module


class ReLU(Module):
    def forward(self, x):
        return x.relu()
