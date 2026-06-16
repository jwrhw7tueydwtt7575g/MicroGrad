from .module import Module


class Sequential(Module):
    """A list of layers applied in order."""

    def __init__(self, *layers):
        super().__init__()
        for layer in layers:
            self._children.append(layer)

    def add(self, layer):
        self._children.append(layer)

    def forward(self, x):
        for layer in self._children:
            x = layer(x)
        return x
