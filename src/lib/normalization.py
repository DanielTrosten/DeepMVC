from torch import nn


class L2Norm(nn.Module):
    def __init__(self, dim=-1):
        super(L2Norm, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        return nn.functional.normalize(inputs, dim=self.dim)


def get_normalizer(name, dim=-1):
    if name is None:
        norm = nn.Identity()
    elif name == "l2":
        norm = L2Norm(dim=dim)
    elif name == "softmax":
        norm = nn.Softmax(dim=dim)
    else:
        raise RuntimeError(f"Invalid normalizer: {name}")
    return norm
