import torch.nn as nn

from models.clustering_module.kernel_width import get_kernel_width_module


class DDC(nn.Module):
    def __init__(self, cfg, input_size):
        super().__init__()

        hidden_layers = [nn.Linear(input_size[0], cfg.n_hidden), nn.ReLU()]
        if cfg.use_bn:
            hidden_layers.append(nn.BatchNorm1d(num_features=cfg.n_hidden))
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Sequential(nn.Linear(cfg.n_hidden, cfg.n_clusters), nn.Softmax(dim=1))
        
        self.kernel_width = get_kernel_width_module(cfg.kernel_width_config, input_size=[cfg.n_hidden])
        
    def forward(self, x):
        hidden = self.hidden(x)
        output = self.output(hidden)
        return hidden, output
