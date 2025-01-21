import torch.nn as nn
from collections import OrderedDict

class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs, use_batch_norm=False):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_outputs: output dimension of the MLP
          use_batch_norm: If True, add a Batch-Normalization layer in between
                          each Linear and ELU layer.

        """
        super(MLP, self).__init__()

        in_features = n_inputs
        self.layers = []
        for i, hidden_size in enumerate(n_hidden):
            self.layers += [nn.Linear(in_features, hidden_size)]
            if use_batch_norm:
                self.layers += [nn.BatchNorm1d(hidden_size)]
            self.layers += [nn.SELU(alpha=1, inplace=True)]
            in_features = hidden_size
        n_last_layer_inp = n_inputs if len(n_hidden) == 0 else n_hidden[-1]
        self.layers.append(nn.Linear(n_last_layer_inp, n_outputs))
        self.layers = nn.Sequential(*self.layers)

        """ 
        for l_idx, m in enumerate(self.layers):
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
                if l_idx == 0:
                    nn.init.normal_(m.weight, std=1.0/m.weight.shape[1]**0.5)
                else:
                    nn.init.kaiming_normal_(m.weight)
        """

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        return self.layers(x)