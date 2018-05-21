import torch.nn as nn

class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class Observer(nn.Module):
    """
    Simple single linear layer

    Attributes
    ----------
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, output_channels=10, input_channels=288, **kwargs):
        """
        Creates a single layer observer

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(Observer, self).__init__()

        self.expected_input_size = 288  # Last layer of CNN_Basic
        self.output_channels = output_channels

        # Classification layer
        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, self.output_channels)
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        x = self.fc(x)
        return x
