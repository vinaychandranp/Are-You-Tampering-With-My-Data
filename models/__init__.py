# Keep the list of models implemented up-2-date
from .CNN_basic import CNN_basic
from .FC_medium import FC_medium
from .FC_simple import FC_simple
from .TNet import TNet
from ._AlexNet import _AlexNet, alexnet
from ._ResNet import resnet18, resnet34, resnet50, resnet101, resnet152
from .observer import Observer

"""
Formula to compute the output size of a conv. layer

new_size =  (width - filter + 2padding) / stride + 1
"""
