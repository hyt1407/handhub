from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule

from projects.mmdet3d_plugin.models.utils.norm import GRN, LayerNorm2d, build_norm_layer


class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    def __init__(self, init_cfg=None):
        super(BaseBackbone, self).__init__(init_cfg)

    @abstractmethod
    def forward(self, x):
        """Forward computation.

        Args:
            x (tensor | tuple[tensor]): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        """
        pass

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)


__all__ = ['GRN', 'BaseBackbone', 'LayerNorm2d', 'build_norm_layer']
