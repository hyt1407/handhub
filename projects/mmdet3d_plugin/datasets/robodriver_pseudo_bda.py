from mmdet.datasets import DATASETS
import torch

from projects.mmdet3d_plugin.datasets.mtl_nuscenes_dataset_ego import MTLEgoNuScenesDataset


# from projects.mmdet3d_plugin.datasets import RobodriveDataset


@DATASETS.register_module()
class RobodriveDatasetPseudo(MTLEgoNuScenesDataset):

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            bs = data['img_inputs'][-1].size(0)
            data['img_inputs'] = [i for i in data['img_inputs']] + [torch.eye(3).expand(bs, -1, -1)]
            return data
