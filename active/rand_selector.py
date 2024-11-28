import torch
import numpy as np
from typing import List, Dict, Union, Optional
from copy import deepcopy
import random


class RandSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed

    
    def nbvs(self, gaussian, scene, num_views, *args, **kwargs) -> List[int]:
        candidate_views = deepcopy(list(scene.get_candidate_set()))
        random.Random(self.seed).shuffle(candidate_views)

        return candidate_views[:num_views]
    
    def forward(self, x):
        return x


class FarthestPointSelector(torch.nn.Module):

    def __init__(self, args) -> None:
        super().__init__()
        self.seed = args.seed

    def nbvs(self, gaussian, scene, num_views, *args, **kwargs) -> List[int]:
        candidate_views = list(deepcopy(scene.get_candidate_set()))
        candidate_cameras = scene.getCandidateCameras().copy()
        train_views = scene.getTrainCameras().copy()

        acq_scores = torch.zeros(len(candidate_cameras))
        for i,view in enumerate(candidate_cameras):
            for train_view in train_views:
                acq_scores[i]+=torch.norm(view.camera_center-train_view.camera_center).item()

        _, indices = torch.sort(acq_scores, descending=True)
        selected_idxs = [candidate_views[i] for i in indices[:num_views].tolist()]
        print(f"acq_scores_max: {[acq_scores[i] for i in indices[:num_views + 3].tolist()]}, selected_idxs: {[candidate_views[i] for i in indices[:num_views + 3].tolist()]}")

        return selected_idxs

    def forward(self, x):
        return x

