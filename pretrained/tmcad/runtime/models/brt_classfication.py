# Inference export, 2026-11-13: unused Lightning training wrapper removed.
# The BRTClassification class is unchanged; upstream attribution is in NOTICE.md.
from .brt import BRT
from .encoders import _NonLinearClassifier
from torch import nn
import torch


class BRTClassification(nn.Module):
    def __init__(self, num_classes, d_model=64, masking_rate=None, max_face_length=100,num_control_pts=28):
        super().__init__()
        self.model = BRT(
            dmodel=d_model, hidden_dim=512, n_layers=2, n_heads=4, max_face_length=max_face_length, dropout=0.25,num_control_pts=num_control_pts
        )
        self.head = _NonLinearClassifier(d_model, num_classes)
        self.masking_rate = masking_rate
        self.max_face_length = max_face_length

    def forward(self, x):
        if self.masking_rate is not None:
            reserved_num = int(self.max_face_length * self.masking_rate)
            face_emb, mask = self.model(**x, reserved_num=reserved_num)
        else:
            face_emb, mask = self.model(**x)

        face_emb = face_emb.masked_fill(torch.logical_not(mask.unsqueeze(-1)), 0)
        count = mask.sum(dim=1)
        feature = face_emb.sum(dim=1) / count.unsqueeze(-1)

        return self.head(feature)

