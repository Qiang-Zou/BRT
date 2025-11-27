from .brt import BRT
from .encoders import _NonLinearClassifier
from torch import nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics


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


class ClassificationPL(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """

    def __init__(self, num_classes=15, method="brt", masking_rate=None,num_control_pts=28):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = BRTClassification(num_classes=num_classes, masking_rate=masking_rate,num_control_pts=num_control_pts)
        self.masking_rate = masking_rate
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=num_classes,
        )

    def forward(self, batched_graph):
        logits = self.model(batched_graph)
        return logits

    def training_step(self, batch, batch_idx):
        inputs = dict()
        cpu_inputs = self.model.model.topo_layer.cpu_input()
        for key, value in batch.items():
            if key in ["filename"]:
                continue
            if key in cpu_inputs:
                inputs[key] = value.cpu()
            else:
                inputs[key] = value.to(self.device)

        labels = batch["label"]

        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.getBatchSize(batch))
        preds = F.softmax(logits, dim=-1)
        self.log(
            "train_acc",
            self.train_acc(preds, labels),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.getBatchSize(batch),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = dict()
        cpu_inputs = self.model.model.topo_layer.cpu_input()
        for key, value in batch.items():
            if key in ["filename"]:
                continue
            if key in cpu_inputs:
                inputs[key] = value.cpu()
            else:
                inputs[key] = value.to(self.device)

        labels = batch["label"]

        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.getBatchSize(batch))
        preds = F.softmax(logits, dim=-1)
        self.log(
            "val_acc",
            self.val_acc(preds, labels),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.getBatchSize(batch),
        )
        return loss

    def test_step(self, batch, batch_idx):
        inputs = dict()
        cpu_inputs = self.model.model.topo_layer.cpu_input()
        for key, value in batch.items():
            if key in ["filename"]:
                continue
            if key in cpu_inputs:
                inputs[key] = value.cpu()
            else:
                inputs[key] = value.to(self.device)

        labels = batch["label"]

        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.getBatchSize(batch))
        preds = F.softmax(logits, dim=-1)
        self.log(
            "test_acc",
            self.test_acc(preds, labels),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.getBatchSize(batch),
        )
        return loss

    def getBatchSize(self, batch):
        return batch["label"].shape[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
