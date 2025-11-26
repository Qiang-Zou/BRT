# brt segmenation
from .brt import BRT
from .encoders import _NonLinearClassifier
from torch import nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import torchmetrics


class BRTSegmentation(nn.Module):
    def __init__(self, num_classes, d_model=64, max_face_length=100, masking_rate=None):
        super().__init__()
        self.model = BRT(
            dmodel=d_model, hidden_dim=8 * d_model, n_layers=2, n_heads=8, dropout=0.25, max_face_length=max_face_length
        )
        self.head = _NonLinearClassifier(d_model, num_classes)
        self.masking_rate = masking_rate
        self.max_face_length = max_face_length

    def forward(self, x):
        if self.masking_rate is not None:
            reserved_num = int(self.max_face_length * self.masking_rate)
            face_emb, mask = self.model(**x, reserved_num=reserved_num)
            num_faces_per_solid = torch.sum(mask, dim=1).long()
        else:
            face_emb, mask = self.model(**x)
            num_faces_per_solid = x["num_faces_per_solid"]
        face_emb = self.model.BatchesIntoOneLine(face_emb, num_faces_per_solid)

        return self.head(face_emb)


class SegmentationPL(pl.LightningModule):
    """
    PyTorch Lightning module to train/test the classifier.
    """

    def __init__(self, num_classes=25, method="brt", masking_rate=None):
        """
        Args:
            num_classes (int): Number of per-solid classes in the dataset
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = BRTSegmentation(num_classes=num_classes, masking_rate=masking_rate)
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

        self.train_iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
        )
        self.val_iou = torchmetrics.JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
        )
        self.test_iou = torchmetrics.JaccardIndex(
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

        self.model.masking_rate = self.masking_rate
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.getBatchSize(batch))
        preds = F.softmax(logits, dim=-1)

        self.train_acc(preds, labels)
        self.train_iou(preds, labels)

        return loss

    def on_train_epoch_end(self):
        self.log("train_iou", self.train_iou.compute())
        self.log("train_acc", self.train_acc.compute())

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

        self.model.masking_rate = None
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.getBatchSize(batch))
        preds = F.softmax(logits, dim=-1)
        self.val_acc(preds, labels)
        self.val_iou(preds, labels)
        return loss

    def on_validation_epoch_end(self):
        self.log("val_iou", self.val_iou.compute())
        self.log("val_acc", self.val_acc.compute())

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
        self.model.masking_rate = None
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, labels, reduction="mean")
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=self.getBatchSize(batch))
        preds = F.softmax(logits, dim=-1)
        self.test_iou(preds, labels)
        self.test_acc(preds, labels)
        return loss

    def on_test_epoch_end(self):
        self.log("test_iou", self.test_iou.compute())
        self.log("test_acc", self.test_acc.compute())

    def getBatchSize(self, batch):
        return batch["label"].shape[0]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
