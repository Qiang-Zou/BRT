import argparse
import pathlib
import time

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch import seed_everything
import datasets.brt_dataset
from models.brt_classfication import ClassificationPL as BRTClassification
import datasets

import torch

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

parser = argparse.ArgumentParser(
    "Parameterized Surface Reconstruction from Bezier Patches")
parser.add_argument(
    "traintest", choices=("train", "test"), help="Whether to train or test"
)

parser.add_argument(
    "--num_classes", type=int, help="Number of classes"
)

parser.add_argument(
    "--method", choices=("brt"), default='brt',help="Specific Method"
)
parser.add_argument(
    "--precision", choices=("medium", "high", "highest"), default='medium',help="Pytorch Float Precision"
)
parser.add_argument(
    "--gpu", type=int,default=0, help="choose gpu"
)

parser.add_argument("--dataset_dir", type=str, help="Directory to datasets")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows, any other value leads to poor performance",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)

# parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

experiment_name = args.experiment_name

torch.set_float32_matmul_precision(args.precision)
results_path = (
    pathlib.Path(__file__).parent.joinpath(
        "results").joinpath(experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

# Define a path to save the results based date and time. E.g.
# results/args.experiment_name/0430/123103
month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_last=True,
)

trainer = Trainer(callbacks=[checkpoint_callback], logger=[TensorBoardLogger(
    str(results_path), name=month_day, version=hour_min_second,
)],devices=[args.gpu],accelerator='gpu')

if args.method == "brt":
    ClassificationModel = BRTClassification
else:
    raise NotImplementedError

if args.method == "brt":
    Dataset = datasets.brt_dataset.BRTDataset_cls_online
else:
    raise NotImplementedError

if args.traintest == "train":
    seed_everything(workers=True)
    print(
        f"""
-----------------------------------------------------------------------------------
BRT Classification
-----------------------------------------------------------------------------------
Logs written to results/{experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    if args.checkpoint is not None:
        model=ClassificationModel.load_from_checkpoint(args.checkpoint)
    else:
        model_hparams = {'method': args.method,'num_classes':args.num_classes,"masking_rate":None}
        if model_hparams is not None:
            model = ClassificationModel(**model_hparams)
        else:
            model = ClassificationModel()
    train_data = Dataset(root_dir=args.dataset_dir, split="train",masking_rate=None,masking_rate_v2=None)
    val_data = Dataset(root_dir=args.dataset_dir, split="val",masking_rate=None,masking_rate_v2=None)
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    trainer.fit(model, train_loader, val_loader)
else:
    # Test
    assert (
        args.checkpoint is not None
    ), "Expected the --checkpoint argument to be provided"
    test_data = Dataset(root_dir=args.dataset_dir, split="test")
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = ClassificationModel.load_from_checkpoint(args.checkpoint)
    results = trainer.test(model=model, dataloaders=[
                           test_loader], verbose=True)
    print(
        f"Classfication Loss on test set: {results[0]['test_loss']}"
    )
