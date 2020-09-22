from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Optional, Union, TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

if TYPE_CHECKING:
    import torch.nn
    import torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter


MetricNames = Mapping[Callable[[np.ndarray, np.ndarray], float], str]


METRIC_NAMES: MetricNames = {
    accuracy_score: "acc",
    average_precision_score: "ap",
    f1_score: "f1",
    precision_score: "precision",
    recall_score: "recall",
    roc_auc_score: "roc-auc",
}


@dataclass(frozen=True)
class EpochResults:
    outputs: np.ndarray
    targets: np.ndarray
    average_loss: float


@dataclass(frozen=True)
class TrainTestDataloaders:
    train: torch.utils.data.DataLoader
    test: torch.utils.data.DataLoader


@dataclass(frozen=True)
class TrainTestEpochResults:
    train: EpochResults
    test: EpochResults


def freeze_layers(
    model: nn.Module, last_layer: Union[int, str], inplace: bool = True
) -> Optional[nn.Module]:
    if not inplace:
        model = copy.deepcopy(model)

    if isinstance(last_layer, int):
        for child in list(model.children())[: last_layer + 1]:
            for param in child.parameters():
                param.requires_grad = False
    elif isinstance(last_layer, str):
        for name, child in model.named_children():
            for param in child.parameters():
                param.requires_grad = False
            if name == last_layer:
                break
    else:
        raise TypeError(
            "Incorrect `last_layer` type. Integer or string type expected."
        )

    return model if not inplace else None


def train(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    objective: nn.modules.loss._Loss,
) -> EpochResults:
    if objective.reduction != "mean":
        return ValueError(
            "`objective` parameter accepts only losses with `reduction='mean'`"
        )

    model = model.to(device)
    model.train()

    outputs = []
    targets = []
    average_loss = 0.0
    for input_batch, target_batch in tqdm(dataloader):
        target_batch = target_batch.view(-1, 1).type_as(input_batch)
        targets.append(target_batch.cpu().detach().numpy())

        input_batch = input_batch.to(device)
        target_batch = target_batch.to(device)

        optimizer.zero_grad()

        output_batch = model(input_batch)
        outputs.append(output_batch.cpu().detach().numpy())

        loss = objective(output_batch, target_batch)
        average_loss += loss.item() * input_batch.shape[0]
        loss.backward()

        optimizer.step()

    average_loss /= len(dataloader)

    return EpochResults(
        outputs=np.concatenate(outputs, axis=0),
        targets=np.concatenate(targets, axis=0),
        average_loss=average_loss,
    )


def test(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    objective: nn.modules.loss._Loss,
) -> EpochResults:
    if objective.reduction != "mean":
        return ValueError(
            "`objective` parameter accepts only losses with `reduction='mean'`"
        )

    model = model.to(device)
    model.eval()

    outputs = []
    targets = []
    average_loss = 0.0
    with torch.no_grad():
        for input_batch, target_batch in tqdm(dataloader):
            target_batch = target_batch.view(-1, 1).type_as(input_batch)
            targets.append(target_batch.cpu().detach().numpy())

            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            output_batch = model(input_batch)
            outputs.append(output_batch.cpu().detach().numpy())

            average_loss += (
                objective(output_batch, target_batch).item()
                * input_batch.shape[0]
            )

    average_loss /= len(dataloader)

    return EpochResults(
        outputs=np.concatenate(outputs, axis=0),
        targets=np.concatenate(targets, axis=0),
        average_loss=average_loss,
    )


def run_experiment(
    model: nn.Module,
    dataloaders: TrainTestDataloaders,
    device: torch.device,
    optimizer: optim.Optimizer,
    objective: nn.modules.loss._Loss,
    epochs: int = 10,
    threshold: float = 0.5,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    artifacts_dir: Optional[Union[str, Path]] = None,
    writer: Optional[SummaryWriter] = None,
) -> None:
    for epoch in tqdm(range(epochs)):
        train_epoch_results = train(
            model=model,
            dataloader=dataloaders.train,
            device=device,
            optimizer=optimizer,
            objective=objective,
        )
        test_epoch_results = test(
            model=model,
            dataloader=dataloaders.test,
            device=device,
            objective=objective,
        )
        if scheduler is not None:
            scheduler.step()

        if artifacts_dir is not None:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            now = datetime.now()
            torch.save(
                checkpoint,
                Path(artifacts_dir).joinpath(
                    "checkpoint.{}.pth".format(
                        now.strftime("%d-%m-%Y%.%H_%M_%S")
                    )
                ),
            )

        if writer is not None:
            train_test_epoch_results = TrainTestEpochResults(
                train=train_epoch_results, test=test_epoch_results
            )
            _write_summary(
                writer=writer,
                epoch=epoch,
                train_test_epoch_results=train_test_epoch_results,
                threshold=threshold,
                scheduler=scheduler,
            )
    if writer is not None:
        writer.close()


def _write_summary(
    writer: torch.utils.tensorboard.SummaryWriter,
    epoch: int,
    train_test_epoch_results: TrainTestEpochResults,
    threshold: float,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    metric_names: Optional[MetricNames] = None,
) -> None:
    if metric_names is None:
        metric_names = METRIC_NAMES

    names = ["train", "test"]
    outputs = [
        train_test_epoch_results.train.outputs.flatten(),
        train_test_epoch_results.test.outputs.flatten(),
    ]
    targets = [
        train_test_epoch_results.train.targets.flatten(),
        train_test_epoch_results.test.targets.flatten(),
    ]
    predictions = [
        np.where(
            train_test_epoch_results.train.outputs > threshold, 1.0, 0.0
        ).flatten(),
        np.where(
            train_test_epoch_results.test.outputs > threshold, 1.0, 0.0
        ).flatten(),
    ]

    writer.add_scalar(
        "avg-loss-train", train_test_epoch_results.train.average_loss, epoch
    )
    writer.add_scalar(
        "avg-loss-test", train_test_epoch_results.test.average_loss, epoch
    )

    for score in [average_precision_score, roc_auc_score]:
        for name, output, target in zip(names, outputs, targets):
            writer.add_scalar(
                METRIC_NAMES[score] + "-" + name, score(target, output), epoch
            )

    for score in [accuracy_score, f1_score, precision_score, recall_score]:
        for name, prediction, target in zip(names, predictions, targets):
            writer.add_scalar(
                METRIC_NAMES[score] + "-" + name,
                score(target, prediction),
                epoch,
            )

    if scheduler is not None:
        writer.add_scalar("learning-rate", scheduler.get_last_lr()[0], epoch)
