from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset






@dataclass
class DummyLoaders:
    train: DataLoader
    val: DataLoader
    bn: DataLoader



class TinyNet(nn.Module):
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))


def _fmt_token(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    return f"{value:g}"



def make_run_name(
    *,
    dataset: str = "cifar10",
    model: str = "resnet18",
    seed: int,
    lr: float,
    bs: int,
    optimizer: str = "sgd",
    weight_decay: float | int | str = "0",
    momentum: float | int | str = "0",
    scheduler: str = "none",
    suffix: str = "deadbeef",
) -> str:
    
    return (
        f"{dataset}_{model}_seed{seed}"
        f"__opt{optimizer}_lr{_fmt_token(lr)}_bs{bs}"
        f"_wd{_fmt_token(weight_decay)}_mom{_fmt_token(momentum)}"
        f"_sch{scheduler}__{suffix}"
    )



def make_random_loaders(
    *,
    n_train: int = 32,
    n_val: int = 32,
    batch_size: int = 16,
    num_classes: int = 10,
    seed: int = 0,
    train_shuffle: bool = False,
) -> DummyLoaders:
    
    g = torch.Generator().manual_seed(seed)

    x_train = torch.randn((n_train, 3, 32, 32), generator=g)
    y_train = torch.randint(0, num_classes, (n_train,), generator=g)
    x_val = torch.randn((n_val, 3, 32, 32), generator=g)
    y_val = torch.randint(0, num_classes, (n_val,), generator=g)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=train_shuffle, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    bn_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)



def make_easy_loaders(
    *,
    train_batch_size: int,
    val_batch_size: int | None = None,
    bn_batch_size: int | None = None,
    n_train: int = 24,
    n_val: int = 12,
    num_classes: int = 10,
    shuffle_train: bool = False,
    shuffle_seed: int = 0,
) -> DummyLoaders:
    
    val_batch = int(train_batch_size if val_batch_size is None else val_batch_size)
    bn_batch = int(val_batch if bn_batch_size is None else bn_batch_size)

    y_train = torch.arange(n_train, dtype=torch.long) % num_classes
    y_val = torch.arange(n_val, dtype=torch.long) % num_classes

    x_train = torch.zeros((n_train, 3, 32, 32), dtype=torch.float32)
    x_val = torch.zeros((n_val, 3, 32, 32), dtype=torch.float32)
    x_train[torch.arange(n_train), 0, 0, y_train] = 1.0
    x_val[torch.arange(n_val), 0, 0, y_val] = 1.0

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    shuffle_gen = torch.Generator().manual_seed(int(shuffle_seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_batch_size),
        shuffle=shuffle_train,
        generator=shuffle_gen if shuffle_train else None,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch,
        shuffle=False,
        num_workers=0,
    )
    bn_loader = DataLoader(
        train_ds,
        batch_size=bn_batch,
        shuffle=False,
        num_workers=0,
    )
    return DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)



def make_easy_model(num_classes: int = 10) -> TinyNet:
    
    model = TinyNet(num_classes=num_classes)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        for cls in range(num_classes):
            model.fc.weight[cls, cls] = 8.0
    return model



def write_fake_ckpt(
    *,
    root: Path,
    run_name: str,
    epoch: int,
    payload: dict,
    runs_subdir: str = "outputs/runs_lr_bs_grid",
) -> Path:
    
    ckpt_dir = root / runs_subdir / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path
