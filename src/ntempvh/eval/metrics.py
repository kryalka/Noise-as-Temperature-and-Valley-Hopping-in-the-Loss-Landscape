from __future__ import annotations

from typing import Optional, Any

import torch
import torch.nn as nn


@torch.no_grad()
def eval_classification(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> dict[str, float]:
    """
    Подсчет итогового loss/accuracy на даталоадере для сравнения моделей и чекпоинтов (опционально на первых N батчах)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    loss_sum = 0.0
    correct = 0
    n = 0
    batches_seen = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss_sum += float(criterion(logits, y).item())

        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += int(x.size(0))
        batches_seen += 1

        if max_batches is not None and batches_seen >= max_batches:
            break

    if n <= 0:
        return {"loss": float("nan"), "acc": float("nan"), "n": 0.0, "batches": 0.0}

    return {
        "loss": loss_sum / n,
        "acc": correct / n,
        "n": float(n),
        "batches": float(batches_seen),
    }


def params_to_vector(model: nn.Module) -> torch.Tensor:
    """
    Упаковка параметров модели в один вектор для геометрических и интерполяционных экспериментов
    """
    return torch.nn.utils.parameters_to_vector(list(model.parameters()))


def vector_to_params(model: nn.Module, vec: torch.Tensor) -> None:
    torch.nn.utils.vector_to_parameters(vec, list(model.parameters()))