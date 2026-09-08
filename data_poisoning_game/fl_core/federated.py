import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional


class FederatedClient:
    def __init__(self, client_id: int, dataset, device: str = "cpu"):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device

    def train(self, global_model: nn.Module, epochs: int, lr: float, batch_size: int) -> Dict[str, torch.Tensor]:
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

        update = {}
        for name, param in model.named_parameters():
            update[name] = param.data - global_model.state_dict()[name].to(self.device)
        return update


class FederatedServer:
    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.global_model = model.to(device)
        self.device = device

    def aggregate(self, updates: List[Dict[str, torch.Tensor]], method: str = "fedavg",
                  **kwargs) -> Dict[str, torch.Tensor]:
        if method == "fedavg":
            return self._fedavg(updates)
        elif method == "krum":
            return self._krum(updates, multi=False)
        elif method == "multi_krum":
            return self._krum(updates, multi=True, k=kwargs.get("k", 5))
        elif method == "trimmed_mean":
            return self._trimmed_mean(updates, beta=kwargs.get("beta", 0.2))
        elif method == "coord_median":
            return self._coordinate_median(updates)
        elif method == "norm_clip":
            return self._norm_clip(updates, tau=kwargs.get("tau", 5.0))
        elif method == "rfa":
            return self._rfa(updates, max_iter=kwargs.get("max_iter", 50))
        else:
            raise ValueError(f"Unknown aggregation: {method}")

    def apply_update(self, aggregated_update: Dict[str, torch.Tensor]):
        state = self.global_model.state_dict()
        for name in aggregated_update:
            state[name] = state[name] + aggregated_update[name].to(self.device)
        self.global_model.load_state_dict(state)

    def evaluate(self, test_dataset, batch_size: int = 256) -> dict:
        self.global_model.eval()
        loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
                for c in target.unique():
                    c_val = c.item()
                    mask = target == c
                    per_class_correct[c_val] = per_class_correct.get(c_val, 0) + pred[mask].eq(target[mask]).sum().item()
                    per_class_total[c_val] = per_class_total.get(c_val, 0) + mask.sum().item()

        accuracy = correct / total
        per_class_acc = {c: per_class_correct[c] / per_class_total[c] for c in per_class_total}
        worst_class_acc = min(per_class_acc.values()) if per_class_acc else 0.0

        return {"accuracy": accuracy, "worst_class_accuracy": worst_class_acc, "per_class_accuracy": per_class_acc}

    def _fedavg(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        avg = {}
        for name in updates[0]:
            stacked = torch.stack([u[name].float() for u in updates])
            avg[name] = stacked.mean(dim=0)
        return avg

    def _krum(self, updates: List[Dict[str, torch.Tensor]], multi: bool = False, k: int = 5) -> Dict[str, torch.Tensor]:
        n = len(updates)
        flat = [torch.cat([u[name].flatten() for name in updates[0]]) for u in updates]
        flat = torch.stack(flat).cpu().float()

        distances = torch.cdist(flat.unsqueeze(0), flat.unsqueeze(0)).squeeze(0)
        f = max(1, n // 5)
        scores = []
        for i in range(n):
            sorted_dists, _ = distances[i].sort()
            scores.append(sorted_dists[1:n - f].sum().item())

        if multi:
            selected_indices = sorted(range(n), key=lambda i: scores[i])[:k]
        else:
            selected_indices = [min(range(n), key=lambda i: scores[i])]

        selected_updates = [updates[i] for i in selected_indices]
        return self._fedavg(selected_updates)

    def _trimmed_mean(self, updates: List[Dict[str, torch.Tensor]], beta: float = 0.2) -> Dict[str, torch.Tensor]:
        result = {}
        n = len(updates)
        trim_count = int(n * beta)
        for name in updates[0]:
            shape = updates[0][name].shape
            device = updates[0][name].device
            stacked = torch.stack([u[name].flatten().cpu().float() for u in updates])
            sorted_vals, _ = stacked.sort(dim=0)
            trimmed = sorted_vals[trim_count:n - trim_count]
            result[name] = trimmed.mean(dim=0).reshape(shape).to(device)
        return result

    def _coordinate_median(self, updates: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        result = {}
        for name in updates[0]:
            shape = updates[0][name].shape
            device = updates[0][name].device
            stacked = torch.stack([u[name].flatten().cpu().float() for u in updates])
            result[name] = stacked.median(dim=0).values.reshape(shape).to(device)
        return result

    def _norm_clip(self, updates: List[Dict[str, torch.Tensor]], tau: float = 5.0) -> Dict[str, torch.Tensor]:
        clipped = []
        for u in updates:
            flat = torch.cat([u[name].flatten() for name in u])
            norm = flat.norm()
            scale = min(1.0, tau / (norm.item() + 1e-8))
            clipped.append({name: u[name] * scale for name in u})
        return self._fedavg(clipped)

    def _rfa(self, updates: List[Dict[str, torch.Tensor]], max_iter: int = 50, tol: float = 1e-6) -> Dict[str, torch.Tensor]:
        device = updates[0][next(iter(updates[0]))].device
        flat_updates = [torch.cat([u[name].flatten() for name in updates[0]]).cpu().float() for u in updates]
        stacked = torch.stack(flat_updates)

        estimate = stacked.mean(dim=0)
        for _ in range(max_iter):
            diffs = stacked - estimate.unsqueeze(0)
            norms = diffs.norm(dim=1, keepdim=True).clamp(min=1e-8)
            weights = 1.0 / norms
            weights = weights / weights.sum()
            new_estimate = (weights * stacked).sum(dim=0)
            if (new_estimate - estimate).norm() < tol:
                break
            estimate = new_estimate

        result = {}
        offset = 0
        for name in updates[0]:
            numel = updates[0][name].numel()
            result[name] = estimate[offset:offset + numel].reshape(updates[0][name].shape).to(device)
            offset += numel
        return result
