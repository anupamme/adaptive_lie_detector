"""
Constructs the empirical payoff matrix by running all (attack, defense) pairs.
"""
import copy
import json
import os
import sys
import torch
import numpy as np
from itertools import product
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FLConfig, ExperimentConfig, GameConfig
from fl_core import get_federated_dataset, get_model, FederatedServer, FederatedClient
from attacks import get_attack
from attacks.attack_strategies import backdoor_pixel_fn


def evaluate_backdoor(model, test_dataset, trigger_size=4, target_class=0, device="cpu"):
    from torch.utils.data import DataLoader
    model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    triggered_correct = 0
    triggered_total = 0

    with torch.no_grad():
        for data, target in loader:
            mask = target != target_class
            if not mask.any():
                continue
            data, target = data[mask].to(device), target[mask].to(device)
            data[:, :, -trigger_size:, -trigger_size:] = 1.0
            output = model(data)
            pred = output.argmax(dim=1)
            triggered_correct += (pred == target_class).sum().item()
            triggered_total += data.size(0)

    return triggered_correct / max(triggered_total, 1)


def run_single_experiment(attack_name: str, defense_name: str,
                          fl_config: FLConfig, exp_config: ExperimentConfig) -> dict:
    torch.manual_seed(exp_config.seed)
    np.random.seed(exp_config.seed)

    device = exp_config.device if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    client_datasets, test_dataset, num_classes = get_federated_dataset(
        exp_config.dataset, fl_config.num_clients, exp_config.dirichlet_alpha, exp_config.seed
    )

    model = get_model(exp_config.model, num_classes)
    server = FederatedServer(model, device)

    num_adversarial = int(fl_config.num_clients * exp_config.adversarial_fraction)
    adversarial_ids = set(range(num_adversarial))

    attack = get_attack(attack_name)

    clients = []
    for i in range(fl_config.num_clients):
        dataset = client_datasets[i]
        if i in adversarial_ids:
            dataset = attack.poison_dataset(dataset)
        clients.append(FederatedClient(i, dataset, device))

    clients_per_round = getattr(fl_config, 'clients_per_round', max(10, fl_config.num_clients // 10))
    lr_decay = getattr(fl_config, 'lr_decay', 1.0)
    current_lr = fl_config.learning_rate

    for round_idx in tqdm(range(fl_config.num_rounds), desc=f"{attack_name} vs {defense_name}", leave=False):
        participant_ids = np.random.choice(fl_config.num_clients,
                                           size=min(clients_per_round, fl_config.num_clients),
                                           replace=False)
        updates = []
        for cid in participant_ids:
            update = clients[cid].train(
                server.global_model, fl_config.local_epochs,
                current_lr, fl_config.local_batch_size
            )
            if cid in adversarial_ids:
                update = attack.manipulate_update(update, server.global_model)
            updates.append(update)

        aggregated = server.aggregate(updates, method=defense_name)
        server.apply_update(aggregated)
        current_lr *= lr_decay

    eval_result = server.evaluate(test_dataset)

    asr = 0.0
    if attack_name in ("backdoor_pixel", "model_scaling", "dba"):
        asr = evaluate_backdoor(server.global_model, test_dataset, device=device)

    return {
        "accuracy": eval_result["accuracy"],
        "worst_class_accuracy": eval_result["worst_class_accuracy"],
        "attack_success_rate": asr,
        "attack": attack_name,
        "defense": defense_name,
    }


def run_full_payoff_matrix(fl_config: FLConfig = None, exp_config: ExperimentConfig = None,
                           game_config: GameConfig = None, output_dir: str = "results") -> dict:
    if fl_config is None:
        fl_config = FLConfig()
    if exp_config is None:
        exp_config = ExperimentConfig()
    if game_config is None:
        game_config = GameConfig()

    os.makedirs(output_dir, exist_ok=True)
    results = {}

    total = len(game_config.attacks) * len(game_config.defenses)
    print(f"Running {total} (attack, defense) combinations...")

    for attack_name, defense_name in tqdm(list(product(game_config.attacks, game_config.defenses)), desc="Payoff Matrix"):
        trial_results = []
        for trial in range(exp_config.num_trials):
            trial_config = ExperimentConfig(
                dataset=exp_config.dataset,
                model=exp_config.model,
                dirichlet_alpha=exp_config.dirichlet_alpha,
                adversarial_fraction=exp_config.adversarial_fraction,
                num_trials=1,
                device=exp_config.device,
                seed=exp_config.seed + trial,
            )
            result = run_single_experiment(attack_name, defense_name, fl_config, trial_config)
            trial_results.append(result)

        avg_result = {
            "accuracy": np.mean([r["accuracy"] for r in trial_results]),
            "worst_class_accuracy": np.mean([r["worst_class_accuracy"] for r in trial_results]),
            "attack_success_rate": np.mean([r["attack_success_rate"] for r in trial_results]),
            "accuracy_std": np.std([r["accuracy"] for r in trial_results]),
            "attack": attack_name,
            "defense": defense_name,
        }
        results[(attack_name, defense_name)] = avg_result

        with open(os.path.join(output_dir, "results_partial.json"), "w") as f:
            serializable = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
            json.dump(serializable, f, indent=2)

    with open(os.path.join(output_dir, "payoff_results.json"), "w") as f:
        serializable = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
        json.dump(serializable, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--adv_fraction", type=float, default=0.2)
    parser.add_argument("--num_rounds", type=int, default=200)
    parser.add_argument("--num_trials", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    fl_config = FLConfig(num_rounds=args.num_rounds)
    exp_config = ExperimentConfig(
        dataset=args.dataset,
        model=args.model,
        dirichlet_alpha=args.alpha,
        adversarial_fraction=args.adv_fraction,
        num_trials=args.num_trials,
    )
    game_config = GameConfig()

    run_full_payoff_matrix(fl_config, exp_config, game_config, args.output_dir)
