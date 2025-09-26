# hybrid_pop_gen_stateless.py
# No optimizer baggage, just raw param -= lr * grad
import copy, math, random
import torch
import torch.nn as nn
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
random.seed(42)

POP_SIZE = 12
NOISE_DIM = 16
OUT_DIM = 32
INNER_BATCH = 32
INNER_UPDATES = 4
GENS = 200
LR = 1e-2                 # slightly bigger since no momentum
MUTATION_STD = 0.02
ELITE_FRACTION = 0.25
RANDOM_INJECTION = 0.1

# ---------- Generator ----------
class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, out_dim=OUT_DIM, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, z): return self.net(z)

# ---------- Utilities ----------
def clone_model(m: nn.Module): return copy.deepcopy(m)

def get_flat_params(m: nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(m.parameters()).detach().clone()

def set_flat_params(m: nn.Module, vec: torch.Tensor):
    torch.nn.utils.vector_to_parameters(vec, m.parameters())

def mutate_params(params_vec: torch.Tensor, std=MUTATION_STD) -> torch.Tensor:
    return params_vec + torch.randn_like(params_vec) * std

def crossover_params(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mask = torch.rand_like(a) < 0.5
    out = a.clone(); out[mask] = b[mask]
    return out

# ---------- Feedback ----------
TARGET = torch.randn(OUT_DIM).to(DEVICE) * 2.0
def feedback_fn(outputs: torch.Tensor) -> torch.Tensor:
    d = torch.norm(outputs - TARGET.unsqueeze(0), dim=1)
    return -d  # higher = closer to target

# ---------- Manual update ----------
def stateless_update(model: nn.Module, loss: torch.Tensor, lr=LR):
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad.zero_()

# ---------- Init ----------
population: List[Generator] = [Generator().to(DEVICE) for _ in range(POP_SIZE)]

# ---------- Loop ----------
for gen in range(1, GENS+1):
    gen_rewards = []
    for model in population:
        model.train()
        for _ in range(INNER_UPDATES):
            z = torch.randn(INNER_BATCH, NOISE_DIM, device=DEVICE)
            out = model(z)
            rewards = feedback_fn(out)
            loss = -rewards.mean()
            stateless_update(model, loss, lr=LR)

        with torch.no_grad():
            z_eval = torch.randn(256, NOISE_DIM, device=DEVICE)
            out_eval = model(z_eval)
            avg_reward = feedback_fn(out_eval).mean().item()
            gen_rewards.append(avg_reward)

    ranked = sorted(enumerate(gen_rewards), key=lambda x: x[1], reverse=True)
    elites_n = max(1, int(math.ceil(ELITE_FRACTION * POP_SIZE)))
    elite_indices = [idx for idx,_ in ranked[:elites_n]]

    new_population: List[Generator] = []
    flat_params = [get_flat_params(m).cpu() for m in population]

    # 1) copy elites
    for idx in elite_indices:
        new_population.append(clone_model(population[idx]).to(DEVICE))

    # 2) mutate / crossover / inject
    while len(new_population) < POP_SIZE:
        if random.random() < RANDOM_INJECTION:
            new_population.append(Generator().to(DEVICE))
            continue
        pa, pb = random.choice(elite_indices), random.choice(range(POP_SIZE))
        child_vec = crossover_params(flat_params[pa], flat_params[pb])
        child_vec = mutate_params(child_vec)
        child = Generator().to(DEVICE)
        set_flat_params(child, child_vec.to(DEVICE))
        new_population.append(child)

    population = new_population

    if gen % 10 == 0 or gen == 1:
        print(f"[Gen {gen:03d}] best={max(gen_rewards):.4f}, avg={sum(gen_rewards)/len(gen_rewards):.4f}")

# Final best
with torch.no_grad():
    final_scores = []
    for model in population:
        z = torch.randn(1024, NOISE_DIM, device=DEVICE)
        final_scores.append(feedback_fn(model(z)).mean().item())
    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    print("=== Done ===")
    print("Best reward:", final_scores[best_idx])

