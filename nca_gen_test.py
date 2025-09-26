# hybrid_2d_nca_stable.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy, random, math
from typing import List

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)
random.seed(42)

# ---------- Settings ----------
POP_SIZE = 8
H, W = 16, 16           # grid size
C = 8                   # channels per cell
INNER_BATCH = 1
INNER_UPDATES = 4
GENS = 100
LR = 1e-3               # smaller LR for stability
MUTATION_STD = 0.02
ELITE_FRACTION = 0.25
RANDOM_INJECTION = 0.1
STEPS = 8               # NCA iterations per grid
DELTA_SCALE = 0.1       # scale delta to prevent explosions
OFFSET_MAX = 1.0        # clamp learned offsets

# ---------- Neural Cellular Automaton ----------
class CoordNCA(nn.Module):
    def __init__(self, channels=C, hidden=32):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.mlp = nn.Sequential(
            nn.Linear(2*channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels)
        )
        self.offsets = nn.Parameter(torch.randn(H, W, 2) * 0.1)  # initial offsets

    def forward(self, grid):
        B = grid.shape[0]
        new_grid = grid.clone()
        offsets_clamped = torch.clamp(self.offsets, -OFFSET_MAX, OFFSET_MAX)
        for i in range(H):
            for j in range(W):
                dx, dy = offsets_clamped[i,j]
                x = torch.clamp(j + dx, 0, W-1-1e-3)
                y = torch.clamp(i + dy, 0, H-1-1e-3)
                # normalize to [-1,1] for grid_sample
                gx = (x/(W-1))*2 - 1
                gy = (y/(H-1))*2 - 1
                sample_coords = torch.tensor([[[[gx, gy]]]], device=DEVICE).repeat(B,1,1,1)
                sampled = F.grid_sample(grid, sample_coords, mode='bilinear', align_corners=True)
                sampled = sampled.view(B, C)
                cell_state = grid[:,:,i,j].view(B,C)
                mlp_input = torch.cat([cell_state, sampled], dim=1)
                delta = DELTA_SCALE * torch.tanh(self.mlp(mlp_input))  # limit step size
                new_grid[:,:,i,j] = cell_state + delta
        return new_grid

# ---------- Feedback ----------
TARGET_GRID = torch.randn(C,H,W,device=DEVICE)*2.0
def feedback_fn(grid):
    return -torch.norm(grid - TARGET_GRID.unsqueeze(0), dim=(1,2,3))

# ---------- Stateless update with gradient clipping ----------
def stateless_update(model, loss, lr=LR, max_grad_norm=1.0):
    loss.backward()
    with torch.no_grad():
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
                p.grad.zero_()

# ---------- Population ----------
population: List[CoordNCA] = [CoordNCA().to(DEVICE) for _ in range(POP_SIZE)]

# ---------- Main loop ----------
for gen in range(1, GENS+1):
    gen_rewards = []
    for model in population:
        model.train()
        for _ in range(INNER_UPDATES):
            z = torch.randn(INNER_BATCH, C, H, W, device=DEVICE)
            out = z
            for _ in range(STEPS):
                out = model(out)
            rewards = feedback_fn(out)
            loss = -rewards.mean()
            stateless_update(model, loss)

        # evaluation
        with torch.no_grad():
            z_eval = torch.randn(1, C, H, W, device=DEVICE)
            out_eval = z_eval
            for _ in range(STEPS):
                out_eval = model(out_eval)
            avg_reward = feedback_fn(out_eval).mean().item()
            gen_rewards.append(avg_reward)

    # ES selection + mutation
    ranked = sorted(enumerate(gen_rewards), key=lambda x: x[1], reverse=True)
    elites_n = max(1,int(math.ceil(ELITE_FRACTION*POP_SIZE)))
    elite_indices = [idx for idx,_ in ranked[:elites_n]]

    new_population: List[CoordNCA] = []
    flat_params = [torch.nn.utils.parameters_to_vector(m.parameters()).cpu() for m in population]

    # copy elites
    for idx in elite_indices:
        new_population.append(copy.deepcopy(population[idx]).to(DEVICE))

    # fill rest
    while len(new_population) < POP_SIZE:
        if random.random() < RANDOM_INJECTION:
            new_population.append(CoordNCA().to(DEVICE))
            continue
        pa, pb = random.choice(elite_indices), random.choice(range(POP_SIZE))
        child_vec = flat_params[pa]
        child_vec = (child_vec + flat_params[pb])/2
        child_vec = child_vec + torch.randn_like(child_vec)*MUTATION_STD
        child = CoordNCA().to(DEVICE)
        torch.nn.utils.vector_to_parameters(child_vec.to(DEVICE), child.parameters())
        new_population.append(child)

    population = new_population

    if gen % 10 == 0 or gen == 1:
        print(f"[Gen {gen:03d}] best={gen_rewards[elite_indices[0]]:.4f}, avg={sum(gen_rewards)/len(gen_rewards):.4f}")

# final evaluation
with torch.no_grad():
    final_scores = []
    for model in population:
        z = torch.randn(1, C, H, W, device=DEVICE)
        out = z
        for _ in range(STEPS):
            out = model(out)
        final_scores.append(feedback_fn(out).mean().item())
    best_idx = max(range(len(final_scores)), key=lambda i: final_scores[i])
    print("=== Done ===")
    print("Best reward:", final_scores[best_idx])

