import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

# -----------------------------
# GNN Encoder
# -----------------------------
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, out_tokens=4, num_convs_hidden=10, hidden=128, edge_dim=1):
        super().__init__()
        self.out_tokens = out_tokens
        self.convs = nn.ModuleList()
        self.convs.append(gnn.GATv2Conv(in_dim, hidden, heads=1, edge_dim=edge_dim))
        for _ in range(num_convs_hidden):
            self.convs.append(gnn.GATv2Conv(hidden, hidden, heads=1, edge_dim=edge_dim))
        self.pool = gnn.TopKPooling(hidden, ratio=out_tokens / 100)
        self.final_proj = nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.convs[0](x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        for conv in self.convs[1:]:
            x_res = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.relu(x)
            x = x + x_res
        x, edge_index, edge_attr, batch, _, _ = self.pool(
            x, edge_index, edge_attr=edge_attr, batch=batch
        )
        x = self.final_proj(x)
        batch_size = batch.max().item() + 1 if batch is not None else 1
        out = []
        for i in range(batch_size):
            xi = x[batch == i]
            if xi.size(0) > self.out_tokens:
                xi = xi[:self.out_tokens]
            elif xi.size(0) < self.out_tokens:
                pad = torch.zeros(self.out_tokens - xi.size(0), xi.size(1), device=x.device)
                xi = torch.cat([xi, pad], dim=0)
            out.append(xi)
        out = torch.stack(out, dim=0)  # [batch_size, out_tokens, out_dim]
        return out

# -----------------------------
# Evaluator
# -----------------------------
class Evaluator(nn.Module):
    def __init__(self, input_dim, input_tokens):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        x = self.fc(x)  # <--- Remove F.relu here!
        x = x.mean(dim=1)
        return x

# -----------------------------
# Generator: vectorized 2D CoordNCA over tokens
# -----------------------------
class Generator(nn.Module):
    def __init__(self, input_dim, input_tokens, hidden=64, steps=4):
        super().__init__()
        self.input_dim = input_dim
        self.input_tokens = input_tokens
        self.hidden = hidden
        self.steps = steps

        self.mlp = nn.Sequential(
            nn.Linear(2*input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )
        # Learned 2D offsets: [token, dx, dy]
        self.offsets = nn.Parameter(torch.randn(input_tokens, 2) * 0.1)
        self.register_buffer("context", torch.zeros(1, input_tokens, input_dim))

    def forward(self, x):
        """
        x: [B, tokens, dim]
        """
        B, T, D = x.shape
        out = x + self.context

        # clamp offsets
        offsets_clamped = torch.clamp(self.offsets, -1.0, 1.0)
        # convert to integer target indices in [0, T-1] (simplified 2D->1D)
        base_idx = torch.arange(T, device=x.device).float()
        target_idx = torch.clamp((base_idx + offsets_clamped[:,0]).long(), 0, T-1)

        for _ in range(self.steps):
            # gather target states
            target_state = out[:, target_idx, :]
            mlp_input = torch.cat([out, target_state], dim=2)
            delta = 0.1 * torch.tanh(self.mlp(mlp_input))
            out = out + delta

        return out

    def update_context(self, feedback):
        # feedback: [B, tokens, input_dim]
        self.context = 0.9*self.context + 0.1*feedback.mean(dim=0, keepdim=True)

class Tester(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
# -----------------------------
# Task wrapper
# -----------------------------
class Task:
    def __init__(self, name):
        self.name = name

    def convert_to_graph(self, grid):
        colors = torch.unique(grid)
        num_nodes = len(colors)
        H, W = grid.shape
        x = torch.stack([(grid == c).float().flatten() for c in colors], dim=0)
        row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
        edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
        mask = row.flatten() != col.flatten()
        edge_index = edge_index[:, mask]
        edge_attr = []
        for i in range(edge_index.size(1)):
            s, t = edge_index[:, i]
            sim = torch.dot(x[s], x[t]) / (x[s].norm() * x[t].norm() + 1e-8)
            edge_attr.append(torch.tensor([sim]))
        edge_attr = torch.stack(edge_attr, dim=0)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # Toy ARC grid
    grid = torch.tensor([
        [1,0,0],
        [1,2,0],
        [0,2,2],
    ])

    task = Task("toy")
    data = task.convert_to_graph(grid)

    in_dim = data.x.size(1)
    encoder = GNNEncoder(in_dim=in_dim, out_dim=64, out_tokens=4, hidden=32)
    evaluator = Evaluator(input_dim=64, input_tokens=4)
    generator = Generator(input_dim=64, input_tokens=4)

    # Optimizers
    gen_opt = torch.optim.Adam(generator.parameters(), lr=1e-3)
    eval_opt = torch.optim.Adam(evaluator.parameters(), lr=1e-3)
    # Dummy reward function: reward is high if the mean of generator output is close to 1
    def reward_fn(gen_out):
        # Example: reward is higher the closer the mean is to 1
        return 

    NUM_GENS = 1000
    for g in range(NUM_GENS):
        # 1. Encode
        enc_out = encoder(data.x, data.edge_index, edge_attr=data.edge_attr)
        # 2. Generate
        gen_out = generator(enc_out)
        # 3. Get true reward
        true_reward = reward_fn(gen_out.detach())
        # 4. Evaluator predicts reward
        pred_reward = evaluator(gen_out.detach())  # <-- detach here!

        # --- Train Evaluator to predict true reward ---
        eval_loss = F.mse_loss(pred_reward, true_reward.expand_as(pred_reward))
        eval_opt.zero_grad()
        eval_loss.backward()
        eval_opt.step()

        # --- Train Generator to maximize predicted reward ---
        gen_out = generator(enc_out)  # re-generate for new grad
        pred_reward = evaluator(gen_out)
        gen_loss = -pred_reward.mean()
        gen_opt.zero_grad()
        print(gen_loss.item())
        gen_loss.backward()
        gen_opt.step()

        generator.update_context(gen_out.detach())  # Use generator output, not pred_reward

        if (g+1) % 5 == 0 or g == 0:
            print(f"Gen {g+1:02d} | True reward: {true_reward.item():.4f} | Pred: {pred_reward.mean().item():.4f}")

    print("Final generator context:", generator.context)
