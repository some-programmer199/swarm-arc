import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data

# -----------------------------
# GNN Encoder
# -----------------------------
class GNNEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128, out_tokens=4, num_convs_hidden=2, hidden=32, edge_dim=1):
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
        x = self.fc(x)
        x = x.mean(dim=1)
        return x

# -----------------------------
# Generator with fixed soft indexing
# -----------------------------
class Generator(nn.Module):
    def __init__(self, input_dim, input_tokens, hidden=64, steps=8):
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
        self.offsets = nn.Parameter(torch.randn(input_tokens) * 0.1)
        self.register_buffer("context", torch.zeros(1, input_tokens, input_dim))

    def forward(self, x):
        B, T, D = x.shape
        out = x + self.context

        base_idx = torch.arange(T, device=x.device).float()  # [T]
        target_idx = torch.clamp(base_idx[None, :] + self.offsets[None, :], 0, T-1)  # [1, T]

        for _ in range(self.steps):
            idx0 = target_idx.long()               # [1, T]
            idx1 = torch.clamp(idx0 + 1, max=T-1)
            weight = target_idx - idx0.float()     # [1, T]

            # Expand to batch
            idx0_exp = idx0.expand(B, -1)          # [B, T]
            idx1_exp = idx1.expand(B, -1)
            weight_exp = weight.expand(B, -1).unsqueeze(2)  # [B, T, 1]

            # Gather target states
            target0 = torch.gather(out, 1, idx0_exp.unsqueeze(2).expand(-1, -1, D))  # [B, T, D]
            target1 = torch.gather(out, 1, idx1_exp.unsqueeze(2).expand(-1, -1, D))  # [B, T, D]
            target_state = (1 - weight_exp) * target0 + weight_exp * target1          # [B, T, D]

            mlp_input = torch.cat([out, target_state], dim=2)  # [B, T, 2*D]
            delta = 0.3 * torch.tanh(self.mlp(mlp_input))
            out = out + delta

        return out

    def update_context(self, feedback):
        self.context = 0.5*self.context + 0.5*feedback.mean(dim=0, keepdim=True)

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
# Main training
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

    # Reward
    def reward_fn(gen_out):
        return -((gen_out - 1.0)**2).mean()

    NUM_GENS = 500
    for g in range(NUM_GENS):
        enc_out = encoder(data.x, data.edge_index, edge_attr=data.edge_attr)

        # Train evaluator
        gen_out_eval = generator(enc_out).detach()
        true_reward = reward_fn(gen_out_eval)
        pred_reward = evaluator(gen_out_eval)
        eval_loss = F.mse_loss(pred_reward, true_reward.expand_as(pred_reward))
        eval_opt.zero_grad()
        eval_loss.backward()
        eval_opt.step()

        # Train generator
        gen_out = generator(enc_out)
        pred_reward = evaluator(gen_out)
        gen_loss = -pred_reward.mean()
        gen_opt.zero_grad()
        gen_loss.backward()
        gen_opt.step()

        generator.update_context(gen_out.detach())

        if (g+1) % 20 == 0 or g == 0:
            print(f"Step {g+1:03d} | True reward: {true_reward.item():.4f} | Pred: {pred_reward.mean().item():.4f} | Gen loss: {gen_loss.item():.4f}")

    print("Final generator context:", generator.context)


