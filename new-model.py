import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Decoder ----------------
class GenomeDecoder(nn.Module):
    def __init__(self, genome_dim=128, hidden=256, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(genome_dim+1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, genome, idx):
        idx = idx.float().unsqueeze(0) if idx.dim()==0 else idx.float()
        if genome.dim()==1:
            genome = genome.unsqueeze(0)
        x = torch.cat([genome, idx.expand(genome.size(0), -1)], dim=-1)
        return self.net(x)

# ---------------- Differentiable Mutator ----------------
class HybridMutator(nn.Module):
    def __init__(self, token_dim=64, gru_hidden=256, attn_dim=64, num_context_tokens=8):
        super().__init__()
        self.token_dim = token_dim
        self.gru_hidden = gru_hidden
        self.attn_dim = attn_dim
        self.num_context_tokens = num_context_tokens

        self.input_proj = nn.Linear(token_dim, token_dim)
        self.gru = nn.GRU(input_size=token_dim, hidden_size=gru_hidden, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=gru_hidden, num_heads=1, batch_first=True)
        self.final_proj = nn.Linear(gru_hidden, token_dim)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden//2),
            nn.ReLU(),
            nn.Linear(gru_hidden//2, 1)
        )
        self.norm = nn.LayerNorm(token_dim)

    def condense_context(self, neighbor_vectors):
        b, k, L, td = neighbor_vectors.shape
        seg_len = max(1, L // self.num_context_tokens)
        truncated = neighbor_vectors[..., :seg_len*self.num_context_tokens, :]
        truncated = truncated.view(b, k, self.num_context_tokens, seg_len, td)
        pooled = truncated.mean(dim=3)
        context = pooled.view(b, k*self.num_context_tokens, td)
        return context

    def forward(self, token_vectors, neighbor_vectors, weights=None):
        b, L, td = token_vectors.shape
        x = self.input_proj(token_vectors)
        x = self.norm(x)
        gru_out, _ = self.gru(x)

        context = self.condense_context(neighbor_vectors)
        context_proj = F.relu(nn.Linear(td, self.gru_hidden).to(token_vectors.device)(context))
        attn_out, attn_weights = self.cross_attn(query=gru_out, key=context_proj, value=context_proj)
        combined = gru_out + 0.1 * attn_out

        gate_logits = self.gate_mlp(combined).squeeze(-1)
        delta = self.final_proj(combined)

        keep_probs = torch.sigmoid(gate_logits)
        mutated = keep_probs.unsqueeze(-1) * token_vectors + (1.0 - keep_probs.unsqueeze(-1)) * (token_vectors + delta)

        return mutated, {
            "keep_probs": keep_probs,
            "delta_norm": delta.norm(dim=-1).mean(),
            "attn_weights": attn_weights
        }

# ---------------- Differentiable Agent ----------------
class Agent(nn.Module):
    def __init__(self, vector_dim, genome_dim, mutator_param_dim):
        super().__init__()
        self.vector = nn.Parameter(torch.randn(1, vector_dim, vector_dim))  # treat as [b, L, td]
        self.genome = nn.Parameter(torch.randn(genome_dim))
        self.decoder = GenomeDecoder(genome_dim, hidden=256, out_dim=mutator_param_dim)
        self.mutator = HybridMutator(token_dim=vector_dim, gru_hidden=256, attn_dim=64, num_context_tokens=8)

    def step(self, neighbors):
        weights = self.decoder(self.genome, torch.tensor([0], device=self.genome.device))
        neighbor_batch = torch.stack(neighbors)
        mutated, _ = self.mutator(self.vector, neighbor_batch, weights)
        self.vector = nn.Parameter(mutated.detach())
        return self.vector

# ---------------- Swarm ----------------
def swarm_step(agents, k=3):
    all_vectors = [a.vector for a in agents]
    updated = []
    for i, agent in enumerate(agents):
        sims = [F.cosine_similarity(agent.vector.view(-1), v.view(-1), dim=0) for v in all_vectors]
        _, idx = torch.topk(torch.tensor(sims), k+1)
        neighbors = [all_vectors[j] for j in idx if j != i][:k]
        updated.append(agent.step(neighbors))
    return updated

# ---------------- Training ----------------
def train_swarm(agents, steps=4, k=3, lr=1e-3, epochs=100):
    opt = torch.optim.Adam([a.genome for a in agents], lr=lr)
    for ep in range(epochs):
        opt.zero_grad()
        for _ in range(steps):
            swarm_step(agents, k)
        all_vecs = torch.stack([a.vector for a in agents])
        loss = all_vecs.pow(2).mean()  # minimize norm
        (-loss).backward()
        opt.step()
        if ep % 10 == 0:
            print(f"epoch {ep} reward={-loss.item():.4f}")

# ---------------- Demo ----------------
if __name__ == "__main__":
    agents = [Agent(vector_dim=32, genome_dim=128, mutator_param_dim=512) for _ in range(5)]
    train_swarm(agents, steps=3, k=2, lr=1e-3, epochs=50)
