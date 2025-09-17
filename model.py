import torch
import torch.nn as nn
import torch.nn.functional as F





# ---------------- Decoder ----------------
class GenomeDecoder(nn.Module):
    """
    Decoder takes a genome and index, outputs a 256-dim vector.
    Used repeatedly to 'grow' all parameters of a mutator.
    """
    def __init__(self, genome_dim=128, hidden=256, out_dim=256):
        super().__init__()
        self.out_dim = out_dim
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


# ---------------- Helper ----------------
def fill_params(module, genome, decoder):
    """
    Fill parameters of any nn.Module with values from decoder.
    Supports Linear, GRU, MultiheadAttention, LayerNorm.
    """
    idx = 0
    for name, param in module.named_parameters():
        numel = param.numel()
        values = []
        while len(values) < numel:
            out = decoder(genome, torch.tensor([idx]))
            idx += 1
            values.extend(out.view(-1).tolist())
        flat = torch.tensor(values[:numel], dtype=param.dtype)
        new_param = nn.Parameter(flat.view_as(param), requires_grad=False)

        # Correct injection into nested modules
        name_parts = name.split('.')
        submod = module
        for p in name_parts[:-1]:
            submod = getattr(submod, p)
        submod._parameters[name_parts[-1]] = new_param

    return idx


# ---------------- Mutator ----------------
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

    def forward(self, token_vectors, neighbor_vectors):
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
        keep = keep_probs.unsqueeze(-1)
        mutated = keep * token_vectors + (1.0 - keep) * (token_vectors + delta)

        return mutated, {
            "keep_probs": keep_probs,
            "delta_norm": delta.norm(dim=-1).mean().item(),
            "attn_weights": attn_weights
        }
# ---------------- Full Run ----------------
def full_run(num_genomes=10, batch=16, genome_dim=128,device="cpu"):
    genomes=torch.randn(num_genomes, genome_dim) * 0.1
    mutators = [HybridMutator(token_dim=32, gru_hidden=256, attn_dim=64, num_context_tokens=8).to(device=device) for _ in range(num_genomes)]
    decoder = GenomeDecoder(genome_dim=genome_dim, hidden=256, out_dim=256).to(device=device)
    for i in range(num_genomes):
        fill_params(mutators[i], genomes[i], decoder)
    
# ---------------- Demo ----------------
def demo_run():
    device = torch.device("cpu")
    batch = 4
    k_neighbors = 3
    H = 6 * 6
    token_dim = 32
    genome_dim = 128

    vectors = torch.randn(batch, H, token_dim, device=device) * 0.5
    neighbor_vectors = torch.randn(batch, k_neighbors, H, token_dim, device=device) * 0.5
    genomes = torch.randn(batch, genome_dim, device=device) * 0.1

    decoder = GenomeDecoder(genome_dim=genome_dim, hidden=256, out_dim=256).to(device)
    mutator = HybridMutator(token_dim=token_dim, gru_hidden=256, attn_dim=64, num_context_tokens=8).to(device)

    # fill mutator params once per genome
    for b in range(batch):
        fill_params(mutator, genomes[b], decoder)
        print("After filling genome", b, "mutator params:", sum(p.numel() for p in mutator.parameters()))

    mutated, info = mutator(vectors, neighbor_vectors)

    target = torch.zeros_like(vectors)
    def fitness(x):
        return -F.mse_loss(x, target, reduction='none').mean(dim=(1,2))

    orig_f = fitness(vectors)
    new_f = fitness(mutated)
    delta = new_f - orig_f

    print("orig_f:", orig_f.detach().numpy())
    print("new_f: ", new_f.detach().numpy())
    print("delta: ", delta.detach().numpy())
    print("info delta_norm:", info["delta_norm"])
    print("avg keep prob:", info["keep_probs"].mean().item())
    print("decoder params:", sum(p.numel() for p in decoder.parameters()))
    return {
        "vectors": vectors,
        "mutated": mutated,
        "orig_f": orig_f,
        "new_f": new_f,
        "delta": delta,
        "info": info
    }
class Agent:
    def __init__(self, vector_dim, genome_dim, mutator,coordinate:tuple):
       self.coordinate=coordinate
       self.vector = torch.randn(vector_dim)
       self.genome = torch.randn(genome_dim, requires_grad=True)
       self.decoder = GenomeDecoder(genome_dim, hidden=256, out_dim=256)
       self.mutator = HybridMutator(vector_dim, vector_dim)


    def step(self, neighbors):
        # decode genome into mutator weights
        weights = fill_params(self.mutator, self.genome, self.decoder)
        # average neighbor vectors
        neighbor_batch = torch.stack(neighbors)
        new_vector = self.mutator(neighbor_batch, weights)
        self.vector = new_vector.detach()
        return self.vector
    
def swarm_step(agents, k=3):
    # Stack coordinates and vectors
    coords = torch.stack([torch.tensor(a.coordinate, dtype=torch.float32) for a in agents])  # (N, D)
    vectors = torch.stack([a.vector for a in agents])  # (N, vector_dim)
    N = len(agents)

    # Compute pairwise distances (N, N)
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (N, N, D)
    dists = torch.norm(diff, dim=2)  # (N, N)
    dists.fill_diagonal_(float('inf'))  # Exclude self

    # Find k nearest neighbors for each agent
    knn_idx = torch.topk(dists, k, largest=False).indices  # (N, k)

    updated = []
    for i, agent in enumerate(agents):
        neighbor_vectors = vectors[knn_idx[i]]  # (k, vector_dim)
        updated_vector = agent.step([v for v in neighbor_vectors])
        updated.append(updated_vector)
    return updated


def fitness(agents):
    scores = torch.tensor([torch.norm(a.vector) for a in agents])
    # clip scores so bad agents don't drag reward down too much
    clipped = torch.clamp(scores, min=0.1 * scores.mean())
    return clipped.sum(), scores

if __name__ == "__main__":
    _out = demo_run()
