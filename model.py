# PyTorch minimal example implementing the "Hybrid Mutator" sketch.
# This code builds:
# - GenomeDecoder (small MLP hypernetwork) that maps a genome vector -> a small set of parameters
# - HybridMutator which uses a GRU sequential core + a cross-attention block + gating head.
# - A tiny demo: population of agents, genomes -> mutators -> mutate vectors -> compute simple fitness delta.
#
# This is a compact, runnable prototype to illustrate the idea. Not production-grade, but enough to play with.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
torch.manual_seed(0)


class GenomeDecoder(nn.Module):
    """
    Small hypernetwork: genome (g_dim) -> a handful of parameter tensors that will be *injected*
    into the Mutator. To keep the bottleneck effective we only generate small weight chunks:
      - gating bias (for keep_prob)
      - final delta projection weights and bias (maps GRU hidden -> token_delta)
    The mutator contains shared base weights; decoder produces additive deltas to those weights.
    """
    def __init__(self, genome_dim=128, hidden=256, token_dim=64, gru_hidden=256):
        super().__init__()
        self.genome_dim = genome_dim
        self.token_dim = token_dim
        self.gru_hidden = gru_hidden
        self.net = nn.Sequential(
            nn.Linear(genome_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        # outputs sizes
        self.gate_bias_out = nn.Linear(hidden, 1)  # scalar gate bias per mutator (broadcasted)
        # final projection: hidden -> token_dim (we generate small matrix and bias)
        # We'll generate a vector that will be used as bias into a shared final linear layer.
        self.delta_bias_out = nn.Linear(hidden, token_dim)
        # Additionally produce a small "scale" for final weight (scalar)
        self.weight_scale_out = nn.Linear(hidden, 1)

    def forward(self, genome):
        h = self.net(genome)
        gate_bias = self.gate_bias_out(h)        # (batch, 1)
        delta_bias = self.delta_bias_out(h)      # (batch, token_dim)
        weight_scale = torch.tanh(self.weight_scale_out(h))  # keep scale moderate
        return {
            "gate_bias": gate_bias,
            "delta_bias": delta_bias,
            "weight_scale": weight_scale
        }


class HybridMutator(nn.Module):
    """
    Hybrid Mutator:
      - tokenized input: [L, token_dim]
      - sequential core: GRU over tokens (a chosen scan order)
      - cross-attention to condensed context tokens from neighbors (single-head MHA)
      - gating head: produces keep_prob and token delta
    The mutator contains some shared parameters and allows injection of small hypernet outputs
    from the GenomeDecoder to bias gating/delta outputs.
    """
    def __init__(self, token_dim=64, gru_hidden=256, attn_dim=64, num_context_tokens=8):
        super().__init__()
        self.token_dim = token_dim
        self.gru_hidden = gru_hidden
        self.attn_dim = attn_dim
        self.num_context_tokens = num_context_tokens

        # token embed/projection if tokens are raw features
        self.input_proj = nn.Linear(token_dim, token_dim)

        # GRU treats tokens as sequence (we'll feed tokens along sequence dim)
        self.gru = nn.GRU(input_size=token_dim, hidden_size=gru_hidden, batch_first=True)

        # cross-attention: query from GRU outputs, keys/values from context tokens
        self.cross_attn = nn.MultiheadAttention(embed_dim=gru_hidden, num_heads=1, batch_first=True)

        # Shared final projector from GRU hidden -> token delta (we allow decoder to bias its bias/scale)
        self.final_proj = nn.Linear(gru_hidden, token_dim)
        # gating MLP (produces one logit per token for keep probability)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden//2),
            nn.ReLU(),
            nn.Linear(gru_hidden//2, 1)  # single logit per token
        )

        # small normalization
        self.norm = nn.LayerNorm(token_dim)

    def condense_context(self, neighbor_vectors):
        """
        neighbor_vectors: tensor (batch, k, L, token_dim)
        Condense each neighbor vector by a small pooling & MLP to produce a set of context tokens.
        We'll pool across the vector to get `num_context_tokens` tokens per neighbor, then flatten.
        """
        # neighbor_vectors -> (batch, k, L, token_dim)
        b, k, L, td = neighbor_vectors.shape
        # simple linear reduce to num_context_tokens via reshaping
        # reshape L -> (num_context_tokens, seg_len)
        seg_len = max(1, L // self.num_context_tokens)
        # truncate extra if needed
        truncated = neighbor_vectors[..., :seg_len*self.num_context_tokens, :]
        truncated = truncated.view(b, k, self.num_context_tokens, seg_len, td)  # (b,k,ct,seg,td)
        pooled = truncated.mean(dim=3)  # (b,k,ct,td)
        # collapse neighbors into one context bank: (b, k*num_context_tokens, td)
        context = pooled.view(b, k*self.num_context_tokens, td)
        # project to attn dim if needed (we'll let cross_attn handle differences)
        return context  # (b, K_ct, td)

    def forward(self, token_vectors, neighbor_vectors, hyperparams=None):
        """
        token_vectors: (batch, L, token_dim)
        neighbor_vectors: (batch, k, L, token_dim)
        hyperparams: dict with keys gate_bias (batch,1), delta_bias (batch,token_dim), weight_scale (batch,1)
        """
        b, L, td = token_vectors.shape

        x = self.input_proj(token_vectors)  # (b,L,td)
        x = self.norm(x)
        # run GRU sequentially across tokens
        # GRU expects inputs (batch, seq, input_size)
        gru_out, _ = self.gru(x)  # (b, L, gru_hidden)

        # condense neighbor context
        context = self.condense_context(neighbor_vectors)  # (b, K_ct, td)
        # project context into GRU hidden dim to be used as key/value for attention
        # we'll do a simple linear on the fly to match dimensions
        # expand context to (b, K_ct, gru_hidden) via linear
        context_proj = F.relu(nn.Linear(td, self.gru_hidden).to(token_vectors.device)(context))

        # cross-attention: queries = gru_out, keys/values = context_proj
        # MultiheadAttention expects (b, seq, dim)
        attn_out, attn_weights = self.cross_attn(query=gru_out, key=context_proj, value=context_proj)
        # combine
        combined = gru_out + 0.1 * attn_out  # small residual attention contribution

        # gating logits per token
        gate_logits = self.gate_mlp(combined).squeeze(-1)  # (b,L)

        # final delta projection (apply optional hyperparams scaling and bias)
        delta = self.final_proj(combined)  # (b,L,token_dim)

        if hyperparams is not None:
            # hyperparams are per-mutator (per batch element). Apply them broadcasted across tokens.
            gate_bias = hyperparams["gate_bias"].view(b, 1)  # (b,1)
            delta_bias = hyperparams["delta_bias"].view(b, 1, td)  # (b,1,td)
            weight_scale = hyperparams["weight_scale"].view(b, 1, 1)
            gate_logits = gate_logits + gate_bias  # shift gating
            delta = delta * (1.0 + weight_scale) + delta_bias  # scale + bias

        keep_probs = torch.sigmoid(gate_logits)  # (b,L) in (0,1)
        # apply delta with gated residual: token' = keep_prob * token + (1-keep) * (token + delta)
        keep = keep_probs.unsqueeze(-1)  # (b,L,1)
        mutated = keep * token_vectors + (1.0 - keep) * (token_vectors + delta)

        return mutated, {"keep_probs": keep_probs, "delta_norm": delta.norm(dim=-1).mean().item(), "attn_weights": attn_weights}


# -------------------- Tiny demo to show a mutation step --------------------
def demo_run():
    device = torch.device("cpu")
    # hyperparams
    batch = 4  # population size (number of mutators produced)
    k_neighbors = 3
    H = 6 * 6  # flattened grid length (e.g., 6x6 tasks)
    token_dim = 32
    genome_dim = 128

    # make random initial vectors (these are the "notebooks" for each agent)
    # For simplicity: we'll use batch vectors (we can mutate vectors of other agents)
    vectors = torch.randn(batch, H, token_dim, device=device) * 0.5  # (batch, L, token_dim)

    # neighbor buffers: for each mutator (b), we give k neighbor vectors (randomly selected)
    neighbor_vectors = torch.randn(batch, k_neighbors, H, token_dim, device=device) * 0.5

    # genomes: each mutator is produced from a genome vector
    genomes = torch.randn(batch, genome_dim, device=device) * 0.1

    # build decoder + mutator
    decoder = GenomeDecoder(genome_dim=genome_dim, hidden=256, token_dim=token_dim, gru_hidden=256).to(device)
    mutator = HybridMutator(token_dim=token_dim, gru_hidden=256, attn_dim=64, num_context_tokens=8).to(device)

    # decode genomes -> hyperparams for each mutator
    hyperparams = decoder(genomes)
    # forward mutate: each mutator mutates its own vector (for demo we keep simple mapping)
    mutated, info = mutator(vectors, neighbor_vectors, hyperparams=hyperparams)

    # compute tiny dummy fitness: we define a "target pattern" and measure negative MSE
    target = torch.zeros_like(vectors)  # prefer zero vectors; mutators should make vectors smaller
    def fitness(x):
        return -F.mse_loss(x, target, reduction='none').mean(dim=(1,2))  # (batch,) negative loss is higher is better

    orig_f = fitness(vectors)
    new_f = fitness(mutated)
    delta = new_f - orig_f  # improvement (positive = better)

    print("orig_f:", orig_f.detach().numpy())
    print("new_f: ", new_f.detach().numpy())
    print("delta: ", delta.detach().numpy())
    print("info delta_norm:", info["delta_norm"])
    print("avg keep prob:", info["keep_probs"].mean().item())

    return {
        "vectors": vectors,
        "mutated": mutated,
        "orig_f": orig_f,
        "new_f": new_f,
        "delta": delta,
        "info": info
    }

# run demo
_out = demo_run()
