import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Genome Decoder: Transformer + RNN weight generator
# -----------------------------
class RNNWeightGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, chunk_size, n_chunks):
        super().__init__()
        self.rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, chunk_size)
        self.n_chunks = n_chunks

    def forward(self, latent):
        h = latent.unsqueeze(1)
        out_chunks = []
        hidden = None
        for _ in range(self.n_chunks):
            _, hidden = self.rnn(h, hidden)
            chunk = self.output(hidden.squeeze(0))
            out_chunks.append(chunk)
            h = chunk.unsqueeze(1)
        return torch.cat(out_chunks, dim=-1)

class GenomeDecoder(nn.Module):
    def __init__(self, genome_len=16, genome_dim=16, hidden_dim=512, n_layers=2,
                 n_heads=4, embed_dim=256, chunk_size=64):
        super().__init__()
        self.embed = nn.Linear(genome_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim*4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.embed_dim = embed_dim

        total_params = 4*embed_dim*embed_dim + 4*embed_dim  # Q/K/V/O + biases
        n_chunks = (total_params + chunk_size - 1)//chunk_size
        self.rnn_gen = RNNWeightGenerator(hidden_dim, hidden_dim, chunk_size, n_chunks)
        self.total_params = total_params

    def forward(self, genome_seq):
        h = self.embed(genome_seq)
        h = self.encoder(h)
        latent = h.mean(dim=1)
        flat_params = self.rnn_gen(latent)
        return flat_params[:, :self.total_params]

# -----------------------------
# Agent Attention
# -----------------------------
class AgentAttention(nn.Module):
    def __init__(self, embed_dim=256, n_heads=8, chunk_size=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)

    def forward(self, x, flat_params):
        # x: [batch, seq_len, embed_dim]
        batch_size = x.size(0)
        outputs = []
        for i in range(batch_size):
            d = self.embed_dim
            offset = 0
            q = flat_params[i, offset:offset+d*d].view(d,d); offset += d*d
            k = flat_params[i, offset:offset+d*d].view(d,d); offset += d*d
            v = flat_params[i, offset:offset+d*d].view(d,d); offset += d*d
            o = flat_params[i, offset:offset+d*d].view(d,d); offset += d*d
            q_b = flat_params[i, offset:offset+d]; offset += d
            k_b = flat_params[i, offset:offset+d]; offset += d
            v_b = flat_params[i, offset:offset+d]; offset += d
            o_b = flat_params[i, offset:offset+d]; offset += d

            self.mha.in_proj_weight.data = torch.cat([q,k,v], dim=0)
            self.mha.in_proj_bias.data = torch.cat([q_b,k_b,v_b], dim=0)
            self.mha.out_proj.weight.data = o
            self.mha.out_proj.bias.data = o_b

            out, _ = self.mha(x[i:i+1], x[i:i+1], x[i:i+1])
            outputs.append(out)
        return torch.cat(outputs, dim=0)

# -----------------------------
# Blender network (shared GRU)
# -----------------------------
class BlenderGRU(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, agent_vectors, incoming_vectors, attention_weights):
        # Simple weighted update
        combined = agent_vectors + (incoming_vectors * attention_weights.unsqueeze(-1))
        out, _ = self.gru(combined)
        return out

# -----------------------------
# Aggregator (RNN-style)
# -----------------------------
class AggregatorRNN(nn.Module):
    def __init__(self, embed_dim=256, hidden_dim=256):
        super().__init__()
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 1)  # reward prediction

    def forward(self, x, steps=3):
        h = None
        for _ in range(steps):
            out, h = self.rnn(x, h)
        reward = self.output(out.mean(dim=1))
        return reward

# -----------------------------
# Full Hybrid Optimizer Loop
# -----------------------------
def hybrid_train_step(genomes, decoder, attention, blender, aggregator, grid, alpha=1.0, beta=0.1):
    """
    genomes: [batch, genome_len, genome_dim] (learnable)
    grid: task input
    alpha: task reward weight
    beta: attention reward weight
    """
    flat_params = decoder(genomes)
    agent_vectors = torch.randn(grid.size(0), 1, attention.embed_dim)  # initial hypothesis vectors
    agent_vectors = attention(agent_vectors, flat_params)  # forward attention

    # Blender updates agent vectors
    # For simplicity, assume self-connection as incoming_vectors and attention_weights=1
    agent_vectors = blender(agent_vectors, agent_vectors, torch.ones(agent_vectors.size(0)))

    # Aggregator computes reward
    task_reward = aggregator(agent_vectors)

    # Optional: attention-based creativity reward
    attention_reward = agent_vectors.norm(dim=-1).mean(dim=-1, keepdim=True)  # simple proxy

    total_loss = - (alpha * task_reward + beta * attention_reward).mean()
    total_loss.backward()

    # Adam step for decoder + aggregator + blender
    # genomes are updated via their own gradients (tiny memory cost)
    return total_loss.item()

# -----------------------------
# Example usage
# -----------------------------
batch_size = 16
genome_len, genome_dim = 16, 16
decoder = GenomeDecoder(genome_len, genome_dim)
attention = AgentAttention()
blender = BlenderGRU()
aggregator = AggregatorRNN()
genomes = nn.Parameter(torch.randn(batch_size, genome_len, genome_dim))
grid = torch.randn(batch_size, 10, 10)  # placeholder for ARC input

optimizer = torch.optim.Adam(list(decoder.parameters()) +
                             list(attention.parameters()) +
                             list(blender.parameters()) +
                             list(aggregator.parameters()) +
                             [genomes], lr=1e-3)

loss = hybrid_train_step(genomes, decoder, attention, blender, aggregator, grid)
optimizer.step()
