import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data


class GNNEncoder(nn.Module):
    """GNN encoder for ARC-style tasks with edge features and batching."""

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
        # First conv (no residual)
        x = self.convs[0](x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        # Subsequent convs with residuals
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


class TaskObject:
    def __init__(self, name, obj):
        self.name = name
        self.obj = obj


class Task:
    def __init__(self, name, train_pairs, test_pair):
        self.name = name
        self.train_pairs = train_pairs
        self.test_pairs = test_pair
        self.num_train = len(train_pairs)

    def convert_to_graph(self, grid):
        """
        Convert ARC grid to graph:
        - Nodes: one per color (flattened bitboard)
        - Edges: fully connected, edge features = cosine similarity
        """
        colors = torch.unique(grid)
        num_nodes = len(colors)
        H, W = grid.shape

        # Node features
        x = torch.stack([(grid == c).float().flatten() for c in colors], dim=0)

        # Fully connected edges excluding self-loops
        row, col = torch.meshgrid(torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij")
        edge_index = torch.stack([row.flatten(), col.flatten()], dim=0)
        mask = row.flatten() != col.flatten()
        edge_index = edge_index[:, mask]

        # Edge features: cosine similarity
        edge_attr = []
        for i in range(edge_index.size(1)):
            s, t = edge_index[:, i]
            sim = torch.dot(x[s], x[t]) / (x[s].norm() * x[t].norm() + 1e-8)
            edge_attr.append(torch.tensor([sim]))
        edge_attr = torch.stack(edge_attr, dim=0)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def generate_variation(self, grid):
        """Placeholder for data augmentation (rotation, flips, color swaps)."""
        return grid


class Evaluator(nn.Module):
    def __init__(self, input_dim, input_tokens):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, batch_first=True)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: [batch_size, input_tokens, input_dim]
        x, _ = self.attn(x, x, x)
        x = F.relu(self.fc(x))  # [batch_size, input_tokens, 1]
        x = x.mean(dim=1)       # [batch_size, 1]
        return x

class Generator(nn.Module):
    def __init__(self, input_dim, input_tokens, context_dim=32, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.input_tokens = input_tokens
        self.context_dim = context_dim
        self.attn = nn.MultiheadAttention(embed_dim=input_dim + context_dim, num_heads=4, batch_first=True)
        self.blocks = nn.Sequential(*[
            nn.Sequential(
                nn.MultiheadAttention(embed_dim=input_dim + context_dim, num_heads=4, batch_first=True),
                nn.Linear(input_dim + context_dim, input_dim + context_dim),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_dim + context_dim, input_dim)
        self.out_proj = nn.Linear(input_dim, 1)

        self.context = None  # Will be initialized per batch
        self.context_updatorattn = nn.MultiheadAttention(embed_dim=context_dim + input_dim + 1, num_heads=1, batch_first=True)
        self.context_updatorfc = nn.Linear(context_dim + input_dim + 1, context_dim)

    def forward(self, x):
        # x: [batch_size, input_tokens, input_dim]
        batch_size = x.size(0)
        device = x.device
        if self.context is None or self.context.size(0) != batch_size:
            # Initialize context per batch
            self.context = torch.zeros(batch_size, 1, self.context_dim, device=device)
        context_expanded = self.context.expand(-1, self.input_tokens, -1)  # [batch_size, input_tokens, context_dim]
        x = torch.cat([x, context_expanded], dim=-1)  # [batch_size, input_tokens, input_dim + context_dim]
        x, _ = self.attn(x, x, x)
        x = F.relu(self.fc(x))  # [batch_size, input_tokens, input_dim]
        out = torch.sigmoid(self.out_proj(x)).squeeze(-1)  # [batch_size, input_tokens]

        # Add small random noise
        delta = (torch.rand_like(out) - 0.5) * 0.1
        out = out + delta

        return out

    def update_context(self, feedbackv, reward):
        # feedbackv: [batch_size, input_tokens, input_dim]
        # reward: [batch_size, 1]
        batch_size = feedbackv.size(0)
        device = feedbackv.device
        if self.context is None or self.context.size(0) != batch_size:
            self.context = torch.zeros(batch_size, 1, self.context_dim, device=device)
        # Expand reward to match tokens
        reward_expanded = reward.unsqueeze(1).expand(-1, feedbackv.size(1), -1)  # [batch_size, input_tokens, 1]
        # Expand context to match tokens
        context_expanded = self.context.expand(-1, feedbackv.size(1), -1)  # [batch_size, input_tokens, context_dim]
        # Concatenate along last dim
        feedback = torch.cat([context_expanded, feedbackv, reward_expanded], dim=2)  # [batch_size, input_tokens, context_dim + input_dim + 1]
        new_context, _ = self.context_updatorattn(feedback, feedback, feedback)
        new_context = F.relu(self.context_updatorfc(new_context)).mean(dim=1, keepdim=True)  # [batch_size, 1, context_dim]
        self.context = new_context.detach()
class decoder
def train_step(decoder,generator,evaluator)
# Example usage in __main__:
if __name__ == "__main__":
    # Toy ARC grid
    grid = torch.tensor([
        [1, 0, 0],
        [1, 2, 0],
        [0, 2, 2],
    ])

    task = Task("toy", [], [])
    data = task.convert_to_graph(grid)

    # Node feature dim = H*W
    in_dim = data.x.size(1)
    encoder = GNNEncoder(in_dim=in_dim, out_dim=64, out_tokens=4, hidden=32)
    evaluator = Evaluator(input_dim=64, input_tokens=4)
    out = encoder(data.x, data.edge_index, edge_attr=data.edge_attr)
    print("Node features shape:", data.x.shape)
    print("Edge index shape:", data.edge_index.shape, "Edge features shape:", data.edge_attr.shape)
    print("Encoder output shape:", out.shape)
    print(f"encoder params: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")
    print(f"evaluator params: {sum(p.numel() for p in evaluator.parameters() if p.requires_grad)}")
    generator = Generator(input_dim=64, input_tokens=4)
    print(f"generator params: {sum(p.numel() for p in generator.parameters() if p.requires_grad)}")
    gen_out = generator(out)
    print("Generator output shape:", gen_out.shape)

    # Simulate feedback: Evaluator evaluates generator's output
    # For demonstration, let's use the encoder output as input to the evaluator
    eval_score = evaluator(out)  # [batch_size, 1]
    print("Evaluator score shape:", eval_score.shape)

    # Feedback vector: for simplicity, use the encoder output (could be gradients, etc.)
    feedbackv = out  # [batch_size, input_tokens, input_dim]
    reward = eval_score  # [batch_size, 1]

    generator.update_context(feedbackv, reward)
    print("Updated context shape:", generator.context.shape)



