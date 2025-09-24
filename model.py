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
class LinearWithMemory(nn.Module):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.fastweights = nn.Parameter(torch.zeros(input_dim, output_dim), requires_grad=False)
    def forward(self, x):
        return x @ (self.weights + self.fastweights) + self.bias
class Generator(nn.Module):
    def __init__(self, input_dim, input_tokens, context_size=256, hidden_dim=128, num_blocks=2, num_heads=4):
        super().__init__()
        self.input_dim = input_dim
        self.input_tokens = input_tokens
        self.context_size = context_size
        self.input_proj = nn.Linear(input_dim + context_size, hidden_dim)
        self.contextcondenser = nn.Linear(input_dim * input_tokens, context_size)
        self.context = torch.zeros(1, context_size)  # [1, context_size]
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(hidden_dim),
                'attn': nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True),
                'ln2': nn.LayerNorm(hidden_dim),
                'ff': nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            })
            for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: [batch_size, input_tokens, input_dim]
        batch_size, input_tokens, input_dim = x.shape
        # Expand context to match tokens
        context = self.context.expand(batch_size, self.context_size)  # [batch_size, context_size]
        context = context.unsqueeze(1).expand(-1, input_tokens, -1)   # [batch_size, input_tokens, context_size]
        x = torch.cat((x, context), dim=-1)  # [batch_size, input_tokens, input_dim + context_size]
        x = self.input_proj(x)
        for block in self.blocks:
            x_res = x
            x = block['ln1'](x)
            x, _ = block['attn'](x, x, x)
            x = x + x_res

            x_res = x
            x = block['ln2'](x)
            x = block['ff'](x)
            x = x + x_res
        x = self.output_proj(x)  # [batch_size, input_tokens, input_dim]
        return x

    def update_context(self, x):
        # x: [batch_size, input_tokens, input_dim]
        batch_size = x.size(0)
        context = self.contextcondenser(x.reshape(batch_size, -1)).mean(dim=0, keepdim=True)
        self.context = context  # [1, context_size]
        return context

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
    # Simulate feedback and update context
    feedback = torch.randn_like(gen_out)
    generator.update_context(feedback)
    print("Updated context shape:", generator.context.shape)



