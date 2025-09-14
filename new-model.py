import torch
import torch.nn as nn
import torch.nn.functional as F
def cut_at_stop(batch, stop_id):
    result = []
    for seq in batch:
        idx = (seq == stop_id).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            result.append(seq[:idx[0]])
        else:
            result.append(seq)
    return torch.tensor(data=result,dtype=torch.float32)
class Mutator(nn.Module):
    def __init__(self, sequence, seq_len=256, embed_dim=4, num_heads=4):
        super(Mutator, self).__init__()
        self.sequence = sequence
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.atten = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.confhead = nn.Linear(seq_len * embed_dim, 1)
        self.updater = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.updaterfc = nn.Linear(seq_len * embed_dim, seq_len)

    def forward(self, x):
        batch_size = x.size(0)

        # Conv over sequence
        x = self.conv(x)  # [batch, 1, L]

        # Pad to fixed length
        pad = (0, self.seq_len - x.shape[2])
        x = F.pad(x, pad, mode='constant', value=0)  # [batch, 1, seq_len]

        # Repeat across embed_dim
        x = x.repeat(1, self.embed_dim, 1)  # [batch, embed_dim, seq_len]

        # Rearrange to (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # [batch, seq_len, embed_dim]

        # First attention
        out, weights = self.atten(x, x, x)  # [batch, seq_len, embed_dim]

        # Confidence score
        conf = torch.sigmoid(self.confhead(out.flatten(1)))  # [batch, 1]

        # If confidence is high, return stored sequence
        if (conf > 0.5).all():
            return self.sequence.expand(batch_size, -1), conf, weights

        # Otherwise update
        updated, _ = self.updater(out, out, out)
        updated_seq = torch.relu(self.updaterfc(updated.flatten(1))) # [batch, seq_len]
        updated_seq=cut_at_stop(updated_seq, stop_id=0)
        self.sequence=updated_seq  # Update stored sequence
        return updated_seq, conf, weights


# Test
mutator = Mutator(torch.randn(256))  # store a default seq of length 256
inp = torch.randn(1, 1, 5)  # batch=4
out, conf, weights = mutator(inp)

print(out.shape)      # (4, 256)
print(conf.shape)     # (4, 1)
print(weights.shape)  # (4, 4, 256, 256)
