import torch
import torch.nn as nn
import torch.nn.functional as F

class Mutator(nn.Module):
    def __init__(self):
        super(Mutator, self).__init__()
        self.shape = (1, 1, 256)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        # embed_dim must be divisible by num_heads
        self.atten = nn.MultiheadAttention(embed_dim=4, num_heads=4, batch_first=True)
        self.confhead = nn.Linear(1024, 1)  # Confidence head
    def forward(self, x):
        # x: [batch, channels, seq_len]
        x = self.conv(x)  # [batch, 1, L]

        # Pad to fixed length
        pad = (0, self.shape[2] - x.shape[2])
        x = F.pad(x, pad, mode='constant', value=0)  # [batch, 1, 256]

        # Repeat feature dim → embed_dim=4
        x = x.repeat(1, 4, 1)  # [batch, 4, 256]

        # Rearrange to (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # [batch, 256, 4]

        # Apply attention
        out, weights = self.atten(x, x, x)  # out: [batch, 256, 4]
        conf=self.confhead(out.flatten(1))
        return out, conf, weights
        
        
mutator = Mutator()
inp = torch.randn(1, 1, 5)
out,conf,weights = mutator(inp)
print(out.shape)  # (1, 256, 4)
print(conf)
print(weights.shape)  # (1, 256, 256)
