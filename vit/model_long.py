import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import random

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_size))

    def forward(self, x: Tensor) :
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1)
        # add position embedding
        x += self.positions
        return x
    
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,  in_channels, patch_size, emb_size, img_size):
        super(Embeddings, self).__init__()

        n_patches = (img_size // patch_size) * (img_size // patch_size)

        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=emb_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, emb_size))
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B = x.shape[0]
        #cls_tokens = self.cls_token.expand(B, -1, -1)

        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        #x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        self.scaling = (self.emb_size // num_heads) ** -0.5

        self.ln = nn.LayerNorm(self.emb_size)

    def forward(self, x : Tensor, mask: Tensor = None) -> Tensor:
        # split keys, queries and values in num_heads
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        att = F.softmax(energy, dim=-1) * self.scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return self.ln(out)

class ResidualAdd(nn.Module):
    def __init__(self, ln, fn, dr):
        super().__init__()
        self.fn = fn
        self.ln = ln
        self.dr = dr

    def forward(self, x, **kwargs):
        res = x
        x = self.dr(self.fn(self.ln(x), **kwargs))
        return x
    
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__()
        self.l1 = nn.Linear(emb_size, expansion * emb_size)
        self.gelu = nn.GELU()
        self.dr = nn.Dropout(drop_p)
        self.l2 = nn.Linear(expansion * emb_size, emb_size)

        self.d1 = nn.Linear(emb_size, expansion * emb_size, bias=False)
        self.d2 = nn.Linear(expansion * emb_size, emb_size, bias=False)
        I1 = self.dirac(expansion * emb_size, emb_size)
        I2 = I1.transpose(0,1)
        self.d1.weight = nn.Parameter(I1, requires_grad=False)
        self.d2.weight = nn.Parameter(I2, requires_grad=False)

    def forward(self, x):
        x = self.dr(self.gelu(self.l1(x) + self.d1(x)))
        x = self.l2(x) + self.d2(x)
        return x

    def dirac(self, n=10, m=5):
        identity_part = torch.eye(m)
        zero_part = torch.zeros((n - m, m))
        return torch.vstack((identity_part, zero_part))

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0,
                 ** kwargs):
        super(TransformerEncoderBlock, self).__init__()

        self.R1 = ResidualAdd(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
             )
        self.R2 = ResidualAdd(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
             )

        self.linear = nn.Linear(emb_size, emb_size)
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(emb_size)

    def forward(self, x):
        x = self.R1(x)
        x = self.R2(x)
        return x
    

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([TransformerEncoderBlock(**kwargs) for _ in range(depth)])
        self.encoder_norm = nn.LayerNorm(768)

    def forward(self, x):
        current = x
        intermid = [current]
        for i,layer in enumerate(self.layers):
            x = layer(x)
            current = current + x
            intermid.append(current)

        x = current
        return x, intermid

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes))
        
class ViTL(nn.Sequential):
    def __init__(self,
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super(ViTL, self).__init__()

        #self.E = Embeddings(in_channels, patch_size, emb_size, img_size)
        self.E = PatchEmbedding(in_channels, patch_size, emb_size, img_size)
        self.TE = TransformerEncoder(depth, emb_size=emb_size, **kwargs)
        self.C = ClassificationHead(emb_size, n_classes)
        #self.C_ft = ClassificationHead(emb_size, n_classes)

    def forward(self, x, eval_=False):
        x = self.E(x)
        x, intermid = self.TE(x)
        logits = self.C(x)

        intermid_preds = []
        if eval_:
            for i in intermid: intermid_preds.append(self.C(i.detach().clone()))

        return logits, intermid_preds

