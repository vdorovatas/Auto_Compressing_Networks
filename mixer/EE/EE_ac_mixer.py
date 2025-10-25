import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class MLPMixer(nn.Module):
    def __init__(self,in_channels=3,img_size=32, patch_size=4, hidden_size=512, hidden_s=256,
                hidden_c=2048, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False, vanilla=True):
        super(MLPMixer, self).__init__()
        num_patches = img_size // patch_size * img_size // patch_size
        # (b, c, h, w) -> (b, d, h//p, w//p) -> (b, h//p*w//p, d)
        self.is_cls_token = is_cls_token
        self.vanilla = vanilla

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size ,kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1

        self.mixer_layers = nn.ModuleList([MixerLayer(num_patches, hidden_size, hidden_s,
                            hidden_c, drop_p, off_act, vanilla=self.vanilla) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_size)

        self.branch_layers = [4, 8, 12]
        #self.branches = nn.ModuleList([nn.Linear(hidden_size, num_classes) for _ in range(num_layers+1)])
        self.branches = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, 2*hidden_size), 
                                                     nn.GELU(), nn.Linear(2*hidden_size, num_classes)) for _ in range(len(self.branch_layers))])
        
        self.clf = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)

        current = out
        intermid = []
        i_br = 0
        for i, layer in enumerate(self.mixer_layers):
            out = layer(out)
            current = current + out
            
            if i in self.branch_layers:
                intermid.append(self.branches[i_br](out[:, 0] if self.is_cls_token else out.mean(dim=1)))
                i_br += 1

        out = current
        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)

        return out, intermid

class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act, vanilla):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)

        self.vanilla = vanilla
        self.relu =  nn.ReLU()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        ########
        out = self.ln(self.relu(self.linear(out)+x))
        ########
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x
    
    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x:x

    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x)) + self.linear1(x) ))
        out = self.do2(self.fc2(out) + self.linear2(out))
        return out

if __name__ == '__main__':
    net = MLPMixerL(
        in_channels=3,
        img_size=32,
        patch_size=4,
        hidden_size=128,
        hidden_s=512,
        hidden_c=64,
        num_layers=8,
        num_classes=10,
        drop_p=0.,
        off_act=False,
        is_cls_token=True,
        vanilla=False
        )

