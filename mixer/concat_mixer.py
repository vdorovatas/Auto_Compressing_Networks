'''
code based on: https://github.com/omihub777/MLP-Mixer-CIFAR/blob/main/mlp_mixer.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import copy

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
        
        self.concat_weights = nn.ModuleList([nn.Linear((i+1) * hidden_size, hidden_size, bias=False) for i in range(num_layers)])
        for i, layer in enumerate(self.concat_weights):
            self.init_sum_like(layer, i, hidden_size)
        self.clf = nn.Linear(hidden_size, num_classes)

    def forward(self, x, eval_= False, det=False):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out], dim=1)

        intermid = [out]
        for i, layer in enumerate(self.mixer_layers):
            out = self.concat(intermid, i)
            out = layer(out)
            intermid.append(out)
        intermid_preds = []
        if eval_==True: intermid_preds = [self.clf(self.ln(i[:, 0])) for i in intermid]

        out = self.ln(out)
        out = out[:, 0] if self.is_cls_token else out.mean(dim=1)
        out = self.clf(out)

        return out, intermid_preds
    
    def concat(self, previous_layers, i):
        concat_features = torch.cat(previous_layers, dim=-1)  # [B, S, (num_prev+1)*dim]
        return self.concat_weights[i](concat_features)


    def init_sum_like(self, linear_layer, layer_index, hidden_size):
        """
        Initialize linear layer to sum all same-position features from previous outputs.
        """
        with torch.no_grad():
            linear_layer.weight.zero_()
            for k in range(layer_index + 1):  # each previous layer
                start = k * hidden_size
                end = (k + 1) * hidden_size
                # add identity block
                linear_layer.weight[:, start:end] = torch.eye(hidden_size)

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
        mlp1_out = self.mlp1(x)
        mlp2_out = self.mlp2(mlp1_out)
        ########
        mlp2_out = self.ln(self.relu(self.linear(mlp2_out)+x))
        ########
        return mlp2_out

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
        return out+x
    
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
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out+x
    


if __name__ == '__main__':
    net = MLPMixer(
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
        is_cls_token=True
        )
    

