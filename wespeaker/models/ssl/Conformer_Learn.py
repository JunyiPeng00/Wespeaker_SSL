import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from .Conformer import *
from einops import rearrange, repeat

class Conformer_Spkformer(Conformer):
    def __init__(self, cfg):
        super(Conformer_Spkformer, self).__init__(cfg)

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: Optional[int] = None,
        ret_layer_results: bool = False,
    ):

        # Only Transformer Learnable, while the CNN is fixed
        features = self.feature_extractor(source)

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)
#         print(features.shape)

        x = features
        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=None if output_layer is None else output_layer - 1
        )
        return x, layer_results


class spk_extractor(nn.Module):
    def __init__(self,**kwargs):
        super(spk_extractor, self).__init__()
        ckpt_path = '/mnt/proj3/open-24-5/pengjy_new/HuBERT/fairseq/fairseq_cli/Vox2_S2_Conformer_Vox2_1200K_AUG/checkpoints/checkpoint_best.pt'
        # ckpt_path = '/mnt/proj3/open-24-5/pengjy_new/WavLM/Pretrained_model/WavLM-Base+.pt'
        checkpoint = torch.load(ckpt_path)

        cfg = ConformerConfig(checkpoint['cfg'])
        # cfg.adapter_dim = kwargs['adapter_dim']
        
        self.model = Conformer_Spkformer(cfg)
        # print(self.model)

        self.weights_k = nn.Parameter(data=torch.ones(13),requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13),requires_grad=True)

        self.compression_k = nn.Linear(768,128)
        self.compression_v = nn.Linear(768,128)
        
        self.attention_head = nn.Linear(128,32) # f, h

        # Compression Layer
        self.mlp = nn.Linear(32*128,256)

        #Load Para
        self.loadParameters(checkpoint['model'])

        self.attention_weights_visual = None
        self.xk = None
        self.xv = None

    def forward(self,wav_and_flag):
        
        x = wav_and_flag[0]
        rep, layer_results =  self.model.extract_features(x, output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)

        x_k = torch.sum(x.mul(nn.functional.softmax(self.weights_k)),dim=-1).transpose(1,2) # B,T, F
        x_v = torch.sum(x.mul(nn.functional.softmax(self.weights_v)),dim=-1).transpose(1,2) # B,T, F

        w_k = self.compression_k(x_k) # B,T,f
        w_v = self.compression_v(x_v).unsqueeze(-2) # B,T,f

        self.xk = w_k
        self.xv = w_v.squeeze(-2)

        attention_weights = self.attention_head(w_k) # B, T, H
        self.attention_weights_visual = nn.functional.softmax(attention_weights,dim=1) # B,T,H
        tmp = torch.sum(w_v.mul(nn.functional.softmax(attention_weights,dim=1).unsqueeze(-1)),dim=1) # B T H 1 x B T 1 F-> B T H F -> B H F
        b,h,f = tmp.shape
        pooling_out = tmp.reshape(b,h*f)
        
        out = self.mlp(pooling_out)

        return out

    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            

            if name not in self_state:
                print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);


def MainModel(**kwargs):
    model = spk_extractor(**kwargs)
    return model