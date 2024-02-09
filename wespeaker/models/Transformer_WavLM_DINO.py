import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from wespeaker.models.ssl.WavLM import *
from einops import rearrange, repeat
from torch.nn.utils import remove_weight_norm
from wespeaker.models.ssl.modules import GradMultiply
from wespeaker.models.ssl.Conformer import *


class MHFA(nn.Module):
    def __init__(self,head_nb=8, inputs_dim=768, compression_dim=128, outputs_dim=256):
        super(MHFA, self).__init__()
        self.weights_k = nn.Parameter(data=torch.ones(13),requires_grad=True)
        self.weights_v = nn.Parameter(data=torch.ones(13),requires_grad=True)
        self.head_nb = head_nb
        self.ins_dim = inputs_dim
        self.cmp_dim = compression_dim
        self.ous_dim = outputs_dim
        self.cmp_linear_k = nn.Linear(self.ins_dim, self.cmp_dim)
        self.cmp_linear_v = nn.Linear(self.ins_dim, self.cmp_dim)
        self.att_head = nn.Linear(self.cmp_dim, self.head_nb)
        self.pooling_fc = nn.Linear(self.head_nb*self.cmp_dim, self.ous_dim)

    def forward(self,x):
        # X shape is [Batch, Dim, Frame_len, Nb_Layer]
        k = torch.sum(x.mul(nn.functional.softmax(self.weights_k,dim=-1)),dim=-1).transpose(1,2)
        v = torch.sum(x.mul(nn.functional.softmax(self.weights_v,dim=-1)),dim=-1).transpose(1,2)

        k = self.cmp_linear_k(k)
        v = self.cmp_linear_v(v)

        att_k = self.att_head(k)
        v = v.unsqueeze(-2)
        pooling_outs = torch.sum(v.mul(nn.functional.softmax(att_k,dim=1).unsqueeze(-1)),dim=1)
        b,h,f = pooling_outs.shape
        pooling_outs = pooling_outs.reshape(b,-1)
        outs = self.pooling_fc(pooling_outs)
        return outs

class WavLM_Base_MHFA(nn.Module):
    def __init__(self,model_path,head_nb,embed_dim):
        super(WavLM_Base_MHFA, self).__init__()
        # checkpoint = torch.load('/mnt/proj3/open-24-5/pengjy_new/WavLM/Pretrained_model/WavLM-Base+.pt')
        # set_seed(42)
        # checkpoint = torch.load(model_path)

        # checkpoint['cfg']['encoder_layerdrop']=0.0

        cfg = ConformerConfig()
        self.model = Conformer(cfg)
        # self.model = remove_weight_norm(self.model)
        # self.loadParameters(checkpoint['model'])
        self.back_end = MHFA(head_nb=head_nb,outputs_dim=embed_dim,inputs_dim=cfg.encoder_embed_dim)
        # self.feature_grad_mult = 0.1

    def forward(self,wav_and_flag):
        
        x = x.permute(0, 2, 1)  # (B,T,F) -> (B,F,T)

        # out1 = self.layer1(x)        # with torch.no_grad():
        rep, layer_results = self.model.extract_features(x, output_layer=13)
        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
                
        spk_embedding = self.back_end(x)
        
        return spk_embedding
