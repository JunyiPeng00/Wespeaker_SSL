import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from wespeaker.models.ssl.whisper_model import WhisperModelNoDecode
from transformers import WhisperFeatureExtractor
from einops import rearrange, repeat
from torch.nn.utils import remove_weight_norm
from wespeaker.models.ssl.modules import GradMultiply
from wespeaker.models.ssl_backend import *

SAMPLE_RATE = 16000


class Whisper_MHFA(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim, group):
        super(Whisper_MHFA, self).__init__()
        ckpt = model_path # openai/whisper-small
        self.extracter = WhisperFeatureExtractor.from_pretrained(ckpt)
        self.model = WhisperModelNoDecode.from_pretrained(ckpt)

        print(pooling)

        if pooling == 'MHFA':
            self.back_end = MHFA(head_nb=head_nb,outputs_dim=embed_dim)
        elif pooling == 'G_MHFA':
            self.back_end = MHFA_Group(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'QKV':
            self.back_end = MHFA_Dotproduct(compression_dim=256, outputs_dim=embed_dim)
        elif pooling == 'G_MHFA_MQSKMV':
            self.back_end = MHFA_Group_MQ_SK_MV(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'G_MHFA_MQMKSV':
            self.back_end = MHFA_Group_MQ_MK_SV(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'G_MHFA_Conv2D':
            self.back_end = MHFA_Group_Conv2D(head_nb=head_nb, outputs_dim=embed_dim, group_nb=group)
        elif pooling == 'MHFA_Context':
            self.back_end = MHFA_context(head_nb=head_nb,outputs_dim=embed_dim)
        self.feature_grad_mult = 0.05

    def forward(self,wavs):
        device = wavs[0].device
        wavs = [wav.detach().cpu().numpy() for wav in wavs]

        input_values = self.extracter(
            wavs,
            return_tensors="pt",
            padding=True,
            sampling_rate=SAMPLE_RATE,
        ).to(device)
        # print(input_values.input_features.shape)
        output_values = self.model(input_values.input_features, output_hidden_states=True)
        # output_values = self.model(wavs, output_hidden_states=True)

        layer_reps = [x for x in output_values]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
                
        x = GradMultiply.apply(x, self.feature_grad_mult)
        
        spk_embedding = self.back_end(x)
        
        return spk_embedding
        

if __name__=='__main__':
    model = Whisper_MHFA('openai/whisper-small','MHFA',64,256,1)
    x = torch.randn(5, 300*160)
    out = model(x)
    print(out.shape)
