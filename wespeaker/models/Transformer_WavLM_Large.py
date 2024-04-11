import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from wespeaker.models.ssl.WavLM_Large import *
from einops import rearrange, repeat
from torch.nn.utils import remove_weight_norm
from wespeaker.models.ssl.modules import GradMultiply
from wespeaker.models.ssl_backend import *



class WavLM_Large_MHFA(nn.Module):
    def __init__(self,model_path, pooling, head_nb, embed_dim, group=1, cnn_scale=1.0,layer_drop=0.05,frozen=False):
        super(WavLM_Large_MHFA, self).__init__()
        checkpoint = torch.load(model_path)
        checkpoint['cfg']['encoder_layerdrop']=layer_drop
        checkpoint['cfg']['feature_grad_mult']=cnn_scale
        cfg = WavLMConfig(checkpoint['cfg'])
        print('During the training, SSL is kept frozen:{}\n'.format(frozen))
        self.model = WavLM(cfg)
        self.loadParameters(checkpoint['model'])
        self.frozen = frozen
        if pooling == 'MHFA':
            self.back_end = MHFA(inputs_dim=1024, head_nb=head_nb,outputs_dim=embed_dim,nb_layer=25)
        elif pooling == 'G_MHFA_Conv2D':
            self.back_end = MHFA_Group_Conv2D(inputs_dim=1024, head_nb=head_nb, outputs_dim=embed_dim, group_nb=group, nb_layer=25)
        self.feature_grad_mult = 0.03

    def forward(self,wav_and_flag):
        
        x = wav_and_flag
        if self.frozen:
            with torch.no_grad():
                rep, layer_results = self.model.extract_features(x[:,:480000], output_layer=25)
        else:
            rep, layer_results = self.model.extract_features(x[:,:480000], output_layer=25)

        layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
        x = torch.stack(layer_reps).transpose(0,-1).transpose(0,1)
        x = GradMultiply.apply(x, self.feature_grad_mult)
        
        spk_embedding = self.back_end(x)
        
        return spk_embedding


    def loadParameters(self, param):

        self_state = self.model.state_dict();
        loaded_state = param

        for name, param in loaded_state.items():
            origname = name;
            
            if name not in self_state:
                # print("%s is not in the model."%origname);
                continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

if __name__ == "__main__":
    from thop import profile
    # from ptflops import get_model_complexity_info
    model_path = '/home/jpeng/ntt/work/Data/pretrained_model/WavLM-Large.pt'
    pooling = 'MHFA'
    embed_dim = 256
    head_nb = 64
    group = 1
    model = WavLM_Large_MHFA(model_path, pooling, head_nb, embed_dim, group,cnn_scale=0.0,layer_drop=0.00)
    flops, params = profile(model.eval(), inputs=(torch.randn(1, 16000*2),))

    print("FLOPS: {} G, Params: {} M".format(flops / 1e9, params / 1e6))