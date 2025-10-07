import torch
import torch.nn as nn
from .base_model import BaseModel
from .transformer import FeatureTransformer,FeatureFlowAttention
from . import networks
from .utils import feature_add_position
from .backbone import CNNEncoder,UNETeccoder,RESencoder

class FLOWnet(nn.Module):
    def __init__(self,
                 opt,
                 feature_channels=64,
                 num_scales=1
                 ):
        super(FLOWnet, self).__init__()
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=[0,1])    ###self.gpu_ids)
        self.transformer = FeatureTransformer(num_layers=6,
                                              d_model=64,
                                              nhead=1,
                                              attention_type='swin',
                                              ffn_dim_expansion=4)
        self.backbone = CNNEncoder(output_dim=feature_channels,num_output_scales=num_scales)
        self.resnetencoder = RESencoder(input_nc=3)
        self.UNETencoder = UNETeccoder()

    def unet_feature(self,real):
        features = self.UNETencoder(real)
        return features

    def extract_feature(self, real):
        features = self.backbone(real)
        return features

    def resencoder(self,real):
        features = self.resnetencoder(real)
        return features


    def forward(self,real_A,real_C,fake_occ,attn_num_splits=64,dim=1,feature_channels=64):

        ###real_A = self.extract_feature(real_A)
        ###real_C = self.extract_feature(real_C)

        real_A = self.unet_feature(real_A)
        real_C = self.unet_feature(real_C)
        fake_occ = self.unet_feature(fake_occ)

        real_A,real_C = feature_add_position(real_A,real_C,attn_num_splits,feature_channels)
        real_A,real_C = self.transformer(real_A,real_C,attn_num_splits)
        #real_A = self.cnndecoder(real_A)
        #real_C = self.cnndecoder(real_C)
        feak_B = self.netG(torch.cat((torch.cat([real_A,real_C],dim=1),fake_occ), 1))
        #feak_B = self.netG((torch.cat(real_A,fake_occ),1))
        return feak_B