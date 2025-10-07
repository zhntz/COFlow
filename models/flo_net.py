import torch
import torch.nn as nn
from .base_model import BaseModel
from .networks import get_norm_layer, init_net
from .transformer import FeatureTransformer, FeatureFlowAttention, split_feature, merge_splits
from . import networks, networks_occ
from .utils import  PositionEmbeddingSine  #feature_add_position,
from .backbone import CNNEncoder,UNETeccoder,RESencoder


class FLO_net(nn.Module):
    def __init__(self,
                 opt,
                 feature_channels=64,
                 num_scales=1
                 ):
        super(FLO_net, self).__init__()
        self.netA = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=[0,1])    ###self.gpu_ids)      opt.input_nc 192
        self.netG_occ = networks_occ.define_G_occ(9, 3, opt.ngf, 'unet_256', opt.norm,         ###  9 -> Base+RES+CAFF
                                                  not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=[0,1])
        # self.netG_occ = networks_occ.define_G_occ(9, 3, opt.ngf, 'unet_256', opt.norm,
        #                                           not opt.no_dropout, opt.init_type, opt.init_gain,self.gpu_ids)
        self.transformer = FeatureTransformer(num_layers=6,
                                              d_model=64,
                                              nhead=1,
                                              attention_type='swin',
                                              ffn_dim_expansion=4)
        self.backbone = CNNEncoder(output_dim=feature_channels,num_output_scales=num_scales)
        self.UNETencoder = UNETeccoder()

    def unet_feature1(self,real):
        features = self.UNETencoder(real)
        return features

    def unet_feature2(self,real):
        features = self.UNETencoder(real)
        return features

    def unet_feature3(self,real):
        features = self.UNETencoder(real)
        return features

    def feature_add_position(swlf,feature0, feature1, attn_splits, feature_channels):
        pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)

        if attn_splits > 1:  # add position in splited window      ### >
            feature0_splits = split_feature(feature0, num_splits=attn_splits)
            feature1_splits = split_feature(feature1, num_splits=attn_splits)
            position = pos_enc(feature0_splits)
            feature0_splits = feature0_splits + position
            feature1_splits = feature1_splits + position
            feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
            feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
        else:
            position = pos_enc(feature0)
            feature0 = feature0 + position
            feature1 = feature1 + position

        return feature0, feature1

    def forward(self,real_A,real_C,mb,attn_num_splits=64,dim=1,feature_channels=64):
        fake_occ1 = self.netG_occ(torch.cat((torch.cat([real_A, real_C], dim=1), mb), 1))


        real_A = self.unet_feature1(real_A)
        real_C = self.unet_feature2(real_C)
        fake_occ = self.unet_feature3(fake_occ1)   ### NOocc

        real_A,real_C = self.feature_add_position(real_A,real_C,attn_num_splits,feature_channels)
        real_A,real_C = self.transformer(real_A,real_C,attn_num_splits)
        # feak_B = self.netA(torch.cat((torch.cat([real_A,real_C],dim=1),fake_occ), 1))
        feak_B = self.netA(torch.cat([real_A, real_C], dim=1))
        return feak_B ,fake_occ1