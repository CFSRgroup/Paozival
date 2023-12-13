# @Time :2022/11/11 15:32
# @Author : paozival
import math
import torch
from torch import nn
import torch.nn.functional as F


from feature_map_backbone.aff_net.aff_resnet import BasicBlock,resnet18
from feature_map_backbone.AFF_ResNet import MyAFF_ResNet




def weights_init(m):

    if isinstance(m,nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m,nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Linear):
        nn.init.normal_(m.weight,0.0,0.01)
        if m.bias != None:
            nn.init.zeros_(m.bias)



def freeze_modules(m):
    for param in m.parameters():
        if param.requires_grad:
            param.requires_grad=False

def unfreeze_modules(m):
    for param in m.parameters():
        if not param.requires_grad:
            param.requires_grad=True





class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class Channel_Wise_TransformerEncoder_Backbone(nn.Module):
    def __init__(self,is_eeg:bool,args):
        super(Channel_Wise_TransformerEncoder_Backbone, self).__init__()
        if is_eeg:
            self.channel_feat_dim=args.eeg_channel_feat_dim
            self.num_channel=args.EEG_channel
        else:
            self.channel_feat_dim=1
            self.num_channel=args.peri_feat_dim

        if args.positional_encoding:
            self.positional_encoding=PositionalEncoding(d_model=args.backbone_hidden,dropout=0)
        else:
            self.positional_encoding=None
        self.attn_layer=nn.TransformerEncoderLayer(d_model=args.backbone_hidden,nhead=8,dim_feedforward=1024,batch_first=True)
        self.encoder=nn.TransformerEncoder(encoder_layer=self.attn_layer,num_layers=args.num_layer)
        self.dim_up=nn.Linear(self.channel_feat_dim,args.backbone_hidden)

        if args.backbone_switch==[1,1,1] :
            # self.dim_down=nn.Linear(self.num_channel*args.hidden,args.hidden)
            if args.fusion_method=='HF_ICMA':
                self.dim_down=None
            
            

        else: # single modality , no fusing
            self.dim_down=nn.Linear(self.num_channel*args.backbone_hidden,args.backbone_hidden)

        # self.dim_down=nn.Linear(self.num_channel*self.hidden,self.hidden)
        # self.pooling=nn.AdaptiveAvgPool1d(1)

    def forward(self,x):
        # (bs,62,12) / (bs,33,1)
        assert len(x.shape)==3 and x.shape[1]==self.num_channel and x.shape[2]==self.channel_feat_dim
        x=self.dim_up(x) # (bs,62,fusion_hidden) / (bs,33,fusion_hidden)
        if self.positional_encoding!=None:
            x=self.positional_encoding(x)
        x=self.encoder(x) # (bs,62,fusion_hidden) / (bs,33,fusion_hidden)
        if self.dim_down!=None:
            x=self.dim_down(x.view(x.shape[0],-1)) # (bs,fusion_hidden)
 
        return x

    def init_weights(self):
        self.dim_up.reset_parameters()
        if self.dim_down!=None:
            self.dim_down.reset_parameters()
        # self.dim_down.reset_parameters()
        # encoder_layer initialize
        for layer in self.encoder.layers:
            layer.self_attn._reset_parameters()
            layer.linear1.reset_parameters()
            layer.linear2.reset_parameters()
            layer.norm1.reset_parameters()
            layer.norm2.reset_parameters()



class HF_ICMA(nn.Module):
    def __init__(self,args):
        super(HF_ICMA, self).__init__()

        if args.feature_map_backbone_type=='resnext':
            self.eeg_feature_map_dim_change=nn.Linear(2048,args.fusion_hidden)
        elif args.feature_map_backbone_type=='convnext':
            self.eeg_feature_map_dim_change=nn.Linear(768,args.fusion_hidden)
        else:
            self.eeg_feature_map_dim_change=nn.Linear(512,args.fusion_hidden) #   !!!!! exactly 512 !!!!!
        self.eeg_feature_dim_change=nn.Linear(args.backbone_hidden,args.fusion_hidden)
        self.peri_dim_change=nn.Linear(args.backbone_hidden,args.fusion_hidden)

        self.eeg_feature_map_attn=nn.MultiheadAttention(embed_dim=args.fusion_hidden,num_heads=8,batch_first=True)
        self.eeg_feature_attn1=nn.MultiheadAttention(embed_dim=args.fusion_hidden,num_heads=8,batch_first=True)
        self.eeg_feature_attn2=nn.MultiheadAttention(embed_dim=args.fusion_hidden,num_heads=8,batch_first=True)
        self.peri_attn=nn.MultiheadAttention(embed_dim=args.fusion_hidden,num_heads=8,batch_first=True)
        if args.feature_map_backbone_type=='convnext':
            self.eeg_feature_map_dim_down=nn.Linear(16*args.fusion_hidden,args.fusion_hidden)
        else:
            self.eeg_feature_map_dim_down=nn.Linear(4*args.fusion_hidden,args.fusion_hidden)
       
        self.eeg_feature_dim_down=nn.Linear(args.EEG_channel*args.fusion_hidden,args.fusion_hidden)
        self.peri_dim_down=nn.Linear(args.peri_feat_dim*args.fusion_hidden,args.fusion_hidden)

    def forward(self,eeg_map,eeg,peri):
        # eeg_map (bs,512,2,2)
        # eeg (bs,62,fusion_hidden)
        # peri (bs,31,fusion_hidden)
        eeg_map=eeg_map.view(eeg_map.shape[0],eeg_map.shape[1],-1)
        eeg_map=eeg_map.transpose(1,2) # (bs,4,512)

        eeg_map=self.eeg_feature_map_dim_change(eeg_map) # (bs,4,h)
        eeg=self.eeg_feature_dim_change(eeg) # (bs,62,h)
        peri=self.peri_dim_change(peri) # (bs,31,h)

        eeg_map_attn_out,_=self.eeg_feature_map_attn(eeg_map,eeg,eeg) # (bs,4,h)

        eeg_attn1_out,_=self.eeg_feature_attn1(eeg,eeg_map,eeg_map) # (bs,62,h)

        eeg_attn2_out,cross_modal_attention_value=self.eeg_feature_attn2(eeg_attn1_out,peri,peri) # (bs,62,h)

        # residual
        eeg_attn2_out=eeg_attn2_out+eeg_attn1_out

        peri_attn_out,_=self.peri_attn(peri,eeg_attn1_out,eeg_attn1_out) # (bs,31,h)
        # residual
        peri_attn_out=peri_attn_out+peri

        eeg_map_attn_out=eeg_map_attn_out.contiguous().view(eeg_map_attn_out.shape[0],-1)

        eeg_map_out=self.eeg_feature_map_dim_down(eeg_map_attn_out) # (bs,h)

        eeg_attn2_out=eeg_attn2_out.contiguous().view(eeg_attn2_out.shape[0],-1)
        eeg_out=self.eeg_feature_dim_down(eeg_attn2_out) # (bs,h)

        peri_attn_out=peri_attn_out.contiguous().view(peri_attn_out.shape[0],-1)
        peri_out=self.peri_dim_down(peri_attn_out) # (bs,h)

        fused_tensor=torch.cat([eeg_map_out,eeg_out,peri_out],dim=-1)

        return fused_tensor,cross_modal_attention_value
       

    def init_weights(self):
        self.eeg_feature_map_attn._reset_parameters()
        self.eeg_feature_attn1._reset_parameters()
        self.eeg_feature_attn2._reset_parameters()
        self.peri_attn._reset_parameters()

        self.eeg_feature_map_dim_change.reset_parameters()
        self.eeg_feature_dim_change.reset_parameters()
        self.peri_dim_change.reset_parameters()

        self.eeg_feature_map_dim_down.reset_parameters()
        self.eeg_feature_dim_down.reset_parameters()
        self.peri_dim_down.reset_parameters()


class Fusion_Model(nn.Module):
    def __init__(self,args):
        super(Fusion_Model, self).__init__()
        self.args=args
        hidden=args.hidden
        self.switch=args.backbone_switch
        self.fusion=args.fusion_method
        if args.backbone_switch!=[1,1,1]:
            self.fusion=None
        else:
            self.fusion=args.fusion_method

        self.device=torch.device(args.device)

        

        if self.switch[0]==1:

            if args.feature_map_backbone_type=='aff_resnet':
                if self.switch==[1,1,1]:
                    aff_resnet=MyAFF_ResNet(block=BasicBlock,layers=[2,2,2,2],fuse_type='AFF',small_input=True)
                else:
                    aff_resnet=resnet18(fuse_type='AFF',small_input=True,num_classes=args.backbone_hidden)
                aff_resnet.conv1=nn.Conv2d(args.frequency_band, 64, kernel_size=3, stride=1,padding=1, bias=False)
                self.feature_map_backbone=aff_resnet

    


        
        if self.switch[1]==1:
            if args.backbone_type=='channel_wise_transformer':
                self.eeg_backbone=Channel_Wise_TransformerEncoder_Backbone(is_eeg=True,args=args)
        


        if self.switch[2]==1:
            

            if args.backbone_type=='channel_wise_transformer':
                self.peri_backbone=Channel_Wise_TransformerEncoder_Backbone(is_eeg=False, args=args)

      

      
        if self.switch==[1,1,1]:

            if self.fusion=='HF_ICMA':
                # if args.backbone_type!='channel_wise_transformer' or args.backbone_type!='Conformer':
                #     assert False
                self.HF_ICMA=HF_ICMA(args)
                merge_dim=args.fusion_hidden*3


        elif self.switch==[1,1,0]:
            merge_dim=sum(self.switch)*args.backbone_hidden
        elif self.switch==[0,0,1]:
            merge_dim=sum(self.switch)*args.backbone_hidden
        

        self.merge_dim=merge_dim

        self.stage2_module=nn.Sequential(
            nn.Linear(merge_dim,hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden,hidden),
        )


        self.proj_head=nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), \
                                     nn.Linear(hidden, hidden))
        self.trial_head=nn.Sequential(nn.Linear(merge_dim, hidden), nn.ReLU(inplace=True), \
                                      nn.Linear(hidden, args.trial_cnt))
        
        if args.dataset=='DEAP':
            self.emo_head=nn.Sequential(nn.Linear(hidden,hidden),nn.Linear(hidden,2),\
                                        nn.Sigmoid())
        else:
            self.emo_head=nn.Sequential(nn.Linear(hidden, hidden), \
                                        # nn.ReLU(inplace=True), \
                                    nn.Linear(hidden, args.emo_categories))
            
        

    def forward_eeg(self, data):
        eeg_map=data['eeg_feature_map']
        eeg=data['eeg_feature']
        eeg_map_out=self.feature_map_backbone(eeg_map) # (bs,hidden)
        eeg_out=self.eeg_backbone(eeg)
        x_s1=torch.cat([eeg_map_out,eeg_out],dim=1) # shape (bs, hidden * 2 )
        z_t=self.trial_head(x_s1)

        x_s2=self.stage2_module(x_s1) # (bs,hidden)
        z_c=self.proj_head(x_s2)
        e=self.emo_head(x_s2)
        return x_s1,z_t,x_s2,z_c,e

    def forward_peri(self, data):
        peri=data['peri_feature']
        x_s1=self.peri_backbone(peri)
        z_t=self.trial_head(x_s1)
        x_s2=self.stage2_module(x_s1) # (bs,hidden)
        z_c=self.proj_head(x_s2)
        e=self.emo_head(x_s2)
        return x_s1,z_t,x_s2,z_c,e

    def forward_eeg_peri(self, data):
        eeg_map=data['eeg_feature_map'] 
        eeg=data['eeg_feature']
        peri=data['peri_feature']

        eeg_map_out=self.feature_map_backbone(eeg_map) # (512,2,2)
        eeg_out=self.eeg_backbone(eeg)  # (62/32,backbone_hidden)
        peri_out=self.peri_backbone(peri) # (peri_dim,backbone_hidden)

        # print('check eeg_map')



        if self.fusion=='HF_ICMA':
            
            fused_x,cross_modal_attention_value=self.HF_ICMA(eeg_map_out,eeg_out,peri_out)
            x_s1=fused_x

   
        
        
        z_t=self.trial_head(x_s1)
        x_s2=self.stage2_module(x_s1)
        z_c=self.proj_head(x_s2)
        e=self.emo_head(x_s2)

        if self.fusion=='HF_ICMA':
            # return cross_modal_attention_value,x_s1,z_t,x_s2,z_c,e
            return x_s1,z_t,x_s2,z_c,e


    

    def freeze_stage1(self):

        if self.switch==[1,1,1]:

            if self.fusion=='HF_ICMA':
                for param in self.HF_ICMA.parameters():
                    if param.requires_grad:
                        param.requires_grad=False
           
            

        for param in self.trial_head.parameters():
            if param.requires_grad:
                param.requires_grad=False

        if self.switch[0]==1:
            for param in self.feature_map_backbone.parameters():
                if param.requires_grad:
                    param.requires_grad=False
        if self.switch[1]==1:
            for param in self.eeg_backbone.parameters():
                if param.requires_grad:
                    param.requires_grad=False
        if self.switch[2]==1:
            for param in self.peri_backbone.parameters():
                if param.requires_grad:
                    param.requires_grad=False

    def unfreeze_stage1(self):

        if self.switch==[1,1,1]:

            if self.fusion=='HF_ICMA':
                for param in self.HF_ICMA.parameters():
                    if not param.requires_grad:
                        param.requires_grad=True
          
            

        for param in self.trial_head.parameters():
            if not param.requires_grad:
                param.requires_grad=True

        if self.switch[0]==1:
            for param in self.feature_map_backbone.parameters():
                if not param.requires_grad:
                    param.requires_grad=True
        if self.switch[1]==1:
            for param in self.eeg_backbone.parameters():
                if not param.requires_grad:
                    param.requires_grad=True

        if self.switch[2]==1:
            for param in self.peri_backbone.parameters():
                if not param.requires_grad:
                    param.requires_grad=True

    def init_stage2_module(self):
        for m in self.stage2_module.modules():
            if isinstance(m,nn.Linear):
                m.reset_parameters()
            elif isinstance(m,nn.BatchNorm1d):
                m.reset_parameters()
        for m in self.proj_head.modules():
            if isinstance(m,nn.Linear):
                m.reset_parameters()
        for m in self.emo_head.modules():
            if isinstance(m,nn.Linear):
                m.reset_parameters()

    def init_trial_head(self):
        for m in self.trial_head.modules():
            if isinstance(m,nn.Linear):
                m.reset_parameters()


    def init_stage1_encoder(self):

        # init spatial_map backbone
        if self.switch[0]==1:
            for m in self.feature_map_backbone.modules():
                weights_init(m)
        # init eeg_backbone
        if self.switch[1]==1:
            self.eeg_backbone.init_weights()
        # init eye backbone
        if self.switch[2]==1:
            self.peri_backbone.init_weights()


    def init_all(self):
        if self.switch==[1,1,1]:

            if self.fusion=='HF_ICMA':
                self.HF_ICMA.init_weights()
            
           
            
        self.init_stage1_encoder()
        self.init_stage2_module()
        self.init_trial_head()

