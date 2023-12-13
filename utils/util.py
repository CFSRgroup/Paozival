# @Time :2022/10/26 18:11
# @Author : paozival

import argparse

import numpy as np
import torch
import queue
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler




class Dataset_SD_DEAP(Dataset):
    def __init__(self,x_dict,label,args):
        self.args=args
        self.trial_cnt=args.trial_cnt
        self.switch=args.backbone_switch
        # self.len_=trial_cnt*seg_num


        if self.switch[0]==1:
            self.eeg_feature_map=x_dict['eeg_feature_map']
            if type(self.eeg_feature_map) is np.ndarray:
                self.eeg_feature_map=torch.from_numpy(self.eeg_feature_map)
            # self.eeg_feature_map=self.eeg_feature_map.float()
            assert label.shape[0]==self.eeg_feature_map.shape[0]
    
            self.eeg_feature_map = self.eeg_feature_map.float().to(torch.device(args.device))


        if self.switch[1]==1:
            self.eeg_feature=x_dict['eeg_feature']
            if type(self.eeg_feature) is np.ndarray:
                self.eeg_feature=torch.from_numpy(self.eeg_feature)
            
            
            if args.backbone_type=='channel_wise_transformer':
                self.eeg_feature=self.to_channel_wise(self.eeg_feature,args.EEG_channel,args.eeg_channel_feat_dim)
                
            self.eeg_feature=self.eeg_feature.float().to(torch.device(args.device))

        if self.switch[2]==1:
            self.peri_feature=x_dict['peri_feature']
            if type(self.peri_feature) is np.ndarray:
                self.peri_feature=torch.from_numpy(self.peri_feature)
            self.peri_feature=self.peri_feature.float()
            assert label.shape[0]==self.peri_feature.shape[0]

            assert self.peri_feature.shape[-1]==args.peri_feat_dim


          
            if args.backbone_type=='channel_wise_transformer':
                self.peri_feature=self.peri_feature.unsqueeze(-1)
            self.peri_feature = self.peri_feature.float().to(torch.device(args.device))


        self.y=label
        if type(self.y) is np.ndarray:
            self.y=torch.from_numpy(self.y)
        self.y=self.y.float().to(torch.device(args.device))


    def to_channel_wise(self,x,nc,nf):
        assert len(x.shape)==2 and x.shape[-1]==nc*nf
        channel_wise_x=torch.empty(size=(x.shape[0],nc,nf),dtype=torch.float)
        idx=torch.arange(0,nc*nf,nc)
        for i in range(nc):
            channel_wise_x[:,i,:]=x[:,idx]
            idx+=1

        return channel_wise_x

    def __getitem__(self, item):
    
        batch={}
        if self.args.backbone_switch==[1,1,1]:
            batch['eeg_feature_map']=self.eeg_feature_map[item]
            batch['eeg_feature']=self.eeg_feature[item]
            batch['peri_feature']=self.peri_feature[item]
        elif self.args.backbone_switch==[1,1,0]:  # eeg
            batch['eeg_feature_map']=self.eeg_feature_map[item]
            batch['eeg_feature']=self.eeg_feature[item]
        elif self.args.backbone_switch==[0,0,1]: # peri
            batch['peri_feature']=self.peri_feature[item]
        # elif self.args.backbone_switch==[1,0,1]:
        #     batch['eeg_feature_map']=self.eeg_feature_map[item]
        #     batch['peri_feature']=self.peri_feature[item]

        return batch,self.y[item]


    def __len__(self):
        return self.y.shape[0]

class Dataset_SD_SEED(Dataset):
    def __init__(self, x_dict,part,session,args):
        self.args=args
        self.switch=args.backbone_switch
        self.trial_cnt = args.trial_cnt  # 72


        if session==0:
            self.seg_cnt=args.s1_seg_cnt
            self.session_emo_label=args.s1_emo_label
        elif session==1:
            self.seg_cnt=args.s2_seg_cnt
            self.session_emo_label=args.s2_emo_label
        elif session==2:
            self.seg_cnt=args.s3_seg_cnt
            self.session_emo_label=args.s3_emo_label


        if part == 1:
            self.seg_cnt = self.seg_cnt[:((args.trial_cnt//3)//3)*2]
            self.session_emo_label = self.session_emo_label[:((args.trial_cnt//3)//3)*2]
        else:
            self.seg_cnt = self.seg_cnt[((args.trial_cnt//3)//3)*2:]
            self.session_emo_label = self.session_emo_label[((args.trial_cnt//3)//3)*2:]


        if self.switch[0] == 1:
            self.eeg_feature_map = x_dict['eeg_feature_map']
            if type(self.eeg_feature_map) is np.ndarray:
                self.eeg_feature_map = torch.from_numpy(self.eeg_feature_map)
            self.eeg_feature_map = self.eeg_feature_map.float().to(torch.device(args.device))

        if self.switch[1]==1:
            self.eeg_feature = x_dict['eeg_feature']
            if type(self.eeg_feature) is np.ndarray:
                self.eeg_feature = torch.from_numpy(self.eeg_feature)

            
            if args.backbone_type=='channel_wise_transformer':
                self.eeg_feature=self.to_channel_wise(self.eeg_feature,args.EEG_channel,args.eeg_channel_feat_dim)
                
            self.eeg_feature=self.eeg_feature.float().to(torch.device(args.device))

        if self.switch[2] == 1:
            self.peri_feature = x_dict['peri_feature']
            if type(self.peri_feature) is np.ndarray:
                self.peri_feature = torch.from_numpy(self.peri_feature)
            # self.peri_feature = self.peri_feature.float()
            
            if args.backbone_type=='channel_wise_transformer':
                self.peri_feature = self.peri_feature.unsqueeze(-1) # (xxx,22) -> (xxx,22,1)
            self.peri_feature = self.peri_feature.float().to(torch.device(args.device))

       
        self.y = self.get_label()
        self.y = self.y.long().to(torch.device(args.device))

    def get_label(self):
        emo_label = []
        for i in range(len(self.session_emo_label)):
            emo_label.append(torch.full(size=(self.seg_cnt[i],), fill_value=self.session_emo_label[i]))
        emo_label = torch.cat(emo_label, dim=0)
        return emo_label

    def to_channel_wise(self,x,nc,nf):
        # this function is for eeg_feature of eye_feature
        assert len(x.shape)==2 and x.shape[-1]==nc*nf
        channel_wise_x=torch.empty(size=(x.shape[0],nc,nf),dtype=torch.float)
        idx=torch.arange(0,nc*nf,nc)
        for i in range(nc):
            channel_wise_x[:,i,:]=x[:,idx]
            idx+=1
        # for s in range(x.shape[0]):
        #     for i in range(nc):
        #         for j in range(nf):
        #             channel_wise_x[s,i,j]=x[s,i+j*nc]
        return channel_wise_x


    def __getitem__(self, item):
        # if self.switch==[1,1,1]:  # eeg + eye
        #     return self.eeg_feature_map[item], self.eeg_feature[item], self.peri_feature[item], self.y[item]
        # elif self.switch==[1,1,0]:  # eeg
        #     return self.eeg_feature_map[item], self.eeg_feature[item], self.y[item]
        # else:  # eye
        #     return self.peri_feature[item], self.y[item]
        batch={}
        if self.args.backbone_switch==[1,1,1]:
            batch['eeg_feature_map']=self.eeg_feature_map[item]
            batch['eeg_feature']=self.eeg_feature[item]
            batch['peri_feature']=self.peri_feature[item]
        elif self.args.backbone_switch==[1,1,0]:  # eeg
            batch['eeg_feature_map']=self.eeg_feature_map[item]
            batch['eeg_feature']=self.eeg_feature[item]
        elif self.args.backbone_switch==[0,0,1]: # peri
            batch['peri_feature']=self.peri_feature[item]
        return batch,self.y[item]


    def __len__(self):
        return sum(self.seg_cnt)



class Dataset_SI_two_stage_DEAP(Dataset):
    def __init__(self,x_dict,emo_label_one_p,stage,args):

        self.args=args
        self.stage=stage
        if args.split:
            if stage==1:
                self.plen=int(args.ratio*args.seg_cnt)*args.trial_cnt
            elif stage==2:
                self.plen=(args.seg_cnt-int(args.ratio*args.seg_cnt))*args.trial_cnt
        else:
            self.plen=args.seg_cnt*args.trial_cnt
        self.pnum=None

       

        if args.backbone_switch[0] == 1:
            self.eeg_feature_map = x_dict['eeg_feature_map']
            self.pnum=self.eeg_feature_map.shape[0]//self.plen

            if type(self.eeg_feature_map) is np.ndarray:
                self.eeg_feature_map = torch.from_numpy(self.eeg_feature_map)
            self.eeg_feature_map = self.eeg_feature_map.float().to(torch.device(args.device))

        if args.backbone_switch[1] == 1:
            self.eeg_feature = x_dict['eeg_feature']
            if type(self.eeg_feature) is np.ndarray:
                self.eeg_feature = torch.from_numpy(self.eeg_feature)

            if args.backbone_type=='channel_wise_transformer':
                self.eeg_feature=self.to_channel_wise(self.eeg_feature,args.EEG_channel,args.eeg_channel_feat_dim)
            self.eeg_feature=self.eeg_feature.float().to(torch.device(args.device))

        if args.backbone_switch[2] == 1:
            self.peri_feature = x_dict['peri_feature']
            if self.pnum==None:
                self.pnum=self.peri_feature.shape[0]//self.plen
            if type(self.peri_feature) is np.ndarray:
                self.peri_feature = torch.from_numpy(self.peri_feature)
            if args.backbone_type=='channel_wise_transformer':
                self.peri_feature = self.peri_feature.unsqueeze(-1) # (800,55) -> (800,55,1)
            self.peri_feature = self.peri_feature.float().to(torch.device(args.device))


        if type(emo_label_one_p) is np.ndarray:
            emo_label_one_p=torch.from_numpy(emo_label_one_p)
        self.emo_label_one_p=emo_label_one_p # (xx,2)

        self.y = self.get_label() # (xx,3)

        self.y = self.y.long().to(torch.device(args.device))

    def to_channel_wise(self,x,nc,nf):
        # this function is for eeg_feature
        assert len(x.shape)==2 and x.shape[-1]==nc*nf
        channel_wise_x=torch.empty(size=(x.shape[0],nc,nf),dtype=torch.float)
        idx=torch.arange(0,nc*nf,nc)
        for i in range(nc):
            channel_wise_x[:,i,:]=x[:,idx]
            idx+=1
        # for s in range(x.shape[0]):
        #     for i in range(nc):
        #         for j in range(nf):
        #             channel_wise_x[s,i,j]=x[s,i+j*nc]
        return channel_wise_x

    def get_label(self):
        if self.args.split:
            if self.stage==1:
                trial_label_one_p=torch.empty(size=(self.args.trial_cnt,int(self.args.ratio*self.args.seg_cnt)))
            elif self.stage==2:
                trial_label_one_p=torch.empty(size=(self.args.trial_cnt,self.args.seg_cnt-int(self.args.ratio*self.args.seg_cnt)))
        else:
            trial_label_one_p=torch.empty(size=(self.args.trial_cnt,self.args.seg_cnt))

        for t in range(self.args.trial_cnt):
            trial_label_one_p[t]=t
        trial_label_one_p=trial_label_one_p.reshape(-1)
        if self.args.split==True:
            assert self.emo_label_one_p.shape[0]==trial_label_one_p.shape[0] and self.emo_label_one_p.shape[1]==2
        
        label_for_one_p=torch.cat([self.emo_label_one_p.reshape(-1,2),trial_label_one_p.reshape(-1,1)],dim=1)
       
        label=torch.empty(size=(self.pnum*self.plen,3))
        for i in range(self.pnum):
            label[i*self.plen:(i+1)*self.plen,:]=label_for_one_p
      
        return label

    def __getitem__(self, item):
        batch={}
        if self.args.backbone_switch==[1,1,1]:  # eeg + peri
            batch['eeg_feature_map']=self.eeg_feature_map[item]
            batch['eeg_feature']=self.eeg_feature[item]
            batch['peri_feature']=self.peri_feature[item]
            # batch['y']=self.y[item]
            # return self.eeg_feature_map[item], self.eeg_feature[item], self.peri_feature[item], self.y[item]

        elif self.args.backbone_switch==[1,1,0]:  # eeg
            batch['eeg_feature_map']=self.eeg_feature_map[item]
            batch['eeg_feature']=self.eeg_feature[item]
            # batch['y']=self.y[item]
            # return self.eeg_feature_map[item], self.eeg_feature[item], self.y[item]
        elif self.args.backbone_switch==[0,0,1]:  # peri
            # return self.peri_feature[item], self.y[item]
            batch['peri_feature']=self.peri_feature[item]
            # batch['y']=self.y[item]
        return batch,self.y[item]


    def __len__(self):
        return self.pnum*self.plen


class Dataset_SI_two_stage_SEED(Dataset):
    def __init__(self,x_dict,stage,args):

        self.args=args

        if args.split:
            if stage==1:
                s1_seg_cnt=[int(args.ratio*seg) for seg in args.s1_seg_cnt]
                s2_seg_cnt=[int(args.ratio*seg) for seg in args.s2_seg_cnt]
                s3_seg_cnt=[int(args.ratio*seg) for seg in args.s3_seg_cnt]

            elif stage==2:
                s1_seg_cnt=[seg-int(args.ratio*seg) for seg in args.s1_seg_cnt]
                s2_seg_cnt=[seg-int(args.ratio*seg) for seg in args.s2_seg_cnt]
                s3_seg_cnt=[seg-int(args.ratio*seg) for seg in args.s3_seg_cnt]
            self.seg_cnt=s1_seg_cnt+s2_seg_cnt+s3_seg_cnt


        else:
            self.seg_cnt=args.s1_seg_cnt+args.s2_seg_cnt+args.s3_seg_cnt


        self.emo_label=args.s1_emo_label+args.s2_emo_label+args.s3_emo_label
        self.plen=sum(self.seg_cnt)
        self.pnum=None

       

        if args.backbone_switch[0] == 1:
            self.eeg_feature_map = x_dict['eeg_feature_map']
            self.pnum=self.eeg_feature_map.shape[0]//self.plen
            if type(self.eeg_feature_map) is np.ndarray:
                self.eeg_feature_map = torch.from_numpy(self.eeg_feature_map)
            self.eeg_feature_map = self.eeg_feature_map.float().to(torch.device(args.device))

        if args.backbone_switch[1] == 1:
            self.eeg_feature = x_dict['eeg_feature']
            if type(self.eeg_feature) is np.ndarray:
                self.eeg_feature = torch.from_numpy(self.eeg_feature)

            if args.backbone_type=='channel_wise_transformer' :
                self.eeg_feature=self.to_channel_wise(self.eeg_feature,args.EEG_channel,args.eeg_channel_feat_dim)
            self.eeg_feature=self.eeg_feature.float().to(torch.device(args.device))

        if args.backbone_switch[2] == 1:
            self.peri_feature = x_dict['peri_feature']
            if self.pnum==None:
                self.pnum=self.peri_feature.shape[0]//self.plen
            if type(self.peri_feature) is np.ndarray:
                self.peri_feature = torch.from_numpy(self.peri_feature)
            if args.backbone_type=='channel_wise_transformer' :
                self.peri_feature = self.peri_feature.unsqueeze(-1) # (2505,22) -> (2505,22,1)
            self.peri_feature = self.peri_feature.float().to(torch.device(args.device))


        self.y = self.get_label()

        self.y = self.y.long().to(torch.device(args.device))

    def to_channel_wise(self,x,nc,nf):
        # this function is for eeg_feature
        assert len(x.shape)==2 and x.shape[-1]==nc*nf
        channel_wise_x=torch.empty(size=(x.shape[0],nc,nf),dtype=torch.float)
        idx=torch.arange(0,nc*nf,nc)
        for i in range(nc):
            channel_wise_x[:,i,:]=x[:,idx]
            idx+=1
        # for s in range(x.shape[0]):
        #     for i in range(nc):
        #         for j in range(nf):
        #             channel_wise_x[s,i,j]=x[s,i+j*nc]
        return channel_wise_x

    def get_label(self):
        emo_label_for_one_p=[]
        trial_label_for_one_p=[]
        trial_idx=0
        for i in range(len(self.seg_cnt)):
            emo_label_for_one_p.append(torch.full(size=(self.seg_cnt[i],),fill_value=self.emo_label[i]))
            trial_label_for_one_p.append(torch.full(size=(self.seg_cnt[i],),fill_value=trial_idx))
            trial_idx+=1
        emo_label_for_one_p=torch.cat(emo_label_for_one_p,dim=0)
        trial_label_for_one_p=torch.cat(trial_label_for_one_p,dim=0)
        label_for_one_p=torch.cat([emo_label_for_one_p.reshape(-1,1),trial_label_for_one_p.reshape(-1,1)],dim=1)
        # label_for_one_p=label_for_one_p.numpy()
        # print('check label for one p')
        label=torch.empty(size=( self.pnum*self.plen,2))
        for i in range(self.pnum):
            label[i*self.plen:(i+1)*self.plen,:]=label_for_one_p
        return label

    def __getitem__(self, item):
        batch={}
        if self.args.backbone_switch==[1,1,1]:  # eeg + peri
            batch['eeg_feature_map']=self.eeg_feature_map[item]
            batch['eeg_feature']=self.eeg_feature[item]
            batch['peri_feature']=self.peri_feature[item]
            # batch['y']=self.y[item]
            # return self.eeg_feature_map[item], self.eeg_feature[item], self.peri_feature[item], self.y[item]

        elif self.args.backbone_switch==[1,1,0]:  # eeg
            batch['eeg_feature_map']=self.eeg_feature_map[item]
            batch['eeg_feature']=self.eeg_feature[item]
            # batch['y']=self.y[item]
            # return self.eeg_feature_map[item], self.eeg_feature[item], self.y[item]
        else:  # peri
            # return self.peri_feature[item], self.y[item]
            batch['peri_feature']=self.peri_feature[item]
            # batch['y']=self.y[item]
        return batch,self.y[item]


    def __len__(self):
        return self.pnum*self.plen




class MySampler_DEAP(Sampler):
    def __init__(self,args,subject_number):

        self.args=args
        self.subject_number=subject_number
        if args.split:
            self.seg_cnt=args.seg_cnt-int(args.ratio*args.seg_cnt)
        else:
            self.seg_cnt=args.seg_cnt

    def __iter__(self):
        # p_cnt=self.len_//self.seg_num  # 40
        indices = []
        tot=0
        for p in range(self.subject_number):

            for i in range(self.args.trial_cnt):
                # if i==0:
                #     idx=torch.randint(low=0,high=self.seg_nums[i],size=(1,))
                # else:
                #     idx=torch.randint(low=sum(self.seg_nums[:i]),\
                #                       high=sum(self.seg_nums[:i])+self.seg_nums[i],size=(1,))
                idx = torch.randint(low=i*self.seg_cnt+tot, \
                                    high=(i+1)*self.seg_cnt+tot, size=(1,))
                indices.append(idx)
            tot+=self.args.trial_cnt*self.seg_cnt

        indices = torch.cat(indices, dim=0)
        return iter(indices)

class MySampler_SEED(Sampler):
    def __init__(self,args,subject_number):

        self.args=args
        self.subject_number=subject_number
        # self.trial_cnt = args.trial_cnt

        if args.split:
            s1_seg_cnt=[seg-int(args.ratio*seg) for seg in args.s1_seg_cnt]
            s2_seg_cnt=[seg-int(args.ratio*seg) for seg in args.s2_seg_cnt]
            s3_seg_cnt=[seg-int(args.ratio*seg) for seg in args.s3_seg_cnt]
            self.seg_cnt=s1_seg_cnt+s2_seg_cnt+s3_seg_cnt
        else:
            self.seg_cnt=args.s1_seg_cnt+args.s2_seg_cnt+args.s3_seg_cnt



    def __iter__(self):
        # p_cnt=self.len_//self.seg_num  # 40
        indices = []
        tot=0
        for p in range(self.subject_number):

            for i in range(self.args.trial_cnt):
                # if i==0:
                #     idx=torch.randint(low=0,high=self.seg_nums[i],size=(1,))
                # else:
                #     idx=torch.randint(low=sum(self.seg_nums[:i]),\
                #                       high=sum(self.seg_nums[:i])+self.seg_nums[i],size=(1,))
                idx = torch.randint(low=sum(self.seg_cnt[:i])+tot, \
                                    high=sum(self.seg_cnt[:i])+tot + self.seg_cnt[i], size=(1,))
                indices.append(idx)
            tot+=sum(self.seg_cnt)

        indices = torch.cat(indices, dim=0)
        return iter(indices)

