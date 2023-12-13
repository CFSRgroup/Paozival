# SI_two_stage
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import  DataLoader
from torchvision.transforms.functional import normalize
from torchvision.models import vgg16
# utils
from utils.util import Dataset_SI_two_stage_DEAP,Dataset_SI_two_stage_SEED,MySampler_DEAP,MySampler_SEED
# from utils.distro_vis import t_sne,distribution_visualization
from utils.metrics import CenterLoss, loss_cl, accuracy
from common.utils.meter import AverageMeter, ProgressMeter
from common.utils.data import ForeverDataIterator
from common.utils.logger import CompleteLogger
# sklearn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# others
import os
import warnings
import argparse
from scipy.io import loadmat
import random
import shutil
# import copy
import time
import numpy as np
import pandas as pd
from fusion_framework import Fusion_Model,freeze_modules,weights_init



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_filename(data_path):
    datanames= os.listdir(path=data_path)
    filenames=[]
    for dataname in datanames:
        if os.path.splitext(dataname)[1]=='.mat':
            filenames.append(dataname)
    return filenames

def load_data(filename:str,args:argparse.Namespace):
    
    if args.dataset=='DEAP':
        path='../DEAP_features/'
    else:
        path='../'+args.dataset+'_features/4s_mat/'
    p_dataset=loadmat(path+filename)
    eeg_feature_map=p_dataset['eeg_allband_feature_map'] 
  

    eeg_en_stat=p_dataset['eeg_en_stat'] # (2505,434)
    if args.dataset=='DEAP':
        peri_feature=p_dataset['peri_feature']
    else:
        peri_feature=p_dataset['eye_feature']
    eeg_feature=eeg_en_stat

 
    if args.dataset=='SEED_IV' or args.dataset=='SEED_V':
        peri_feature=peri_feature[:,:args.peri_feat_dim]
    
    return eeg_feature_map,eeg_feature,peri_feature

def split_X_SEED(X:np.ndarray, args):
    '''

    :param X: one modality feature set for one participant
    :param args:
    :return:
    '''
    ratio=args.ratio 
    trial_cnt=args.trial_cnt

    # s1_seg_cnt=args.s1_seg_cnt
    # s2_seg_cnt=args.s2_seg_cnt
    # s3_seg_cnt=args.s3_seg_cnt

    seg_cnt=args.s1_seg_cnt+args.s2_seg_cnt+args.s3_seg_cnt
    X1=[]
    X2=[]
    tot=0
    for i in range(len(seg_cnt)):
        sample_num1=int(ratio*seg_cnt[i]) # part1 sample num
        sample_num2=int((1-ratio)*seg_cnt[i]) # part2 sample num
        if len(X.shape)>2: # spatial_map
            X_trial=X[tot:tot+seg_cnt[i],:,:,:]
            # if type(X_trial) is np.ndarray:
            #     X_trial=torch.from_numpy(X_trial)
            X1.append(X_trial[:sample_num1,:,:,:])
            X2.append(X_trial[sample_num1:,:,:,:])
        else:
            X_trial=X[tot:tot+seg_cnt[i],:]
            # if type(X_trial) is np.ndarray:
            #     X_trial=torch.from_numpy(X_trial)

            X1.append(X_trial[:sample_num1,:])
            X2.append(X_trial[sample_num1:,:])
        tot+=seg_cnt[i]

    # X1=torch.cat(X1,dim=0)
    # X2=torch.cat(X2,dim=0)
    X1=np.concatenate(X1,axis=0)
    X2=np.concatenate(X2,axis=0)

    return X1,X2

def split_X_DEAP(X:np.ndarray,args):
    ratio=args.ratio 
    trial_cnt=args.trial_cnt
    seg_cnt=args.seg_cnt

    sample_num1=int(ratio*seg_cnt)
    sample_num2=seg_cnt-sample_num1

    if len(X.shape)>2: # eeg_map

        X1=np.empty(shape=(trial_cnt*sample_num1,X.shape[1],X.shape[2],X.shape[3]))
        X2=np.empty(shape=(trial_cnt*sample_num2,X.shape[1],X.shape[2],X.shape[3]))
        X=X.reshape(trial_cnt,seg_cnt,X.shape[1],X.shape[2],X.shape[3]) # (800,5,32,32)->(40,20,5,32,32)
    else: # eeg statistics or peri feature
        # Xtr=torch.randn(size=(trial_cnt*sample_num1,X.shape[-1]),dtype=torch.float)
        X1=np.empty(shape=(trial_cnt*sample_num1,X.shape[-1]))
        X2=np.empty(shape=(trial_cnt*sample_num2,X.shape[-1]))
        
        X=X.reshape(trial_cnt,seg_cnt,X.shape[-1])  # (800,xx) -> (40,20,xx)
    for t in range(trial_cnt):
        if len(X.shape)>2: # eeg_map
            
            X1[t*sample_num1:(t+1)*sample_num1]=X[t,:sample_num1]
            X2[t*sample_num2:(t+1)*sample_num2]=X[t,sample_num1:]
            
        else: # eeg statistics or peri feature
            X1[t*sample_num1:(t+1)*sample_num1]=X[t,:sample_num1]
            X2[t*sample_num2:(t+1)*sample_num2]=X[t,sample_num1:]
           
    return X1,X2
    # return type: torch.Tensor

def split_y_DEAP(y:np.ndarray,args):
    ratio=args.ratio
    trial_cnt=args.trial_cnt
    seg_cnt=args.seg_cnt

    sample_num1=int(ratio*seg_cnt)
    sample_num2=seg_cnt-sample_num1

    y=y.reshape(trial_cnt,seg_cnt,2)
    y1=np.empty(shape=(trial_cnt*sample_num1,2))
    y2=np.empty(shape=(trial_cnt*sample_num2,2))
    # ytr=torch.zeros(size=(trial_cnt*sample_num1,))
    # ystage2=torch.zeros(size=(trial_cnt*stage2_sample_num,))
    # yte=torch.zeros(size=(trial_cnt*sample_num2,))

    for t in range(trial_cnt):
        y1[t*sample_num1:(t+1)*sample_num1,:]=y[t,:sample_num1,:]
        y2[t*sample_num2:(t+1)*sample_num2,:]=y[t,sample_num1:,:]
        # ystage2[t*stage2_sample_num:(t+1)*stage2_sample_num]=y[t,stage1_sample_num:stage1_sample_num+stage2_sample_num]

    return y1,y2

def data_normalize(train_data,test_data,args=None):
    if len(train_data.shape)>2: # feature_map
        if type(train_data) is np.ndarray:
            train_data=torch.from_numpy(train_data)
        if type(test_data) is np.ndarray:
            test_data=torch.from_numpy(test_data)
        mean_,std_=train_data.mean([2,3]),train_data.std([2,3])
        mean_=mean_.mean()
        std_=std_.mean()
        train_data=normalize(train_data,mean_,std_)
        test_data=normalize(test_data,mean_,std_)
    else:
        # eeg_feature or peri_feature
        scaler=StandardScaler()
        train_data=scaler.fit_transform(train_data)
        test_data=scaler.transform(test_data)
    return train_data,test_data

def stage1_training(loader, model: Fusion_Model, center_criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    center_losses = AverageMeter('CenterLoss', ':.4f')
    trial_losses = AverageMeter('TrialLoss', ':.4f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, data_time, trial_losses, center_losses],
        prefix="Epoch:[{}]".format(epoch)
    )
    model.train()
    device = torch.device(args.device)
    end = time.time()

    running_accuracy = 0
    for b, batch in enumerate(loader):
        batch_data,label=batch[:]
        data_time.update(time.time()-end)
        if args.backbone_switch==[1,1,1]:
            
            x_s1,z_t,x_s2,z_c,e=model.forward_eeg_peri(batch_data)

        elif args.backbone_switch==[1,1,0]:
            x_s1,z_t,x_s2,z_c,e=model.forward_eeg(batch_data)

        elif args.backbone_switch==[0,0,1]:
            x_s1,z_t,x_s2,z_c,e=model.forward_peri(batch_data)


        trial_loss = F.cross_entropy(z_t, label[:,-1])
        center_loss_tmp = center_criterion(label[:, -1], x_s1)
        center_loss = center_loss_tmp.squeeze()


        running_accuracy += accuracy(z_t, label[:,-1])

        trial_losses.update(trial_loss.item(), z_t.shape[0])
        center_losses.update(center_loss.item()*args.center_loss_lambda,x_s1.shape[0])

        loss = trial_loss + center_loss * args.center_loss_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if b % 100 == 0:
            progress.display(b)

    running_accuracy /= len(loader)
    return model.state_dict(),running_accuracy

def stage2_training(triter, teiter, model:Fusion_Model, optimizer, epoch, args):
    batch_time=AverageMeter('Time',':.3f')
    data_time=AverageMeter('Data',':.3f')
    # center_losses=AverageMeter('CenterLoss',':.4f')
    con_losses=AverageMeter('ConLoss',':.4f')
    # trial_losses=AverageMeter('TrialLoss',':.4f')
    e_losses=AverageMeter('ELoss',':.4f')
    progress = ProgressMeter(
        args.rounds_per_epoch,
        [batch_time,data_time,con_losses,e_losses],
        prefix="Epoch:[{}]".format(epoch)
    )
    model.train()
    device=torch.device(args.device)
    end=time.time()

    # running_accuracy=0
    for b in range(args.rounds_per_epoch):
        tr_batch,tr_label=next(triter)
        te_batch,te_label=next(teiter)
        
        tr_k = sorted(random.sample(list(range(0, args.p_num - 1)), args.tr_k))
        if args.backbone_switch[0]==1:
            eeg_map=[]
        if args.backbone_switch[1] == 1:
            eeg=[]
        if args.backbone_switch[2] == 1:
            peri=[]
        label=[]
        for i in range(len(tr_k)):
            if args.backbone_switch[0] == 1:
                eeg_map.append(tr_batch['eeg_feature_map'][tr_k[i]*args.trial_cnt:(tr_k[i]+1)*args.trial_cnt,:,:,:])
            if args.backbone_switch[1] == 1:
                # eeg.append(tr_batch['eeg_feature'][tr_k[i]*args.trial_cnt:(tr_k[i]+1)*args.trial_cnt,:,:])
                eeg.append(tr_batch['eeg_feature'][tr_k[i]*args.trial_cnt:(tr_k[i]+1)*args.trial_cnt])
            if args.backbone_switch[2] == 1:
                peri.append(tr_batch['peri_feature'][tr_k[i]*args.trial_cnt:(tr_k[i]+1)*args.trial_cnt])
            label.append(tr_label[tr_k[i]*args.trial_cnt:(tr_k[i]+1)*args.trial_cnt,:])
        tr_batch={}
        if args.backbone_switch[0] == 1:
            eeg_map=torch.cat(eeg_map,dim=0)
            tr_batch['eeg_feature_map']=eeg_map
        if args.backbone_switch[1] == 1:
            eeg=torch.cat(eeg,dim=0)
            tr_batch['eeg_feature']=eeg
        if args.backbone_switch[2] == 1:
            peri=torch.cat(peri,dim=0)
            tr_batch['peri_feature']=peri
        tr_label=torch.cat(label,dim=0)


        data_time.update(time.time() - end)

        # forward
        if args.backbone_switch==[1,1,1]:
            x_s1, z_t, x_s2, z_c, e = model.forward_eeg_peri(tr_batch)
            
            te_x_s1, te_z_t, te_x_s2, te_z_c, te_e = model.forward_eeg_peri(te_batch)
            
        elif args.backbone_switch==[1,1,0]:
            x_s1, z_t, x_s2, z_c, e=model.forward_eeg(tr_batch)

            te_x_s1, te_z_t, te_x_s2, te_z_c, te_e=model.forward_eeg(te_batch)
        elif args.backbone_switch==[0,0,1]:
            x_s1, z_t, x_s2, z_c, e=model.forward_peri(tr_batch)

            te_x_s1, te_z_t, te_x_s2, te_z_c, te_e=model.forward_peri(te_batch)

        # calculate loss
        con_loss=torch.tensor(0.0,device=device)


        for i in range(len(tr_k)):
            for j in range(i+1,len(tr_k)):
                con_loss+=loss_cl(
                    z_c[i*args.trial_cnt:(i+1)*args.trial_cnt,:],
                    z_c[j*args.trial_cnt:(j+1)*args.trial_cnt,:]
                )
        con_loss/=(len(tr_k)*(len(tr_k)-1))/2


        # e_loss=F.cross_entropy(e,tr_label[:,0])
        if args.dataset=='DEAP':
            e_loss=F.binary_cross_entropy(e,tr_label[:,:-1].float())
        else:
            e_loss=F.cross_entropy(e,tr_label[:,0])

        # update loss

        con_losses.update(con_loss.item(),te_z_c.shape[0])
        e_losses.update(e_loss.item(),e.shape[0])

        # loss = trial_loss + center_loss * args.center_loss_lambda+con_loss+e_loss
        loss = con_loss+e_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if b % args.rounds_per_epoch == 0:
            progress.display(b)

    return model.state_dict()

def validating(teloader, model:Fusion_Model, latest_statedict, args):
    device=torch.device(args.device)
    model.load_state_dict(latest_statedict)
    model.eval()
    label_list=[]
    pred_list=[]
    with torch.no_grad():
        for b,batch in enumerate(teloader):
            batch_data,label=batch[:]
            if args.backbone_switch==[1,1,1]: # eeg + peri
                x_s1,z_t,x_s2,z_c,e=model.forward_eeg_peri(batch_data)
                # data_for_cos_cri,x_s1,z_t,x_s2,z_c,e = model.forward_eeg_peri(batch_data)
            


            elif args.backbone_switch==[1,1,0]: # eeg
               
                x_s1,z_t,x_s2,z_c,e=model.forward_eeg(batch_data)
            elif args.backbone_switch==[0,0,1]: # peri
              
                x_s1,z_t,x_s2,z_c,e=model.forward_peri(batch_data)

            

            if args.dataset=='DEAP':
                label_list.append(label.long().cpu().numpy())
                probs=(e>0.5).long()
                pred_list.append(probs.cpu().numpy())
            else:
                label_list.append(label.cpu().numpy())
                classes=torch.argmax(e,dim=-1)
                pred_list.append(classes.cpu().numpy())

    label_list=np.concatenate(label_list,axis=0)
    pred_list=np.concatenate(pred_list,axis=0)

    if args.dataset=='DEAP':
        v_cm=confusion_matrix(y_true=label_list[:,0],y_pred=pred_list[:,0],labels=np.arange(0,2),normalize='true')
        a_cm=confusion_matrix(y_true=label_list[:,1],y_pred=pred_list[:,1],labels=np.arange(0,2),normalize='true')

        v_acc=accuracy_score(label_list[:,0],pred_list[:,0])
        a_acc=accuracy_score(label_list[:,1],pred_list[:,1])
        v_f1=f1_score(label_list[:,0],pred_list[:,0],average='macro')
        a_f1=f1_score(label_list[:,1],pred_list[:,1],average='macro')
        return v_acc,a_acc,v_f1,a_f1,v_cm,a_cm

    else:
        cm=confusion_matrix(y_true=label_list[:,0],y_pred=pred_list, \
                            labels=np.arange(0,args.emo_categories),normalize='true')
        # print('check cm')
        acc=accuracy_score(label_list[:,0],pred_list)
        f1=f1_score(label_list[:,0],pred_list,average='macro')
        return acc,f1,cm



    # cm=confusion_matrix(y_true=label_list,y_pred=pred_list, \
    #                     labels=np.arange(0,args.emo_categories),normalize='true')
    # # print('check cm')
    # acc=accuracy_score(label_list,pred_list)
    # f1=f1_score(label_list,pred_list,average='macro')
    # return acc,f1,cm

def results_record(result:list,filename:str):
    new_data=[]

    for i in range(len(result)):
        if type(result[i]) is float :
            new_data.append(str(result[i]))
        else:
            new_data.append(str(result[i].item()))

    with open(filename,'w+') as file:
        for i in range(len(result)):
            file.write(new_data[i])
            file.write('\n')


def main_worker(args):
    if args.seed is not None:
        setup_seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    print(args)

    if args.dataset=='DEAP':
        filenames=get_filename('../'+args.dataset+'_features/')
        logger=CompleteLogger(root='../DEAP_log/SI_two_stage/'+args.phase,phase=args.phase)
    else:
        filenames=get_filename('../'+args.dataset+'_features/4s_mat/')
        logger=CompleteLogger(root='../'+args.dataset+'_log/SI_two_stage/'+args.phase,phase=args.phase)
    
    # the labels of DEAP are obtained from excel tabel.
    if args.dataset=='DEAP':
        video_label=pd.read_excel('./DEAP_video_label.xlsx')
        video_label=video_label.values

        def label_process(labels:np.ndarray)->np.ndarray:
            assert labels.shape==(40,2)
            new_labels=np.random.randn(40,2)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if labels[i,j]<=5:
                        new_labels[i,j]=0
                    else:
                        new_labels[i,j]=1
            return new_labels

        video_label=label_process(video_label)
        # if args.label=='valence':
        #     emo_label=video_label[:,0]
        # else:
        #     emo_label=video_label[:,1]
        emo_label=video_label # (40,2)
        # print('check emo_label')

        tmp_label=np.random.randint(low=0,high=2,size=(args.trial_cnt,args.seg_cnt,2))
        for t in range(args.trial_cnt):
            tmp_label[t,:]=emo_label[t,:]
        emo_label=tmp_label.reshape(-1)
        del tmp_label
        if args.split:
            emo_label1,emo_label2=split_y_DEAP(emo_label,args)


    if args.backbone_switch[0]==1:
        eeg_maps=[]
    if args.backbone_switch[1]==1:
        eegs=[]
    if args.backbone_switch[2]==1:
        peris=[]
    for i in range(args.p_num):
        eeg_feature_map,eeg_feature,peri_feature = load_data(filenames[i],args) # np.ndarray
        if args.backbone_switch[0]==1:
            eeg_maps.append(eeg_feature_map)
        if args.backbone_switch[1]==1:
            eegs.append(eeg_feature)
        if args.backbone_switch[2]==1:
            peris.append(peri_feature)

    print(f'\nall data loaded into memory.\n')
    
    # define model
    device = torch.device(args.device)
    model = Fusion_Model(args).to(device)



    '''
      stage1
    '''









    # prepare dataset for stage 1
    if args.backbone_switch[0]==1:
        s1_eeg_maps=[]
    if args.backbone_switch[1]==1:
        s1_eegs=[]
    if args.backbone_switch[2]==1:
        s1_peris=[]

    for i in range(args.p_num):
        if args.backbone_switch[0]==1:
            if args.split:
                if args.dataset=='DEAP':
                    eeg_map1,eeg_map2=split_X_DEAP(eeg_maps[i],args)
                else:
                    eeg_map1,eeg_map2=split_X_SEED(eeg_maps[i],args)
                s1_eeg_maps.append(eeg_map1)
                
            else:
                s1_eeg_maps.append(eeg_maps[i])

        if args.backbone_switch[1]==1:
            if args.split:
                if args.dataset=='DEAP':
                    eeg1,eeg2=split_X_DEAP(eegs[i],args)
                else:
                    eeg1,eeg2=split_X_SEED(eegs[i],args)
                s1_eegs.append(eeg1)
               
            else:
                s1_eegs.append(eegs[i])
        if args.backbone_switch[2]==1:
            if args.split:
                if args.dataset=='DEAP':
                    peri1,peri2=split_X_DEAP(peris[i],args)
                else:
                    peri1,peri2=split_X_SEED(peris[i],args)
                s1_peris.append(peri1)
            else:
                s1_peris.append(peris[i])


    # stage 1 normalize
    if args.backbone_switch[0]==1:
        s1_eeg_maps=torch.from_numpy(np.concatenate(s1_eeg_maps,axis=0))
        mean_, std_ = s1_eeg_maps.mean([2, 3]), s1_eeg_maps.std([2, 3])
        mean_=mean_.mean()
        std_=std_.mean()
        s1_eeg_maps=normalize(s1_eeg_maps,mean_,std_)
    if args.backbone_switch[1]==1:
        scaler=StandardScaler()
        s1_eegs=np.concatenate(s1_eegs,axis=0)
        s1_eegs=scaler.fit_transform(s1_eegs)
    if args.backbone_switch[2]==1:
        scaler=StandardScaler()
        s1_peris=np.concatenate(s1_peris,axis=0)
        s1_peris=scaler.fit_transform(s1_peris)
    
    s1_dict={}
    if args.backbone_switch[0]==1:
        s1_dict['eeg_feature_map']=s1_eeg_maps
    if args.backbone_switch[1]==1:
        s1_dict['eeg_feature']=s1_eegs
    if args.backbone_switch[2]==1:
        s1_dict['peri_feature']=s1_peris

    if args.dataset=='DEAP':
        if args.split:
            s1_set=Dataset_SI_two_stage_DEAP(s1_dict,emo_label1,1,args)
        else:
            s1_set=Dataset_SI_two_stage_DEAP(s1_dict,emo_label,1,args)
    else:
        s1_set=Dataset_SI_two_stage_SEED(s1_dict,1,args)







    s1_loader=DataLoader(dataset=s1_set,batch_size=args.batch_size,shuffle=True)

    # define center loss
    center_criterion = CenterLoss(args.trial_cnt, model.merge_dim)
    center_criterion = center_criterion.to(device)
    
    

    model.init_all()
    parameters = []
    if args.backbone_switch[0]==1:
        parameters.extend(model.feature_map_backbone.parameters())
    if args.backbone_switch[1]==1:
        parameters.extend(model.eeg_backbone.parameters())
    if args.backbone_switch[2] == 1:
        parameters.extend(model.peri_backbone.parameters())
    if args.backbone_switch==[1,1,1]:

        if args.fusion_method=='HF_ICMA':
            parameters.extend(model.HF_ICMA.parameters())
    
        
    parameters.extend(model.trial_head.parameters())
    optimizer = Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    best_trial_acc = .0
    print(f'stage 1:')
    for epoch in range(1, args.stage1_epochs + 1):
        
        if args.fusion_method=='HF_ICMA':
            latest_statedict,acc_trial=stage1_training(s1_loader, model, center_criterion, optimizer, epoch,args)


        

        if acc_trial > best_trial_acc:
            
            torch.save(latest_statedict,logger.get_checkpoint_path('best_stage1'))
        best_trial_acc = max(best_trial_acc, acc_trial)
        if epoch % 5 == 0:
            print(f'stage1 cur trial training acc = {acc_trial},best trial training acc = {best_trial_acc}')

    # del the s1 data to freeze some memory
    if args.backbone_switch[0]==1:
        del s1_eeg_maps
    if args.backbone_switch[1]==1:
        del s1_eegs
    if args.backbone_switch[2]==1:
        del s1_peris










    model.load_state_dict(torch.load(logger.get_checkpoint_path('best_stage1')))
    
    
    model.freeze_stage1()

    print('\n')




    if args.split:
        if args.dataset=='DEAP':
            p_samples=int(args.ratio*args.seg_cnt)*args.trial_cnt
        else:
            p_samples=sum([seg-int(args.ratio*seg) for seg in args.s1_seg_cnt]+
                      [seg-int(args.ratio*seg) for seg in args.s2_seg_cnt]+
                      [seg-int(args.ratio*seg) for seg in args.s3_seg_cnt])
    else:
        if args.dataset=='DEAP':
            p_samples=args.trial_cnt*args.seg_cnt
        else:
            p_samples = sum(args.s1_seg_cnt+args.s2_seg_cnt+args.s3_seg_cnt)
        

    if args.dataset=='DEAP':
        p_v_acc=[]
        p_a_acc=[]
        p_v_f1=[]
        p_a_f1=[]
    else:
        p_acc = []
        p_f1 = []
    '''
    participant-independent loop
    '''
    for te_p in range(0,1):
        print(f'\nsubject {te_p + 1} as test:')
        tr_p = list(range(0, args.p_num))
        del tr_p[te_p]
        if args.backbone_switch[0] == 1:
            tr_eeg_maps = []
        if args.backbone_switch[1] == 1:
            tr_eegs = []
        if args.backbone_switch[2] == 1:
            tr_peris = []

        # get train subject data
        for i in range(len(tr_p)):
            if args.backbone_switch[0] == 1:
                if args.split:
                    if args.dataset=='DEAP':
                        _,eeg_map2=split_X_DEAP(eeg_maps[tr_p[i]],args)
                    else:
                        _,eeg_map2=split_X_SEED(eeg_maps[tr_p[i]],args)
                    tr_eeg_maps.append(eeg_map2)
                else:
                    tr_eeg_maps.append(eeg_maps[tr_p[i]])
            if args.backbone_switch[1] == 1:
                if args.split:
                    if args.dataset=='DEAP':
                        _,eeg2=split_X_DEAP(eegs[tr_p[i]],args)
                    else:
                        _,eeg2=split_X_SEED(eegs[tr_p[i]],args)
                    tr_eegs.append(eeg2)
                else:
                    tr_eegs.append(eegs[tr_p[i]])
            if args.backbone_switch[2] == 1:
                if args.split:
                    if args.dataset=='DEAP':
                        _,peri2=split_X_DEAP(peris[tr_p[i]],args)
                    else:
                        _,peri2=split_X_SEED(peris[tr_p[i]],args)
                    tr_peris.append(peri2)
                else:
                    tr_peris.append(peris[tr_p[i]])

        if args.backbone_switch[0] == 1:
            tr_eeg_maps = torch.from_numpy(np.concatenate(tr_eeg_maps,axis=0))
        if args.backbone_switch[1] == 1:
            tr_eegs = np.concatenate(tr_eegs, axis=0)
        if args.backbone_switch[2] == 1:
            tr_peris = np.concatenate(tr_peris, axis=0)

        # get test subject
        if args.backbone_switch[0]==1:
            if args.split:
                if args.dataset=='DEAP':
                    _,te_eeg_map=split_X_DEAP(eeg_maps[te_p],args)
                else:
                    _,te_eeg_map=split_X_SEED(eeg_maps[te_p],args)
            else:
                te_eeg_map=eeg_maps[te_p]
        if args.backbone_switch[1]==1:
            if args.split:
                if args.dataset=='DEAP':
                    _,te_eeg=split_X_DEAP(eegs[te_p],args)
                else:
                    _,te_eeg=split_X_SEED(eegs[te_p],args)
            else:
                te_eeg=eegs[te_p]
        if args.backbone_switch[2]==1:
            if args.split:
                if args.dataset=='DEAP':
                    _,te_peri=split_X_DEAP(peris[te_p],args)
                else:
                    _,te_peri=split_X_SEED(peris[te_p],args)
            else:
                te_peri=peris[te_p]
        
        '''
        stage 2 normalize
        '''
        if args.backbone_switch[0]==1:
            tr_mean_, tr_std_ = tr_eeg_maps.mean([2, 3]), tr_eeg_maps.std([2, 3])
            te_eeg_map=torch.from_numpy(te_eeg_map)
            te_mean_,te_std_ = te_eeg_map.mean([2,3]),te_eeg_map.std([2,3])
            tr_mean_=tr_mean_.mean()
            tr_std_=tr_std_.mean()
            te_mean_=te_mean_.mean()
            te_std_=te_std_.mean()
            tr_eeg_maps=normalize(tr_eeg_maps,tr_mean_,tr_std_)
            te_eeg_map=normalize(te_eeg_map,te_mean_,te_std_)
            



        if args.backbone_switch[1]==1:
            scaler=StandardScaler()
            tr_eegs=scaler.fit_transform(tr_eegs)
            te_eeg=scaler.fit_transform(te_eeg)
           
        if args.backbone_switch[2]==1:
            scaler=StandardScaler()
            tr_peris=scaler.fit_transform(tr_peris)
            te_peri=scaler.fit_transform(te_peri)
            

        tr_dict={}
        te_dict={}
        if args.backbone_switch[0]==1:
            tr_dict['eeg_feature_map']=tr_eeg_maps
            te_dict['eeg_feature_map']=te_eeg_map
        if args.backbone_switch[1]==1:
            tr_dict['eeg_feature']=tr_eegs
            te_dict['eeg_feature']=te_eeg
        if args.backbone_switch[2]==1:
            tr_dict['peri_feature']=tr_peris
            te_dict['peri_feature']=te_peri
        

        if args.dataset=='DEAP':
            if args.split:
                trset=Dataset_SI_two_stage_DEAP(x_dict=tr_dict,emo_label_one_p=emo_label2,stage=2,args=args)
                teset=Dataset_SI_two_stage_DEAP(x_dict=te_dict,emo_label_one_p=emo_label2,stage=2,args=args)
            else:
                trset=Dataset_SI_two_stage_DEAP(x_dict=tr_dict,emo_label_one_p=emo_label,stage=2,args=args)
                teset=Dataset_SI_two_stage_DEAP(x_dict=te_dict,emo_label_one_p=emo_label,stage=2,args=args)
        else: # dataset= SEED_IV or SEED_V
            trset=Dataset_SI_two_stage_SEED(x_dict=tr_dict,stage=2,args=args)
            teset=Dataset_SI_two_stage_SEED(x_dict=te_dict,stage=2,args=args)
        


        if args.dataset=='DEAP':
            triter=ForeverDataIterator(DataLoader(dataset=trset, batch_size=trset.pnum*args.trial_cnt, \
                                              sampler=MySampler_DEAP(args,trset.pnum)))
            teiter = ForeverDataIterator(DataLoader(dataset=teset, batch_size=args.trial_cnt, \
                                                sampler=MySampler_DEAP(args,1)))
        else:
            triter=ForeverDataIterator(DataLoader(dataset=trset, batch_size=trset.pnum*args.trial_cnt, \
                                              sampler=MySampler_SEED(args,trset.pnum)))
            teiter = ForeverDataIterator(DataLoader(dataset=teset, batch_size=args.trial_cnt, \
                                                sampler=MySampler_SEED(args,1)))
        te_val_loader=DataLoader(dataset=teset,batch_size=args.batch_size,shuffle=False)


        model.init_stage2_module()

     
                
        parameters = []
        parameters.extend(model.stage2_module.parameters())
        parameters.extend(model.proj_head.parameters())
        parameters.extend(model.emo_head.parameters())

        optimizer = Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

 

        if args.dataset=='DEAP':
            best_v_acc=.0
            best_a_acc=.0
            best_v_f1=.0
            best_a_f1=.0
            best_v_cm=np.zeros(shape=(args.emo_categories,args.emo_categories))
            best_a_cm=np.zeros(shape=(args.emo_categories,args.emo_categories))

        else:
            best_acc = .0
            best_f1 = .0
            best_cm = np.zeros(shape=(args.emo_categories,args.emo_categories))


        for epoch in range(1, args.stage2_epochs + 1):
      
         
        
            if args.fusion_method=='HF_ICMA':
                latest_statedict=stage2_training(triter,teiter,model,optimizer,epoch,args)

            if args.dataset=='DEAP':
                
              
                if args.fusion_method=='HF_ICMA':
                    v_acc,a_acc,v_f1,a_f1,v_cm,a_cm=validating(te_val_loader,model,latest_statedict,args)

      
                if v_acc>best_v_acc or a_acc>best_a_acc:
                    torch.save(latest_statedict, logger.get_checkpoint_path('best'))

                if v_acc>best_v_acc:
                    best_v_cm=v_cm
                    best_v_acc=v_acc

                if a_acc>best_a_acc:
                    best_a_cm=a_cm
                    best_a_acc=a_acc             
                
                best_v_f1=max(best_v_f1,v_f1)
                best_a_f1=max(best_a_f1,a_f1)
                if epoch % 5==0:
                    print(f'subject {te_p+1} best valence acc = {best_v_acc},best arousal acc= {best_a_acc} \n')

            else:
                
                # if args.fusion_method=='HF_ICMA' or args.fusion_method=='DFAF' or args.fusion_method=='CMHSA':
                if args.fusion_method=='HF_ICMA':
                    val_acc, val_f1, cm=validating(te_val_loader,model,latest_statedict,args)
        

                if val_acc > best_acc:
                    torch.save(latest_statedict, logger.get_checkpoint_path('best'))
                    best_cm = cm
                best_acc = max(best_acc, val_acc)
                best_f1 = max(best_f1, val_f1)

                if epoch % 5 == 0:
                    print(f'subject {te_p + 1} best acc = {best_acc} \n')

        if args.dataset=='DEAP':
            p_v_acc.append(best_v_acc)
            p_a_acc.append(best_a_acc)
            p_v_f1.append(best_v_f1)
            p_a_f1.append(best_a_f1)
        else:
            p_acc.append(best_acc)
            p_f1.append(best_f1)
        
        # if args.dataset=='DEAP':
        #     if args.backbone_switch == [1, 1, 1]:
        #         np.savez('../DEAP_confusion_matrix_save/SI_two_stage/' + args.phase + '/s' + str(te_p + 1) + \
        #                 '_' +'valence_'+ args.feature_map_backbone_type + '_' + args.backbone_type + '_' + args.fusion_method, \
        #                 cm=best_v_cm)
        #         np.savez('../DEAP_confusion_matrix_save/SI_two_stage/' + args.phase + '/s' + str(te_p + 1) + \
        #                 '_' +'arousal_'+ args.feature_map_backbone_type + '_' + args.backbone_type + '_' + args.fusion_method, \
        #                 cm=best_a_cm)

        #     else:
        #         np.savez('../DEAP_confusion_matrix_save/SI_two_stage/' + args.phase + '/s' + str(te_p + 1) + \
        #                 '_' +'valence_'+ args.feature_map_backbone_type + '_' + args.backbone_type, cm=best_v_cm)
        #         np.savez('../DEAP_confusion_matrix_save/SI_two_stage/' + args.phase + '/s' + str(te_p + 1) + \
        #                 '_' +'arousal_'+ args.feature_map_backbone_type + '_' + args.backbone_type, cm=best_a_cm)


        # else: # dataset == SEED_IV or SEED_V
        #     if args.backbone_switch == [1, 1, 1]:
        #         np.savez('../'+args.dataset+'_confusion_matrix_save/SI_two_stage/' + args.phase + '/s' + str(te_p + 1) + \
        #                 '_' + args.feature_map_backbone_type + '_' + args.backbone_type + '_' + args.fusion_method, \
        #                 cm=best_cm)
        #     else:
        #         np.savez('../'+args.dataset+'_confusion_matrix_save/SI_two_stage/' + args.phase + '/s' + str(te_p + 1) + \
        #                 '_' + args.feature_map_backbone_type + '_' + args.backbone_type, cm=best_cm)
                
    # results_record(p_acc, './p_acc.txt')
    # results_record(p_f1, './p_f1.txt')

    # if args.dataset=='DEAP':
    #     results_record(p_v_acc,'./p_v_acc.txt')
    #     results_record(p_v_f1,'./p_v_f1.txt')
    #     results_record(p_a_acc,'./p_a_acc.txt')
    #     results_record(p_a_f1,'./p_a_f1.txt')

    # else:
    #     results_record(p_acc,'./p_acc.txt')
    #     results_record(p_f1,'./p_f1.txt')