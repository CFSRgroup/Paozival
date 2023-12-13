
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize
from torchvision.models import vgg16
# utils
from utils.util import Dataset_SD_DEAP,Dataset_SD_SEED
# from utils.metrics import CenterLoss
from common.utils.meter import AverageMeter,ProgressMeter
from common.utils.logger import CompleteLogger
# sklearn
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
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
from fusion_framework import Fusion_Model



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
    

    eeg_en_stat=p_dataset['eeg_en_stat'] 
    if args.dataset=='DEAP':
        peri_feature=p_dataset['peri_feature']
    else:
        peri_feature=p_dataset['eye_feature']
    eeg_feature=eeg_en_stat

    
    if args.dataset!='DEAP':
        peri_feature=peri_feature[:,:args.peri_feat_dim] 


    return eeg_feature_map,eeg_feature,peri_feature

def load_DEAP_label(filename:str,args:argparse.Namespace=None):
    path='./DEAP_labels_for_all_participants/'
    p_label=loadmat(path+filename)
    p_label=p_label['label']
    # valence=p_label[:,0]
    # arousal=p_label[:,1]
    return p_label

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

def training(trloader,model:Fusion_Model,optimizer,epoch,args):
    batch_time=AverageMeter('Time',':.3f')
    data_time=AverageMeter('Data',':.3f')

    e_losses=AverageMeter('eLoss',':.4f')
    progress = ProgressMeter(
        len(trloader),
        [batch_time,data_time,e_losses],
        prefix="Epoch:[{}]".format(epoch)
    )

    model.train()
    device=torch.device(args.device)
    end=time.time()

    for b,batch in enumerate(trloader):
        batch_data,label=batch[:]
        # DEAP label:(bs,2)
        data_time.update(time.time()-end)
        if args.backbone_switch==[1,1,1]:  # eeg + peri
            # b_feature_map,b_eeg,b_peri=batch[:]
            # b_feature_map=b_feature_map.to(device)
            # b_eeg=b_eeg.to(device)
            # b_peri=b_peri.to(device)
            # label=label.to(device)
            # data_time.update(time.time()-end)

            x_s1,z_t,x_s2,z_c,e=model.forward_eeg_peri(batch_data)
            # print('check attention value')

        elif args.backbone_switch==[1,1,0]: # eeg
            # b_feature_map,b_eeg,label=batch[:]
            # b_feature_map=b_feature_map.to(device)
            # b_eeg=b_eeg.to(device)
            # label=label.to(device)
            # data_time.update(time.time()-end)

            x_s1,z_t,x_s2,z_c,e=model.forward_eeg(batch_data)

        elif args.backbone_switch==[0,0,1]: # peri
            # b_peri,label=batch[:]
            # b_peri=b_peri.to(device)
            # label=label.to(device)
            # data_time.update(time.time()-end)

            x_s1,z_t,x_s2,z_c,e=model.forward_peri(batch_data)
        
        # elif args.backbone_switch==[1,0,1]:
        #     x_s1,z_t,x_s2,z_c,e=model.forward_eeg_peri_subset(batch_data)


       
        if args.dataset=='DEAP':
            e_loss=F.binary_cross_entropy(e,label)
        else:
            e_loss=F.cross_entropy(e,label)
        e_losses.update(e_loss.item(),e.shape[0])

        loss = e_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time()-end)
        end=time.time()

        if b % args.print_freq==0:
            progress.display(b)

    return model.state_dict()

def validating(loader,model:Fusion_Model,latest_statedict,args):
    device=torch.device(args.device)
    model.load_state_dict(latest_statedict)
    model.eval()
    label_list=[]
    pred_list=[]
    with torch.no_grad():
        for b,batch in enumerate(loader):
            batch_data,label=batch[:]
            if args.backbone_switch==[1,1,1]: # eeg + peri
                # b_feature_map,b_eeg,b_peri,label=batch[:]
                # b_feature_map=b_feature_map.to(device)
                # b_eeg=b_eeg.to(device)
                # b_peri=b_peri.to(device)
                # label=label.to(device)
                # x,z_con,z_t,e=model.forward(eeg_map=b_feature_map,eeg_psd=b_psd,eeg_en_stats=b_en_stats,peri=b_peri)
                x_s1,z_t,x_s2,z_c,e=model.forward_eeg_peri(batch_data)
            elif args.backbone_switch==[1,1,0]: # eeg
                # b_feature_map,b_eeg,label=batch[:]
                # b_feature_map=b_feature_map.to(device)
                # b_eeg=b_eeg.to(device)
                # label=label.to(device)
                # x,z_con,z_t,e=model.forward(eeg_map=b_feature_map,eeg_psd=b_psd,eeg_en_stats=b_en_stats)
                x_s1,z_t,x_s2,z_c,e=model.forward_eeg(batch_data)
            elif args.backbone_switch==[0,0,1]: # peri
                # b_peri,label=batch[:]
                # b_peri=b_peri.to(device)
                # label=label.to(device)
                # x,z_con,z_t,e=model.forward(peri=b_peri)
                x_s1,z_t,x_s2,z_c,e=model.forward_peri(batch_data)

            elif args.backbone_switch==[1,0,1]:
                x_s1,z_t,x_s2,z_c,e=model.forward_eeg_peri_subset(batch_data)

            # label=label[:,0]  # validate emo label only
            
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
        v_cm=confusion_matrix(y_true=label_list[:,0],y_pred=pred_list[:,0],labels=[0,1],normalize='true')
        a_cm=confusion_matrix(y_true=label_list[:,1],y_pred=pred_list[:,1],labels=[0,1],normalize='true')
        v_acc=accuracy_score(label_list[:,0],pred_list[:,0])
        a_acc=accuracy_score(label_list[:,1],pred_list[:,1])
        v_f1=f1_score(label_list[:,0],pred_list[:,0],average='macro')
        a_f1=f1_score(label_list[:,1],pred_list[:,1],average='macro')
        return v_acc,a_acc,v_f1,a_f1,v_cm,a_cm
    else:
        cm=confusion_matrix(y_true=label_list,y_pred=pred_list,\
                            labels=np.arange(0,args.emo_categories),normalize='true')
        acc=accuracy_score(label_list,pred_list)
        f1=f1_score(label_list,pred_list,average='macro')
        return acc,f1,cm

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
        
    if args.dataset=='DEAP':
        logger=CompleteLogger(root='../DEAP_log/SD/'+args.phase,phase=args.phase)
        data_filenames=get_filename('../DEAP_features/')
        label_filenames=get_filename('./DEAP_labels_for_all_participants/')
    else:
        logger=CompleteLogger(root='../'+args.dataset+'_log/SD/'+args.phase,phase=args.phase)
        filenames=get_filename('../'+args.dataset+'_features/4s_mat/')

    # define model
    device=torch.device(args.device)
    model=Fusion_Model(args).to(device)




    for param in model.proj_head.parameters():
        if param.requires_grad:
            param.requires_grad = False

    for param in model.trial_head.parameters():
        if param.requires_grad:
            param.requires_grad=False
    

    # start exp
    if args.dataset=='DEAP':
        p_v_acc=[]
        p_a_acc=[]
        p_v_f1=[]
        p_a_f1=[]
    else:
        p_acc=[]
        p_f1=[]

    '''
    subject dependent loop
    '''

    for te_p in range(0,5):
        if args.dataset=='DEAP':
            eeg_feature_map,eeg_feature,peri_feature=load_data(data_filenames[te_p],args)
            p_label=load_DEAP_label(label_filenames[te_p],None)
            ##########
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
            ##########
            p_label=label_process(p_label) # (40,2)
            # if args.label=='valence':
            #     p_label=p_label[:,0]
            # else:
            #     p_label=p_label[:,1]



            label=np.empty(shape=(args.seg_cnt*args.trial_cnt,2)) # (800,2)
            for t in range(args.trial_cnt):
                label[t*args.seg_cnt:(t+1)*args.seg_cnt,:]=p_label[t,:]
        else: # dataset==SEED-IV or SEED-V
            eeg_feature_map,eeg_feature,peri_feature=load_data(filenames[te_p],args)
        
       
        if args.dataset=='DEAP':
            assert args.trial_cnt%args.k_fold==0
            # leave one fold out
            # cv_acc=[]
            # cv_f1=[]
            # cv_cm=[]
            cv_v_acc=[]
            cv_a_acc=[] 
            cv_v_f1=[]
            cv_a_f1=[]
            cv_v_cm=[]
            cv_a_cm=[]


            for k in range(0,args.k_fold):
                tr_dict={}
                te_dict={}

                _idx1=k*(args.trial_cnt//args.k_fold)*args.seg_cnt
                _idx2=(k+1)*(args.trial_cnt//args.k_fold)*args.seg_cnt

                label_tr1=label[0:_idx1,:]
                label_tr2=label[_idx2:,:]
                label_tr=np.concatenate([label_tr1,label_tr2],axis=0)
                label_te=label[_idx1:_idx2,:]

                if args.backbone_switch[0]==1:
                    eeg_feature_map_tr1=eeg_feature_map[0:_idx1,:,:,:]
                    eeg_feature_map_tr2=eeg_feature_map[_idx2:,:,:,:]
                    eeg_feature_map_tr=np.concatenate([eeg_feature_map_tr1,eeg_feature_map_tr2],axis=0)
                    eeg_feature_map_te=eeg_feature_map[_idx1:_idx2,:,:,:]
                    # normalize
                    eeg_feature_map_tr,eeg_feature_map_te=data_normalize(eeg_feature_map_tr,eeg_feature_map_te,None)
                    tr_dict['eeg_feature_map']=eeg_feature_map_tr
                    te_dict['eeg_feature_map']=eeg_feature_map_te

                if args.backbone_switch[1]==1:
                    eeg_feature_tr1=eeg_feature[0:_idx1,:]
                    eeg_feature_tr2=eeg_feature[_idx2:,:]
                    eeg_feature_tr=np.concatenate([eeg_feature_tr1,eeg_feature_tr2],axis=0)
                    eeg_feature_te=eeg_feature[_idx1:_idx2,:]
                    # normalize
                    eeg_feature_tr,eeg_feature_te=data_normalize(eeg_feature_tr,eeg_feature_te,None)
                    tr_dict['eeg_feature']=eeg_feature_tr
                    te_dict['eeg_feature']=eeg_feature_te

                if args.backbone_switch[2]==1:
                    peri_feature_tr1=peri_feature[0:_idx1,:]
                    peri_feature_tr2=peri_feature[_idx2:,:]
                    peri_feature_tr=np.concatenate([peri_feature_tr1,peri_feature_tr2],axis=0)
                    peri_feature_te=peri_feature[_idx1:_idx2,:]
                    # normalize
                    peri_feature_tr,peri_feature_te=data_normalize(peri_feature_tr,peri_feature_te,None)
                    tr_dict['peri_feature']=peri_feature_tr
                    te_dict['peri_feature']=peri_feature_te

                # best_acc_fold=.0
                # best_f1_fold=.0
                # best_cm_fold=None
                best_v_acc_fold=.0
                best_a_acc_fold=.0
                best_v_f1_fold=.0
                best_a_f1_fold=.0
                best_v_cm_fold=np.zeros(shape=(2,2))
                best_a_cm_fold=np.zeros(shape=(2,2))


                trset=Dataset_SD_DEAP(tr_dict,label_tr,args)
                teset=Dataset_SD_DEAP(te_dict,label_te,args)

                trloader=DataLoader(dataset=trset,batch_size=args.batch_size,shuffle=True)
                teloader=DataLoader(dataset=teset,batch_size=args.batch_size,shuffle=False)

                model.init_all()
                parameters=[]
                if args.backbone_switch[0]==1:
                    parameters.extend(model.feature_map_backbone.parameters())
                if args.backbone_switch[1]==1:
                    parameters.extend(model.eeg_backbone.parameters())
                if args.backbone_switch[2]==1:
                    parameters.extend(model.peri_backbone.parameters())
                if args.backbone_switch==[1,1,1]: # fusion
                    if args.fusion_method=='HF_ICMA':
                        parameters.extend(model.HF_ICMA.parameters())
                   

                parameters.extend(model.stage2_module.parameters())
                parameters.extend(model.emo_head.parameters())

                optimizer=Adam(parameters,lr=args.lr,weight_decay=args.weight_decay)

              

                for epoch in range(1,args.SD_epochs+1):
                    
                    if args.fusion_method=='HF_ICMA':
                        latest_statedict=training(trloader,model,optimizer,epoch,args)
                        v_acc,a_acc,v_f1,a_f1,v_cm,a_cm=validating(teloader,model,latest_statedict,args)

                 
                        
                    # if val_acc>best_acc_fold:
                        # shutil.copy(logger.get_checkpoint_path('latest'),logger.get_checkpoint_path('best'))
                    if v_acc>best_v_acc_fold or a_acc>best_a_acc_fold:
                        torch.save(latest_statedict,logger.get_checkpoint_path('best'))
                        # best_cm_fold=cm
                    if v_acc>best_v_acc_fold:
                        best_v_cm_fold=v_cm
                        best_v_acc_fold=v_acc

                    if a_acc>best_a_acc_fold:
                        best_a_cm_fold=a_cm
                        best_a_acc_fold=a_acc

                    # best_acc_fold=max(best_acc_fold,val_acc)
                    # best_f1_fold=max(best_f1_fold,val_f1)
                    best_v_f1_fold=max(best_v_f1_fold,v_f1)
                    best_a_f1_fold=max(best_a_f1_fold,a_f1)

                    if epoch % 5 ==0:
                        print(f'subject {te_p+1}, fold {str(k+1)} : best valence acc = {best_v_acc_fold},best arousal acc= {best_a_acc_fold} \n\n')

                # cv_acc.append(best_acc_fold)
                # cv_f1.append(best_f1_fold)
                # cv_cm.append(best_cm_fold)
                
                cv_v_acc.append(best_v_acc_fold)
                cv_a_acc.append(best_a_acc_fold)
                cv_v_f1.append(best_v_f1_fold)
                cv_a_f1.append(best_a_f1_fold)
                cv_v_cm.append(best_v_cm_fold)
                cv_a_cm.append(best_a_cm_fold)




      
        else:
            
            s_acc=[]
            s_f1=[]
            s_cm=[]
            for s in range(3):
                if s==0:

                    _idx1=0
                    _idx2=sum(args.s1_seg_cnt)
                    tr_seg_cnt=args.s1_seg_cnt[:((args.trial_cnt//3)//3)*2]

                elif s==1:

                    _idx1=sum(args.s1_seg_cnt)
                    _idx2=sum(args.s1_seg_cnt+args.s2_seg_cnt)
                    tr_seg_cnt=args.s2_seg_cnt[:((args.trial_cnt//3)//3)*2]

                elif s==2:

                    _idx1=sum(args.s1_seg_cnt+args.s2_seg_cnt)
                    _idx2=sum(args.s1_seg_cnt+args.s2_seg_cnt+args.s3_seg_cnt)
                    tr_seg_cnt=args.s3_seg_cnt[:((args.trial_cnt//3)//3)*2]
                tr_dict={}
                te_dict={}
                if args.backbone_switch[0]==1:

                    eeg_feature_map_s=eeg_feature_map[_idx1:_idx2,:,:,:]
                    eeg_feature_map_s_tr=eeg_feature_map_s[:sum(tr_seg_cnt),:,:,:]
                    eeg_feature_map_s_te=eeg_feature_map_s[sum(tr_seg_cnt):,:,:,:]
                    #normalize
                    eeg_feature_map_s_tr,eeg_feature_map_s_te=data_normalize(eeg_feature_map_s_tr,eeg_feature_map_s_te)
                    tr_dict['eeg_feature_map']=eeg_feature_map_s_tr
                    te_dict['eeg_feature_map']=eeg_feature_map_s_te

                if args.backbone_switch[1]==1:
                    eeg_feature_s=eeg_feature[_idx1:_idx2,:]
                    eeg_feature_s_tr=eeg_feature_s[:sum(tr_seg_cnt),:]
                    eeg_feature_s_te=eeg_feature_s[sum(tr_seg_cnt):,:]
                    #normalize
                    eeg_feature_s_tr,eeg_feature_s_te=data_normalize(eeg_feature_s_tr,eeg_feature_s_te)
                    tr_dict['eeg_feature']=eeg_feature_s_tr
                    te_dict['eeg_feature']=eeg_feature_s_te

                if args.backbone_switch[2]==1:
                    peri_feature_s=peri_feature[_idx1:_idx2,:]
                    peri_feature_s_tr=peri_feature_s[:sum(tr_seg_cnt),:]
                    peri_feature_s_te=peri_feature_s[sum(tr_seg_cnt):,:]
                    # normalize
                    peri_feature_s_tr,peri_feature_s_te=data_normalize(peri_feature_s_tr,peri_feature_s_te)
                    tr_dict['peri_feature']=peri_feature_s_tr
                    te_dict['peri_feature']=peri_feature_s_te
                
                best_acc_session=.0
                best_f1_session=.0
                best_cm_session=None

                trset=Dataset_SD_SEED(tr_dict,1,s,args)
                teset=Dataset_SD_SEED(te_dict,2,s,args)

                trloader=DataLoader(dataset=trset,batch_size=args.batch_size,shuffle=True)
                teloader=DataLoader(dataset=teset,batch_size=args.batch_size,shuffle=False)

                model.init_all()
                parameters=[]
                if args.backbone_switch[0]==1:
                    parameters.extend(model.feature_map_backbone.parameters())
                if args.backbone_switch[1]==1:
                    parameters.extend(model.eeg_backbone.parameters())
                if args.backbone_switch[2]==1:
                    parameters.extend(model.peri_backbone.parameters())
                if args.backbone_switch==[1,1,1]: # fusion
                    if args.fusion_method=='HF_ICMA':
                        parameters.extend(model.HF_ICMA.parameters())
                   

                parameters.extend(model.stage2_module.parameters())
                
                parameters.extend(model.emo_head.parameters())
                optimizer=Adam(parameters,lr=args.lr,weight_decay=args.weight_decay)
             

                for epoch in range(1,args.SD_epochs+1):

                    # latest_statedict=training(trloader,model,optimizer,epoch,args)
                    
                    # torch.save(latest_statedict,logger.get_checkpoint_path('latest'))
                    # val_acc,val_f1,cm=validating(teloader,model,latest_statedict,args)

                    if args.fusion_method=='HF_ICMA':
                        latest_statedict=training(trloader,model,optimizer,epoch,args)
                        val_acc,val_f1,cm=validating(teloader,model,latest_statedict,args)

                

                    if val_acc>best_acc_session:
                        torch.save(latest_statedict,logger.get_checkpoint_path('best'))
                        best_cm_session=cm
                    best_acc_session=max(best_acc_session,val_acc)
                    best_f1_session=max(best_f1_session,val_f1)
                    if epoch % 5 ==0:
                        print(f'subject {te_p+1}, session {str(s+1)} : best acc = {best_acc_session},best f1 = {best_f1_session} \n\n')
                s_acc.append(best_acc_session)
                s_f1.append(best_f1_session)
                s_cm.append(best_cm_session)

        if args.dataset=='DEAP':
            # p_acc.append(np.mean(cv_acc))
            # p_f1.append(np.mean(cv_f1))
            # p_cm=np.mean(cv_cm,axis=0)
            p_v_acc.append(np.mean(cv_v_acc))
            p_a_acc.append(np.mean(cv_a_acc))    
            p_v_f1.append(np.mean(cv_v_f1))
            p_a_f1.append(np.mean(cv_a_f1))
            p_v_cm=np.mean(cv_v_cm,axis=0)
            p_a_cm=np.mean(cv_a_cm,axis=0)


            # if args.backbone_switch==[1,1,1]:
                
            #     # np.savez('../DEAP_confusion_matrix_save/SD/'+args.phase+\
            #     #      '/s'+str(te_p+1)+'_'+'valence_'+args.feature_map_backbone_type+'_'+args.backbone_type\
            #     #      +'_'+args.fusion_method,cm=p_v_cm)
            #     # np.savez('../DEAP_confusion_matrix_save/SD/'+args.phase+\
            #     #      '/s'+str(te_p+1)+'_'+'arousal_'+args.feature_map_backbone_type+'_'+args.backbone_type\
            #     #      +'_'+args.fusion_method,cm=p_a_cm)
            #     pass

            # else:
            #     np.savez('../DEAP_confusion_matrix_save/SD/'+args.phase+ \
            #             '/s'+str(te_p+1)+'_'+'valence_'+args.feature_map_backbone_type+'_'+args.backbone_type,cm=p_v_cm)
            #     np.savez('../DEAP_confusion_matrix_save/SD/'+args.phase+ \
            #             '/s'+str(te_p+1)+'_'+'arousal_'+args.feature_map_backbone_type+'_'+args.backbone_type,cm=p_a_cm)
        else:
            p_acc.append(np.mean(s_acc))
            p_f1.append(np.mean(s_f1))
            p_cm=np.mean(s_cm,axis=0)
        
            # if args.backbone_switch==[1,1,1]:
            #     # np.savez('../'+args.dataset+'_confusion_matrix_save/SD/' + args.phase + '/s' + str(te_p + 1) + \
            #     #     '_' + args.feature_map_backbone_type + '_' + args.backbone_type + '_'+args.fusion_method, \
            #     #     cm=p_cm)
            #     pass
            # else:
            #     np.savez('../'+args.dataset+'_confusion_matrix_save/SD/' + args.phase + '/s' + str(te_p + 1) + \
            #             '_' + args.feature_map_backbone_type + '_' + args.backbone_type ,cm=p_cm)
    

    if args.dataset=='DEAP':
        print(p_v_acc)
        print(p_a_acc)
    else:
        print(p_acc)

    # if args.dataset=='DEAP':
    #     results_record(p_v_acc,'./p_v_acc.txt')
    #     results_record(p_v_f1,'./p_v_f1.txt')
    #     results_record(p_a_acc,'./p_a_acc.txt')
    #     results_record(p_a_f1,'./p_a_f1.txt')

    # else:
    #     results_record(p_acc,'./p_acc.txt')
    #     results_record(p_f1,'./p_f1.txt')
            

