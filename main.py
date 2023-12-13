# main
import argparse

from SI_two_stage import main_worker
# from SD import main_worker
# &&&&&&
# from pipeline.SI import main_worker
# from pipeline.SI_two_stage_frequency_peri import main_worker
# from pipeline.SD_frequency_peri import main_worker
# &&&&&&


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='SD:participant-dependent, SI:participant-generic')

    # model
    parser.add_argument('--backbone_type',type=str,default='channel_wise_transformer')
    parser.add_argument('--feature_map_backbone_type', type=str, default='aff_resnet')
    parser.add_argument('--fusion_method', type=str, default='HF_ICMA')
    parser.add_argument('--fusion_hidden',type=int,default=512)
    parser.add_argument('--backbone_hidden',type=int,default=256)
    parser.add_argument('--positional_encoding',type=bool,default=True)
    parser.add_argument('--num_fc',type=int,default=5)
    parser.add_argument('--hidden',type=int,default=256)
    parser.add_argument('--num_layer',type=int,default=2)

    # dataset
    parser.add_argument('--dataset',type=str,default='SEED_V')
    parser.add_argument('--label',type=str,default='valence')
    parser.add_argument('--p_num',type=int,default=16) # DEAP:32  SEED-IV:15  SEED-V:16
    parser.add_argument('--seg_cnt',type=int,default=20) # DEAP:20
    parser.add_argument('--trial_cnt',type=int,default=45) # DEAP:40  SEED-IV:72  SEED-V:45
    parser.add_argument('--emo_categories',type=int,default=5) # DEAP:2  SEED-IV:4  SEED-V:5
    parser.add_argument('--EEG_channel',type=int,default=62) #  DEAP:32  SEED-IV:62  SEED-V:62
    parser.add_argument('--eeg_channel_feat_dim',type=int,default=7) 
    parser.add_argument('--frequency_band',type=int,default=5)
    parser.add_argument('--peri_feat_dim',type=int,default=24) #  DEAP:55  SEED-IV:22  SEED-V:24
    parser.add_argument('--ratio',type=float,default=0.1)

    # SEED-IV seg cnt
    # parser.add_argument('--s1_seg_cnt',type=list,default=[42, 23, 49, 32, 22, 40, 38, 52, 36, 42, 12, 27, 54, 42,\
    #                                                       64, 35, 17, 44, 35, 12, 28, 28, 43,34])
    # parser.add_argument('--s2_seg_cnt', type=list,default=[55, 25, 34, 36, 53, 27, 34, 46, 34, 20, 60, 12, 36, 27, \
    #                                                        44, 15, 46, 49, 45, 10, 37, 44, 24,19])
    # parser.add_argument('--s3_seg_cnt', type=list, default=[42, 32, 23, 45, 48, 26, 64, 23, 26, 16, 51, 41, 39, 19, \
    #                                                         28, 44, 14, 17, 45, 22, 39, 38, 41,39])
    # SEED-V seg cnt
    parser.add_argument('--s1_seg_cnt',type=list,default=[18, 24, 59, 46, 36, 64, 74, 17, 66, 35, 43, 43, 58, 60, 38])
    parser.add_argument('--s2_seg_cnt', type=list,default=[59, 47, 16, 31, 32, 14, 60, 57, 30, 24, 46, 29, 23, 54, 19])
    parser.add_argument('--s3_seg_cnt', type=list, default=[72, 16, 41, 22, 13, 59, 21, 18, 57, 71, 55, 29, 51, 32, 44])

    # SEED-IV emo label
    # parser.add_argument('--s1_emo_label',type=list,\
    #                     default=[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3])
    # parser.add_argument('--s2_emo_label', type=list,
    #                     default=[2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1])
    # parser.add_argument('--s3_emo_label', type=list,
    #                     default=[1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0])
    # # SEED-V emo label
    parser.add_argument('--s1_emo_label',type=list,default=[4,1,3,2,0,4,1,3,2,0,4,1,3,2,0])
    parser.add_argument('--s2_emo_label', type=list,default=[2,1,3,0,4,4,0,3,2,1,3,4,1,2,0])
    parser.add_argument('--s3_emo_label', type=list,default=[2,1,3,0,4,4,0,3,2,1,3,4,1,2,0])


    # others
    parser.add_argument('--seed',type=int,default=46)
    parser.add_argument('--center_loss_lambda',type=float,default=0.001)
    parser.add_argument('--backbone_switch',type=list,default=[1,1,1])


    # training
    parser.add_argument('--phase',type=str,default='eeg_peri')
    parser.add_argument('--stage1_epochs',type=int,default=10)  
    parser.add_argument('--stage2_epochs',type=int,default=15)
    parser.add_argument('--SD_epochs',type=int,default=15) 
    parser.add_argument('--rounds_per_epoch',type=int,default=20) 
    parser.add_argument('--print_freq',type=int,default=20)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--split',type=bool,default=True) 
    parser.add_argument('--tr_k', type=int, default=2) 
    parser.add_argument('--k_fold',type=int,default=5) 
    parser.add_argument('--lr',type=float,default=1e-3)
    parser.add_argument('--weight_decay',type=float,default=1e-3)  
    parser.add_argument('--device',type=str,default='cuda:0')

    args=parser.parse_args()

    main_worker(args)
    print(1)