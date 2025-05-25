#!/usr/bin/env python
'''
This is to do model selection among survival models.
Implemented as Functional Procedure
'''
import numpy as np
# import sklearn.cross_validation
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import DeepConvSurv_pytorch as deep_conv_surv
import pandas as pd
import os
from PIL import Image

# ── 把所有路径都基于此目录拼接 ───────────────────────────────
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PATCHES_DIR = os.path.join(DATA_DIR, 'patches', 'TCGA-BL-A3JM')
PATIENTS_CSV = os.path.join(DATA_DIR, 'patients.csv')
CLUSTER_CSV = os.path.join(BASE_DIR, 'cluster_result', 'patches_1000_cls10.csv')
CLUSTER_PATCHES = os.path.join(BASE_DIR, 'cluster_result', 'patches_1000_cls10')
LOG_DIR = os.path.join(BASE_DIR, 'log')

# 确保日志文件夹存在
os.makedirs(LOG_DIR, exist_ok=True)

# hyperparams
model = 'deepconvsurv'
epochs = 20
lr = 5e-4
seed = 1
batchsize = 30


def convert_index(inputpid, expandlabel):
    outputindex = []
    for pid in inputpid:
        tmp = list(expandlabel['pid'][expandlabel['pid'] == pid].index)
        outputindex.append(tmp)
    patient_num = [len(x) for x in outputindex]
    outputindex = [y for x in outputindex for y in x]
    return outputindex, patient_num


def model_selection(img_path, label_path, expand_label_path,
                    train_test_ratio, train_valid_ratio, seed=seed,
                    model='deepconsurv', batchsize=batchsize, epochs=epochs,
                    lr=lr, **kwargs):
    print(' ')
    print('--------------------- Model Selection ---------------------')
    print('---------------Training Model: ', model, '--------------')
    print('---------------------parameters----------------------------')
    print("epochs: ", epochs, "  tr/test ratio: ", train_test_ratio, "  tr/val ratio: ", train_valid_ratio)
    print("learning rate: ", lr, "batch size: ", batchsize)
    print('-----------------------------------------------------------')
    print(' ')
    # load labels
    labels = pd.read_csv(label_path)
    expand_label = pd.read_csv(expand_label_path)

    cluster_id = int(expand_label_path.split('cls')[-1].split('.')[0])

    ## generate index
    e = labels["status"]
    # rdn_index = sklearn.cross_validation.StratifiedKFold(e, n_folds=5,
    #                                                      shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rdn_index = skf.split(np.zeros(len(e)), e)
    testci = []
    index_num = 1
    for trainindex, testindex in rdn_index:
        test_index, test_patchidxcnt = convert_index(labels['pid'].values[testindex], expand_label)
        # cv_idx = sklearn.cross_validation.StratifiedShuffleSplit(e.values[trainindex],
        #                                                          n_iter=1,test_size=1-train_valid_ratio, random_state = seed)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_valid_ratio, random_state=seed)
        cv_idx = sss.split(np.zeros(len(e.values[trainindex])), e.values[trainindex])
        sublabels = labels['pid'].values[trainindex]
        for tr_idx, val_idx in cv_idx:
            train_index, train_patchidxcnt = convert_index(sublabels[tr_idx], expand_label)
            valid_index, valid_patchidxcnt = convert_index(sublabels[val_idx], expand_label)
        tr_idx_name = './log/wsisa_patch10/train_cluster%d_fold%d.csv' % (cluster_id, index_num)
        te_idx_name = './log/wsisa_patch10/test_cluster%d_fold%d.csv' % (cluster_id, index_num)
        va_idx_name = './log/wsisa_patch10/valid_cluster%d_fold%d.csv' % (cluster_id, index_num)
        # risk_name = 'risk%d.csv' %index_num

        print('')
        print(tr_idx_name, te_idx_name, va_idx_name)
        print('')
        # 创建保存文件的目录（如果不存在）
        os.makedirs('./log/wsisa_patch10', exist_ok=True)
        np.savetxt(tr_idx_name, train_index, delimiter=',', header='index')
        np.savetxt(te_idx_name, test_index, delimiter=',', header='index')
        np.savetxt(va_idx_name, valid_index, delimiter=',', header='index')

        # sampleimg = img_path + "/" + os.listdir(img_path)[0] + "/" + os.listdir(img_path + "/" + os.listdir(img_path)[0])[0]
        # sampleimg = '/home/cy/ssd_4t/nlst/nlst_proto_swav/100012/11445_3707_18142_0653_valid.npy'
        # sampleimg = '/smile/nfs/nlst-patch_1000/101287/NLSI0000117_44620_21873_509_509.npy'
        NLSTimage = Image.open(r'D:\pycharm\WSISA-main\WSISA\patches\TCGA-BL-A3JM\0.jpg')
        NLSTimage = np.array(NLSTimage)
        # NLSTimage = np.load(sampleimg)
        width = NLSTimage.shape[0]
        height = NLSTimage.shape[1]
        channel = NLSTimage.shape[2]
        if model == 'deepconvsurv':
            hyperparams = {
                'learning_rate': lr,
                'channel': channel,
                'width': width,
                'height': height,
            }
            network = deep_conv_surv.DeepConvSurv(**hyperparams)
            log = network.train(data_path=img_path, label_path=expand_label_path, train_index=train_index,
                                test_index=valid_index, valid_index=test_index, model_index=index_num,
                                cluster=cluster_id, batch_size=batchsize, ratio=train_test_ratio, num_epochs=epochs)
            testci.append(log)
        else:
            print("please select a right model!")
            continue

        index_num = index_num + 1
        # if model != 'deepcca':
        #    np.savetxt(risk_name, np.c_[test_risk, test_t, test_e], header="risk, time, status",delimiter=',', comments='')
    print("In model: ", model, " the mean value of test: ", np.mean(testci), "standard value of test: ", np.std(testci))


if __name__ == '__main__':
    print("Model_selection Unit Test")
    root_path = CLUSTER_PATCHES
    img_path = PATCHES_DIR
    label_path = PATIENTS_CSV
    expand_label_path = CLUSTER_CSV
    for i in range(4, 7):
        print('======================================================')
        print('======================================================')
        print('=================      cluster  %d      ==============' % i)
        print('======================================================')
        print('======================================================')
        exp_lable_path = root_path + str(i) + '.csv'
        model_selection(img_path=img_path,
                        # clinical_path = '/smile/nfs/nlst-patch_1000/clinicalNormalized.csv',
                        label_path=label_path,
                        expand_label_path=expand_lable_path,
                        model=model, train_test_ratio=0.9, train_valid_ratio=0.9)
else:
    print("Load Model Selection Module")

    ##load labels
    # imgname = map(str,labels["img"])
    # data_num = len(imgname)
    # train_num = np.floor(data_num * train_test_ratio * train_valid_ratio)
    # valid_num = np.floor(data_num * train_test_ratio - train_num)
    # test_num = data_num - train_num - valid_num
    # train_num = int(train_num)
    # valid_num = int(valid_num)
    # test_num = int(test_num)
    # print "number of samples: ", data_num

# python -u cluster_select_deepconvsurv.py | tee -a ./log/clus15_fold5.txt