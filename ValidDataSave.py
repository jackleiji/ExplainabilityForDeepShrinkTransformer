# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 15:38:06 2022

@author: ChenMingfeng
"""
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from TransformerForAdLinerNewNet import ADTransformer
from util.build_criterion import *
from util.data_loader import *
from util.utils import *
from tqdm import tqdm, trange
import numpy as np

def extract_top_n_predictions(pre_labels, true_label):
    pred_dict = {}

    for i in range(1,len(pre_labels)):
        pred_dict['top-{}-预测ID'.format(i)] = pre_labels[i-1]
        pred_dict['top-{}-实际ID'.format(i)] = true_label[i - 1]

    return pred_dict



if __name__ == "__main__":

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    torch.cuda.empty_cache()
    print(device)
    dataset_information = [
                           # ['bank-additional-full-deonehot', 41189, 20, 4, 0.3],
                           # ['celeba_baldvsnonbald_normalised',202600, 39, 3, 0.6],
                            ['census-income-full-mixed-binarized', 299286, 100, 50, 0.14],
                           # ['creditcardfraud_normalised',284808, 29, 1, 0.5],
                           # ['shuttle_normalization',49098, 9, 3, 0.6],
                           #  ['annthyroid_21feat_normalised', 7201, 21, 3, 0.5],
                           # ['UNSW_NB15_traintest_backdoor-deonehot', 95330, 42, 6, 0.5],
                           # ['mammography_normalization', 11183, 6, 2, 0.4]
                            ] #no
    for numb, information in enumerate(dataset_information):
        block_size = information[2]
        sample_size = information[3]

        model = ADTransformer(block_size, num_layers=8, heads= sample_size , device=device).to(device)
        model.eval()

        state_dict = torch.load('./Model_best/Transformer_Base_'+ information[0] +'.pt', map_location=device)
        # state_dict = torch.load('./Model_best_Att/Transformer_Base_'+ information[0] +'.pt', map_location=device)
        # state_dict = torch.load('./Model_best_CNN/Transformer_Base_'+ information[0] +'.pt', map_location=device)


        model.load_state_dict(state_dict)
        sum = 0
        seed = 1024
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        x, labels, feature_name = dataLoading('./dataset/'+ information[0]+'.csv')
        x_train, x_test, y_train, y_test = train_test_split(x, labels, test_size=0.8, random_state=seed, stratify=labels)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print("验证集中 0 的数量：", np.sum(y_test==0), "; 验证集中 1 的数量：", np.sum(y_test==1))
        # pca
        if (information[0] == 'census-income-full-mixed-binarized'):
            pca = joblib.load('./Model_best/pca_for_census.joblib')
            x_test = pca.transform(x_test)
            feature_name = [f"att{i}" for i in range(0, 101)]

        vaild_data_few = DataSetADFew(x_test, y_test)
        test_loader = tqdm(torch.utils.data.DataLoader(vaild_data_few, batch_size=4096, shuffle=False))

        # val_metric_sum = 0.0
        val_step = 0
        AD_accuracy_score = 0.0
        df_pred = pd.DataFrame()
        rauc = np.zeros(len(test_loader))
        ap = np.zeros(len(test_loader))

        # 定义特征，标签，预测值，预测标签写出
        features_to_arr = []
        labels_to_arr = []
        pre_val_to_arr = []
        pre_label_to_arr = []

        for val_step, (features,labels) in enumerate(test_loader):
            # 关闭梯度计算
            with torch.no_grad():
                preditData = model(features).cpu().numpy().squeeze()
                test_lables = labels.cpu().numpy().squeeze()
                acdata = model.accuracy_predict(features, information[4])
                # 获取预测标签
                pre_label = acdata.numpy().squeeze()
                AD_accuracy_score += accuracy_score(pre_label, test_lables)
                # 打印基本指标
                print("AD_accuracy_score_mean", AD_accuracy_score / (val_step + 1))
                # 获取pr和roc值
                rauc[val_step], ap[val_step] = aucPerformance(preditData, test_lables)

                #保存预测结果
                # df_pred = df_pred.append(pre_label, ignore_index=True)
                extract_top_n_predictions(preditData, test_lables)

                features_to_arr.append(features.tolist())
                labels_to_arr.append(test_lables)
                pre_val_to_arr.append(preditData)
                pre_label_to_arr.append(pre_label)


                # pred = model(features)
                # score = pred.cpu().detach().numpy()
                # a = pred - labels
                # prd = torch.norm(a) / torch.norm(labels)
                # sum += prd.item()
        print(information[0]+"average AUC-ROC: %.4f, average AUC-PR: %.4f" % (np.mean(rauc), np.mean(ap)))
        print(df_pred.shape)
        # avg = sum / 112
        # print(avg)
        # 保存输出
        features_to_arr = np.concatenate(features_to_arr)
        labels_to_arr = np.concatenate(labels_to_arr)
        pre_val_to_arr = np.concatenate(pre_val_to_arr)
        pre_label_to_arr = np.concatenate(pre_label_to_arr)

        print(classification_report(labels_to_arr, pre_label_to_arr))

        print(features_to_arr.shape)
        data = {
            # 'Feature': features_to_arr[:, :],  # 假设有多个特征列
            'labels':labels_to_arr[:],
            'pre_val':pre_val_to_arr[:],
            'pre_labels': pre_label_to_arr[:]
            # 'max_score':[np.max(arr) for arr in pre_val_to_arr]
        }
        # print(data)
        # 将字典转换为DataFrame
        df = pd.DataFrame(data)

        # ---------将结果的DataFrame保存为CSV文件------------
        # df.to_csv('./PreDataForModel/'+information[0] +'_val_results.csv', index=False)
        # 获取标签为1的位置和0的随机信息
        one_indexs = np.where(labels_to_arr==1)[0]
        zero_indexs  = np.where(labels_to_arr == 0)[0]
        # 随机选择 4096 个位置
        if len(zero_indexs)>=4096:
            zero_random_indices = np.random.choice(zero_indexs, size=4096, replace=False)
        else:
            zero_random_indices = np.random.choice(zero_indexs, size=len(zero_indexs), replace=False)
        if len(one_indexs)>=4096:
            one_random_indices = np.random.choice(one_indexs, size=4096, replace=False)
        else:
            one_random_indices = np.random.choice(one_indexs, size=len(one_indexs), replace=False)
        # 根据位置提取特征数据
        one_features = features_to_arr[one_random_indices]
        zero_features = features_to_arr[zero_random_indices]

        # -------------------Feature保存为npy文件-----------------
        # np.save('./FeatureData/'+information[0] +'_val.npy', features_to_arr)
        # print("features_to_arr", np.shape(features_to_arr))
        tsen_feature = np.concatenate((one_features, zero_features), axis=0)
        # np.save('./FeatureData/'+information[0] +'_val.npy', tsen_feature)
        # print("features_to_arr", np.shape(tsen_feature))
        # np.save('./FeatureData/'+information[0] +'_val_labels.npy', np.concatenate((np.ones(len(one_indexs)),np.zeros(len(zero_random_indices))), axis=0))

        # -------------------验证集特征保存为csv文件-----------------
        #构建dataframe

        val_df_data = pd.DataFrame(tsen_feature, columns=feature_name[:-1])
        # 添加新的标签列到DataFrame
        val_df_data['class'] =np.concatenate((np.ones(len(one_random_indices)), np.zeros(len(zero_random_indices))), axis=0)
        #乱序
        val_df_data = val_df_data.sample(frac=1).reset_index(drop=True)
        print(val_df_data)
        val_df_data.to_csv('./FeatureDataForModel/' + information[0] + '_val.csv', index=False)