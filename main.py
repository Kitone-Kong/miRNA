# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import os
import pandas as pd
import numpy as np
import re
import torch
from torch import nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.nn import init
import torch.utils.data as D

FILE_PATH_gene = '/Users/kongxuwen/Desktop/竞赛/train_dataset/gene_seq.csv'
FILE_PATH_miRNA = '/Users/kongxuwen/Desktop/竞赛/train_dataset/mirna_seq.csv'
FILE_PATH_Train = '/Users/kongxuwen/Desktop/竞赛/train_dataset/Train.csv'
FILE_PATH_test = '/Users/kongxuwen/Desktop/竞赛/test_dataset.csv'

#创建DNA，RNA特征列表
def rox_label():
    Gene_list = []
    miRNA_list = []

    DNA_list = ['A', 'T', 'G', 'C']
    RNA_list = ['A', 'U', 'G', 'C']

    for a in DNA_list:
        for b in DNA_list:
            for c in DNA_list:
                Gene_list.append(a + b + c)

    for a in RNA_list:
        for b in RNA_list:
            for c in RNA_list:
                miRNA_list.append(a + b + c)

    return (Gene_list, miRNA_list)

#创建DNA，RNA原始数据特征矩阵
def Data(gene_data, Gene_list, rna_data, miRNA_list):
    gene_matrtix = np.empty([16127, len(Gene_list)], dtype=int)

    for i in range(len(Gene_list)):
        for j in range(0, 16127):
            Seq = gene_data.loc[j, 'sequence']
            pattern = re.compile(Gene_list[i])

            Match_list = pattern.findall(Seq)
            gene_matrtix[j][i] = len(Match_list)

    rna_matrtix = np.empty([2656, len(miRNA_list)], dtype=int)

    for i in range(len(miRNA_list)):
        for j in range(0, 2656):
            Seq = rna_data.loc[j, 'seq']
            pattern = re.compile(miRNA_list[i])

            Match_list = pattern.findall(Seq)
            rna_matrtix[j][i] = len(Match_list)

    return (gene_matrtix, rna_matrtix)

#创建训练/测试数据的特征矩阵
def Data_Fix(FILE_PATH,gene_data,rna_data,gene_matrtix,rna_matrtix):
    data = pd.read_csv(FILE_PATH)

    Data = np.empty([len(data), len(Gene_list) + len(miRNA_list) + 1], dtype=int)

    if (data.shape[1] == 3):
        D = {'Functional MTI': 1, 'Non-Functional MTI': 0}

        for i in range(len(data)):
            data.loc[i, 'label'] = D[data.loc[i, 'label']]

        for i in range(len(data)):
            Data[i][-1] = data.loc[i, 'label']

    for i in range(len(data)):
        index_gene = gene_data[gene_data['label'] == data.loc[i, 'gene']].index.tolist()
        for j in range(len(Gene_list) - 1):
            Data[i][j] = gene_matrtix[index_gene[0]][j]

        index_rna = rna_data[rna_data['mirna'] == data.loc[i, 'miRNA']].index.tolist()
        for j in range(len(miRNA_list) - 1):
            Data[i][j + len(Gene_list)] = rna_matrtix[index_rna[0]][j]

    return (Data)

#logistic regression模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        self.Sigmid = nn.Sigmoid()

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        y_pred = y_pred.squeeze(-1)
        return y_pred

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0, 0
    for x, y in data_iter:
        if(net(x).round()==y):
            acc_sum+=1
        n += y.shape[0]
    return acc_sum / n

def train(net, train_iter,test_iter,criterion,num_epochs,optimizer):
    for epoch in range(num_epochs):
        for x,y in train_iter:
            y_pred = net(x)
            Train_loss = criterion(y_pred, y)

            optimizer.zero_grad()
            Train_loss.backward()
            optimizer.step()

        for x,y in test_iter:
            y_pred = net(x)
            Test_loss = criterion(y_pred,y)

        acc = evaluate_accuracy(test_iter,net)

        print("epoch %d,Train_loss %.4f,Test_loss %.4f,acc %.3f"%(epoch,Train_loss.item(),Test_loss.item(),acc))



if __name__ == '__main__':
    Gene_list = rox_label()[0]
    miRNA_list = rox_label()[1]

    gene_data = pd.read_csv(FILE_PATH_gene)
    rna_data = pd.read_csv(FILE_PATH_miRNA)

    gene_matrtix, rna_matrtix = Data(gene_data, Gene_list, rna_data, miRNA_list)

    Train_data = Data_Fix(FILE_PATH_Train,gene_data,rna_data,gene_matrtix,rna_matrtix)
    # Test_data = Data_Fix(FILE_PATH_test)

    #标准化
    # X = np.vstack((Train_data, Test_data))
    X_scaler = StandardScaler()
    x = X_scaler.fit_transform(Train_data[:, :-1])

    #PCA
    pca = PCA(n_components=0.99)
    pca.fit(x)
    x_pca = pca.transform(x)

    n_features = len(x_pca[0][:])
    Train_features = torch.from_numpy(x_pca[:516, :]).to(torch.float32)
    Train_labels = torch.from_numpy(Train_data[:516, -1]).to(torch.float32)

    Test_features = torch.from_numpy(x_pca[516:738, :]).to(torch.float32)
    Test_labels = torch.from_numpy(Train_data[516:738, -1]).to(torch.float32)

    Train_dataset = D.TensorDataset(Train_features, Train_labels)
    Train_iter = D.DataLoader(Train_dataset,shuffle=True)

    Test_dataset = D.TensorDataset(Test_features, Test_labels)
    Test_iter = D.DataLoader(Test_dataset,shuffle=True)
    #网络
    n_hidden = 50

    net = LogisticRegressionModel(n_features)

    for params in net.parameters():
        init.normal_(params, mean=0, std=0.01)
    #loss
    criterion = torch.nn.BCEWithLogitsLoss()
    #优化方法
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    #训练
    train(net,Train_iter,Test_iter,criterion,num_epochs = 1000,optimizer = optimizer)



    # features_test = torch.from_numpy(x_pca[738:923, :]).to(torch.float32)
    # result = pd.DataFrame(net(features_test).detach().numpy(), dtype=float)
    # test = pd.read_csv(FILE_PATH_test)
    # outfile = pd.concat([test, result], axis=1)
    # outfile.to_csv('/Users/kongxuwen/Desktop/result.csv')