import numpy as np
import torch
import pandas as pd
from d2l import torch as d2l
from torch.utils import data
from torch import nn
#基础函数
#d2l.plot()
def load_array(data_arrays,batch_size,is_train=True):
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=is_train)
#读取数据集
train_data=pd.read_csv('D:\\PYCHARM\\kaggle_data\\train.csv')
test_data=pd.read_csv('D:\\PYCHARM\\kaggle_data\\test.csv')
#print(train_data.shape)
#print(test_data.shape)
#print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])
#print(train_data.iloc[0:4,-2:])
#print(train_data.shape[0])
#数据预处理
all_features=pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))
numeric_features=all_features.dtypes[all_features.dtypes!='object'].index
#print(numeric_features.index)
#标准化
#numeric_features=numeric_features.index
all_features[numeric_features]=all_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
all_features[numeric_features]=all_features[numeric_features].fillna(0)
#将非数值数据转为独热编码
all_features=pd.get_dummies(all_features,dummy_na=True,dtype=int)
#all_features=all_features*1
print(all_features.shape)
#转为张量
n_train=train_data.shape[0]
train_features=torch.tensor(all_features[:n_train].values,dtype=torch.float32)
test_features=torch.tensor(all_features[n_train:].values,dtype=torch.float32)
train_labels=torch.tensor(train_data.iloc[:,-1].values.reshape(-1,1),dtype=torch.float32)
#print(train_labels)
#train_labels=torch.tensor(train_data.SalePrice.values.reshape(-1,1),dtype=torch.float32)
#定义训练基本函数
loss=nn.MSELoss()
def get_net():
    net = nn.Sequential(nn.Linear(train_features.shape[1], 1))
    return net

#print(net[0].weight)
#print(net[0].bias)
def log_rmse(net,features,labels):
    clipped_preds=torch.clamp(net(features),1,float('inf'))
    rmse=torch.sqrt(loss(torch.log(clipped_preds),torch.log(labels)))
    return rmse.item()
def train(net,train_features,train_labels,test_features,test_labels,num_epochs,learning_rate,weight_decay,batch_size):
    train_ls,test_ls=[],[]
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=weight_decay)
    train_iter=load_array((train_features,train_labels),batch_size,is_train=True)
    for epoch in range(num_epochs):
        for X,y in train_iter:
            optimizer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:#保证健壮性，利用整个训练集训练时无测试集
            test_ls.append(log_rmse(net,test_features,test_labels))
    return train_ls,test_ls
def commomtrain(train_features,train_labels,epochs,learning_rate,weight_decay,batch_size):
    net=get_net()
    #划分训练集
    part=train_features.shape[0]//4
    X_train=train_features[:part*3]
    y_train=train_labels[:part*3]
    X_valid=train_features[part*3:]
    y_valid=train_labels[part*3:]
    #训练
    train_ls,valid_ls=train(net,X_train,y_train,X_valid,y_valid,epochs,learning_rate,weight_decay,batch_size)
    train_lsum = train_ls[-1]#仅看最后一轮的训练损失和验证损失
    valid_lsum = valid_ls[-1]
    d2l.plot(list(range(1,epochs+1)),[train_ls,valid_ls],xlabel=epochs,ylabel='loss',xlim=[1,epochs],legend=['train','valid'],yscale='log')
    d2l.plt.show()
    return train_lsum,valid_lsum
#train_l,valid_l=commomtrain(train_features,train_labels,1000,5,0,64)
#print(train_l)
#print(valid_l)

def train_predict(train_features,train_labels,test_features,test_data,num_epochs,learning_rate,weight_decay,batch_size):
    net=get_net()
    #训练
    train_ls,_=train(net,train_features,train_labels,None,None,num_epochs,learning_rate,weight_decay,batch_size)
    train_lsum = train_ls[-1]  # 仅看最后一轮的训练损失和验证损失
    d2l.plot(list(range(1,num_epochs+1)), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    d2l.plt.show()
    print(train_lsum)
    preds=net(test_features).detach()#从计算图中分理出张量
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
train_predict(train_features,train_labels,test_features,test_data,1000,5,0,64)