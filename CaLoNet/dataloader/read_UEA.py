import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt


def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    #swapaxes的用法就是交换轴的位置，前后两个的位置没有关系。


def load_UEA(archive_name, args):
    # train_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff','r',encoding='UTF-8'))[0]
    # test_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TEST.arff','r',encoding='UTF-8'))[0]

    # load from cache
    cache_path = f'{args.cache_path}/{archive_name}.dat'##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)


    # load from arff
    else:
        train_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
        test_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

        train_x, train_y = extract_data(train_data)##y为标签
        test_x, test_y = extract_data(test_data)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

        scaler = StandardScaler()
        scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

        # 放到0-Numclass
        labels = np.unique(train_y)##标签
        num_class = len(labels)
        # print(num_class)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    # #图结构获取
    # f_g_train, f_g_test = [], []
    # print(train_x.shape)  #
    # #input()
    # #printtt(type(x_train))
    # for data in train_x:
    #     # printtt(1)
    #     G = graph_feature(data)
    #     f_g_train.append(G)
    #
    # for data in test_x:
    #     G = graph_feature(data)
    #     f_g_test.append(G)
    printt('train_x',train_x.shape)#(40, 100, 6)
    f_g_train = np.loadtxt('G:/桌面/calonet多变量mf/TEgraph/{0}/{1}_train.txt'.format(archive_name, archive_name), delimiter=','). \
        reshape((train_x.shape[0], train_x.shape[2], train_x.shape[2]))

    f_g_test = np.loadtxt('G:/桌面/calonet多变量mf/TEgraph/{0}/{1}_test.txt'.format(archive_name, archive_name), delimiter=','). \
        reshape((test_x.shape[0], test_x.shape[2], test_x.shape[2]))
    f_g_test = np.array(f_g_test)
    f_g_train = np.array(f_g_train)
    printt('生成的图结构',f_g_train.shape)
   # input()
    # TrainDataset = DealDataset(train_x, train_y)
    # TestDataset = DealDataset(test_x, test_y)
    TrainDataset = subDataset(train_x, f_g_train, train_y)
    TestDataset = subDataset(test_x, f_g_test, test_y)

    # return TrainDataset,TestDataset,len(labels)
    #DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    #dataset (Dataset) – 决定数据从哪读取或者从何读取；
    #batchszie：批大小，决定一个epoch有多少个Iteration；
    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    return train_loader, test_loader, num_class
from model.layer import printt
from Teoriginal1 import getTEgraph1
def graph_feature(data):
    x = data
    # print(x.shape)  # 275
    # x = x.reshape(n, -1)
    # print('x', x.shape)  # 5,55

    x = getTEgraph1(x)
    #print('123x', x.shape)
    # G = {'feature': data, 'graph': x}
    # G = {}
    # G['feature'] = torch.tensor(data).cuda()
    # G['graph'] = torch.tensor(x).cuda()
    return x

class subDataset1(Dataset):
	def __init__(self, Feature_1,Feature_2,Label):
		self.Feature_1 = torch.from_numpy(Feature_1)
		self.Feature_2 = torch.from_numpy(Feature_2)
		self.Label = Label
	def __len__(self):
		return len(self.Label)
	def __getitem__(self,index):
		Feature_1 = torch.Tensor(self.Feature_1[index])
		Feature_2 = torch.Tensor(self.Feature_2[index])
		Label = torch.Tensor(self.Label[index])
		return Feature_1,Feature_2,Label
class subDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x,z, y):
        self.x_data = torch.from_numpy(x)
        self.graph = torch.from_numpy(z)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]
        self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        return self.x_data[index] ,self.graph[index],self.y_data[index]

    def __len__(self):
        return self.len
class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x, y):
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]
        self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def num_class(self):
        return len(set(self.y_data))


if __name__ == '__main__':
    load_UEA('Ering')
