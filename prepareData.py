import pandas as pd
import numpy as np
def readDataSet(fileName,rate=0.9):
    '''
    读取数据集
    :param fileName: 文件名
    :param rate: 训练数据集和测试数据集比例，默认为九一开
    :return: 训练数据集、训练标签、测试数据集、测试标签
    '''

    file = pd.read_csv("./dataset/"+fileName)
    f = np.array(file)
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    total_number = f.shape[0]
    split_number  = int(total_number*rate)

    for index,line in enumerate(f):
        if index <= split_number:
            train_data.append(line[:-1])
            train_label.append(line[-1])
        else:
            test_data.append(line[:-1])
            test_label.append(line[-1])
    train_data = np.array(train_data)
    test_data = np.array(test_data)

    return train_data,train_label,test_data,test_label


def getEachDelimiter(train_data,train_label,attribute):
    '''
    获得每个属性的分隔符，其中attribute表示每个属性的状态，在此处，attribute为1表示是离散数据，直接获取，
    attribute为2表示是连续数据，需要排序后取得所有分割点。
    :param train_data: 训练数据
    :param attribute:  表示每个属性的状态
    :return: 训练数据的分割符、训练标签的种类
    '''

    train_data_delimiter = []
    train_label_delimiter = list(set(train_label)) # 获得标签

    DISPERSE = 1
    CONTINUITY = 2

    train_attribute_number = train_data.shape[-1]

    for index in range(train_attribute_number):
        if attribute[index] == DISPERSE:
            # 是离散数据就直接用set
            s = list(set(train_data[:,index]))
            # print(s)
            train_data_delimiter.append(s)
        elif attribute[index] == CONTINUITY:
            # 是连续数据就排序取不同
            data = train_data[:,index]
            data = sorted(list(set(data)))
            length = len(data)
            ans = []
            for i in range(length-1):
                ans.append((data[i]+data[i+1])/2)
            train_data_delimiter.append(ans)
            # print(ans)

    print(train_data_delimiter[3])
    return train_data_delimiter,train_label_delimiter