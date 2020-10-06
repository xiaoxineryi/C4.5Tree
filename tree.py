import prepareData as p
from math import log
import numpy as np
# 全局变量
USED = 0
DISPERSE = 1
CONTINUITY = 2


# 计算信息熵
def calEnt(labels):
    numberEnt = len(labels)
    labelCounts = {}

    for label in labels:
        if label not in labelCounts.keys():
            labelCounts[label] = 0
        labelCounts[label] +=1

    Ent = 0
    for key in labelCounts:
        P = float(labelCounts[key])/numberEnt
        Ent = Ent - P*log(P,2)
    return Ent

# 挑选出对应列中值为特定属性的数据，用于计算选择某个特征后的信息熵
def splitDispersedData(dataset,labels,column_number,value):
    '''
    :param data:  数据
    :param column_number: 对应要去除的列的标号
    :param value:   要去除的数据值
    :return:  对应位置为特定数据量的数据
    '''

    returnData = [] # 返回的数据集列表
    returnLabels = []
    for index,data in enumerate(dataset):
        if data[column_number] == value:
            d = data[:column_number]
            d = np.hstack((d,data[column_number+1:]))
            returnData.append(d)
            returnLabels.append(labels[index])
    return returnData,returnLabels

def getDispersedGainRatio(dataset,labels,column_number,delimiter):
    '''
    :param delimiter:   这个列属性的分割符
    :return: 离散数据对应列的信息增益率
    '''
    # 计算初始数据的信息熵
    datasetLength = len(dataset)
    baseEnt = calEnt(labels)
    IV = 0
    Ent = 0
    # 计算划分后的各个信息熵
    for value in delimiter:
        subDataset,subLabels = splitDispersedData(dataset,labels,column_number,value)
        P = float(len(subDataset))/datasetLength
        if P ==0:
            continue
        Ent += P * calEnt(subLabels)
        IV = IV - P*log(P,2)
    if IV==0:
        return 1
    infoGain = baseEnt - Ent
    infoGain_ratio = infoGain / IV

    return infoGain_ratio


def splitContinuityData(dataset,labels,column_number,threshold):
    lessDataset = []
    moreDataset = []
    lessLabels = []
    moreLabels = []

    for index,data in enumerate(dataset):
        if data[column_number] < threshold:
            lessDataset.append(data)
            lessLabels.append(labels[index])
        else:
            moreDataset.append(data)
            moreLabels.append(labels[index])
    return lessDataset,lessLabels,moreDataset,moreLabels


def getContinuityGainRatio(dataset,labels,column_number,threshold):
    '''
    返回连续数据在对应阈值下的信息增益率
    '''
    lessDataset,lessLabels,moreDataset,moreLabels = \
        splitContinuityData(dataset,labels,column_number,threshold)
    # 计算一开始的信息熵
    baseEnt = calEnt(labels)

    lessP = float(len(lessDataset))/len(dataset)
    moreP = float(len(moreDataset))/len(dataset)

    IV = 0
    Ent = 0

    # 如果可以全部区分的话 就直接返回1
    if (lessP == 0 and len(lessDataset)>10 )or (moreP == 0 and len(moreDataset)>10):
        return 1
    elif lessP ==0 or moreP ==0:
        return 0
    Ent += lessP*calEnt(lessLabels)
    Ent += moreP*calEnt(moreLabels)
    IV  = IV - lessP*log(lessP,2)
    IV = IV - moreP*log(moreP,2)


    infoGain = baseEnt - Ent
    infoGain_ratio = infoGain/IV

    return infoGain_ratio

# 选择最佳特征
def chooseBestFeature(dataset,labels,attribute,delimiter):

    numFeatures = len(dataset[0])
    bestInfoGain_ratio = 0
    bestFeature = -1
    # 这个表示如果是连续型变量的话，就记录切分点
    bestThreshold = -1
    for index in range(numFeatures):
        # 如果是离散型的，那么就调用直接获取
        if attribute[index] == DISPERSE:
            InfoGain_ratio = getDispersedGainRatio(dataset,labels,index,delimiter[index])
            if InfoGain_ratio > bestInfoGain_ratio:
                bestInfoGain_ratio = InfoGain_ratio
                bestFeature = index
        # 如果是连续型的，那么就遍历他所有对应的分隔符,取最佳
        elif attribute[index] == CONTINUITY:
            d = delimiter[index]
            for threshold in d:
                InfoGain_ratio = getContinuityGainRatio(dataset,labels,index,threshold)
                if InfoGain_ratio > bestInfoGain_ratio:
                    bestInfoGain_ratio = InfoGain_ratio
                    bestFeature = index
                    bestThreshold = threshold
        # 如果是USED 就直接不理会


    print(bestFeature,bestInfoGain_ratio)
    if attribute[bestFeature] ==CONTINUITY:
        print(bestThreshold)
    return bestFeature,bestThreshold


def createDecisionTree(dataset,labels,attribute,delimiter,depth,l):
    # 如果深度够了，就直接返回，防止过拟合
    if depth >=5:
        return None
    # 如果所有的属性都被标记为已使用，即全为0,就说明没有可供选择的了，直接返回
    if sum(attribute) ==0:
        return None
    # 如果已经是最有解了，就不继续了
    if len(set(labels)) ==1:
        return None
    if dataset == []:
        return None
    # 没有问题就直接选取最佳特征
    bestFeature, bestThreshold = chooseBestFeature(dataset,labels,attribute,delimiter)

    # 递归创建树
    C45Tree = {l[bestFeature]: {}}

    # 区分连续型变量和离散型变量
    # 如果是离散型变量，就直接设置为已使用，并且直接递归
    if attribute[bestFeature] == DISPERSE:
        attribute[bestFeature] = USED
        for value in delimiter[bestFeature]:
            # 获得对应属性值的数据
            subDataset,subLabels = splitDispersedData(dataset,labels,bestFeature,value)
            C45Tree[l[bestFeature]][value] = createDecisionTree(subDataset,subLabels,attribute,delimiter,depth+1,l)
    # 如果是连续型变量，就递归
    elif attribute[bestFeature] ==CONTINUITY:
        lessDataset,lessLabels,moreDataset,moreLabels = splitContinuityData(dataset,labels,bestFeature,bestThreshold)

        lessDelimiter = delimiter.copy()
        moreDelimiter = delimiter.copy()

        less = lessDelimiter[bestFeature]
        more = moreDelimiter[bestFeature]
        delimiterLength = len(delimiter[bestFeature])
        less = [less[i] for i in range(delimiterLength) if less[i]<bestThreshold]
        more = [more[i] for i in range(delimiterLength) if more[i]>bestThreshold]

        lessDelimiter[bestFeature] = less
        moreDelimiter[bestFeature] = more

        C45Tree[l[bestFeature]][bestThreshold-0.01] = createDecisionTree(lessDataset,lessLabels,attribute,lessDelimiter,depth+1,l)
        C45Tree[l[bestFeature]][bestThreshold+0.01] = createDecisionTree(moreDataset,moreLabels,attribute,moreDelimiter,depth+1,l)

    return C45Tree





