# This is a sample Python script.


import prepareData
import tree
import numpy as np



if __name__ == '__main__':
    train_data,train_label,test_data,test_label = prepareData.readDataSet("heart.csv",rate=1)
    attribute = [2,1,1,2,2,1,1,2,1,2,1,1,1]
    l = ["A","B","C","D","E","F","G","H","I","G","K","L","M","N"]
    train_data_delimiter,train_label_delimiter = prepareData.getEachDelimiter(train_data,train_label,attribute)
    # bestFeature,bestThreshold = tree.chooseBestFeature(train_data,train_label,attribute,train_data_delimiter)
    C45Tree = tree.createDecisionTree(train_data,train_label,attribute,train_data_delimiter,0,l)
    print(C45Tree)