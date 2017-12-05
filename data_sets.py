# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.
import numpy as np
import pandas as pd
import os
import math

class GFDataSet(object):
    def __init__(self):
       self._x_train = None
       self._y_train = None
       self._x_test = None
       self._y_test = None
       self._index_in_epoch = 0 #序列是从0开始的
       self._range = 4

    def __calcu_feature(self,filename): 
        data = pd.read_csv(filename)
       #买卖盘报价差
        data['absp_diff'] = np.fabs(data['pasp'].values - data['pabp'].values)
        data['lbsp_diff'] = np.fabs(data['plsp'] - data['plbp'])


        #买卖盘平均价格
        data['absp_avg'] = (data['pasp'] + data['pabp']) / 2
        data['lbsp_avg'] = (data['plsp'] + data['plbp']) / 2
        
        #买卖盘深度
        data['lbs_deep'] = (data['plbv'] + data['plsv'])
        data['abs_deep'] = (data['pabv'] + data['pasv'])

        #委买委卖量之差
        #data['lbs_ratio'] = data.apply(lambda x: 0 if math.fabs(x['plsv']) < 0.000001 else math.log(x['plbv']/x['plsv']))
        #data['abs_ratio'] = data.apply(lambda x: 0 if math.fabs(x['pasv']) < 0.000001 else math.log(x['pabv']/x['pasv']))

        data['lbsv_diff'] = np.fabs(data['plsv'] - data['plbv'])
        data['absv_diff'] = np.fabs(data['pasv'] - data['pasv'])
        
        #计算涨跌幅:(close - open) / open * 100%
        t_close = data['close'][1:]
        y_close = data['close'][:-1]
        risk_fail = ((t_close.values - y_close.values) / y_close.values) / 100
        first = np.array([0.])
        risk_fail = np.concatenate((first, risk_fail),axis =0)
        data['risk_fail'] = risk_fail
        return data

    #训练集和测试集
    def __calcu_model_data(self,data):
        #买卖盘数据采用最后记录
        data = data.drop(['Unnamed: 0','date','time','mtd','pabp','pabv','pasp','pasv','absp_diff','absp_avg','absv_diff','abs_deep'],1)
        n = data.shape[0]
        p = data.shape[1]

        data = data.values
        train_start = 0
        train_end = int(np.floor(0.8 * n))

        test_start = train_end + 1
        test_end = n
        data_train = data[np.arange(train_start,train_end),:]
        data_test = data[np.arange(test_start, test_end),:]

        #构建X and Y
        self._x_train = data_train[:,:-1] #初步构建，还需要根据输入情况进一步构建
        self._y_train = data_train[:,-1]
        
        self._x_test = data_test[:,:-1]
        self._y_test = data_test[:,-1]

    #获取训练集
    def train_batch(self):
        while self._index_in_epoch < self._x_train.shape[0] - self._range - 1:
            x_train = self._x_train[self._index_in_epoch:self._index_in_epoch + self._range]
            y_train = self._y_train[self._index_in_epoch + self._range + 1]
            self._index_in_epoch +=1
            yield [x_train,y_train]

    def train_count(self):
        return self._x_train.shape[0] - 4

    def calcu_etf(self, filename):
        data = self.__calcu_feature(filename)
        self.__calcu_model_data(data)

class DataSets(object):
    def __init__(self):
        self.gf_datasets = []

    def gf_etf(self, dir):
        for path, dirs, fs in os.walk(dir):
            for f in fs:
                data_set = GFDataSet()
                data_set.calcu_etf(os.path.join(path,f))
                self.gf_datasets.append(data_set)
                #print os.path.join(path, f)

    def gf_train_count(self):
        return len(self.gf_datasets)

    def gf_train_batch(self):
        return iter(self.gf_datasets)

if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    
    datasets = DataSets()
    datasets.gf_etf('./data/out_dir')
    #print datasets.gf_train_batch()
    i = 0
    
    train_batchs = datasets.gf_train_batch()
    while i < datasets.gf_train_count():
        dataset = train_batchs.next()
        j = 0
        print dataset
        while j < dataset.train_count() - 1:
            data_train = dataset.train_batch().next()
            print data_train
            print('i %d--j %d'%(i,j))
            j+=1
        i+=1
    '''
    data_set = GFDataSet()
    filename = './data/out_dir/ag1606_20160104.csv'
    out_dir = './data/temp/'
    data_set.calcu_etf(filename)
    i = 0
    while i < data_set.train_count() - 1:
        data_train  = data_set.train_batch().next()
        print '<------->'
        print data_train[0]
        print data_train[1]
        #print dict.get('x')
        i += 1
        #print x_train
        #print '---------->'
        #print y_train
        #print 'end'
    '''
    #x_train,y_train = data_set.train_next_batch()
    #print x_train
    #print '------>'
    #print y_train
    #print  X_train[0:4]
    #file_object = open('file.txt','w')
    #file_object.write(str(data))
    #file_object.close()

    #print data
    #data.to_csv(out_dir + "/" + os.path.split(filename)[-1].split('.')[0]+"_gf_feature.csv")
