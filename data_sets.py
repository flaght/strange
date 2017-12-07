# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.
import numpy as np
import pandas as pd
import os
import math
import pdb
from sklearn.preprocessing import MinMaxScaler


class PDataSet(object):

    def __init__(self, train_step=55, test_step=50):
        self._x_train = []
        self._y_train = []
        self._x_test = []
        self._y_test = []
        self._train_step = train_step
        self._test_step = test_step

    def __calc_model_data(self, data):
        data = data.drop(
            ['Unnamed: 0', 'date', 'mtd', 'pabp', 'pabv', 'pasp', 'pasv', 'plbp', 'plbv', 'plsp', 'plsv', 'time',
             'vol'], 1)
        n = data.shape[0]
        p = data.shape[1]

        data = data.values
        train_start = 0
        train_end = int(np.floor(0.8 * n))

        test_start = train_end + 1
        test_end = n

        data_train = data[np.arange(train_start, train_end), :]
        data_test = data[np.arange(test_start, test_end), :]
        '''
        for i in range(len(data_train) - self._train_step - 5):
            for u in range(self._train_step - 5):
                if u == 0:
                    x = data_train[u+i:u+i+4,:].reshape(-1,20)
                else:
                    t = data_train[u+i:u+i+4,:].reshape(-1,20)
                    x = np.vstack((x,t))
            
            y = data_train[i+5:i+self._train_step,0,np.newaxis]
            self._x_train.append(x)
            self._y_train.append(y)
        '''

        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)
        normalized_test_data = (data_test - np.mean(data_test, axis=0)) / np.std(data_test, axis=0)

        scaler = MinMaxScaler()
        scaler.fit(normalized_train_data)

        normalized_train_data = scaler.transform(normalized_train_data)
        normalized_test_data = scaler.transform(normalized_test_data)

        for i in range(len(data_train) - self._train_step):
            for u in range(self._train_step - 5):
                if u == 0:
                    x = normalized_train_data[u + i:u + i + 4, :].reshape(-1, 16)
                else:
                    t = normalized_train_data[u + i:u + i + 4, :].reshape(-1, 16)
                    x = np.vstack((x, t))

            # x = normalized_train_data[i:i+self._train_step, :]
            y = normalized_train_data[i + 1:i + self._train_step + 1, 0, np.newaxis]
            self._x_train.append(x)
            self._y_train.append(y)

        test_step = self._test_step if (len(data_test) - 1) > self._test_step else (len(data_test) - 1)

        for i in range(len(data_test) - test_step):
            tx = normalized_test_data[i:test_step, :]
            ty = normalized_test_data[i + 1:test_step + 1, 0, np.newaxis]
            self._x_test.append(tx)
            self._y_test.append(ty)

    def train_batch(self):
        return self._x_train, self._y_train

    def test_batch(self):
        return self._x_test, self._y_test

    def calc_etf(self, filename):
        data = pd.read_csv(filename)
        self.__calc_model_data(data)


class GFDataSet(object):

    def __init__(self):
        self._x_train = []
        self._y_train = []
        self._x_test = []
        self._y_test = []
        self._index_in_epoch = 0  # 序列是从0开始的
        self._range = 4

    def __calc_feature(self, filename):
        data = pd.read_csv(filename)
        # 买卖盘报价差
        data['absp_diff'] = np.fabs(data['pasp'].values - data['pabp'].values)
        data['lbsp_diff'] = np.fabs(data['plsp'] - data['plbp'])

        # 买卖盘平均价格
        data['absp_avg'] = (data['pasp'] + data['pabp']) / 2
        data['lbsp_avg'] = (data['plsp'] + data['plbp']) / 2
        # 买卖盘深度
        data['lbs_deep'] = (data['plbv'] + data['plsv'])
        data['abs_deep'] = (data['pabv'] + data['pasv'])

        # 委买委卖量之差
        data['lbsv_diff'] = np.fabs(data['plsv'] - data['plbv'])
        data['absv_diff'] = np.fabs(data['pasv'] - data['pasv'])

        # 计算涨跌幅:(close - open) / open * 100%
        t_close = data['close'][1:]
        y_close = data['close'][:-1]
        risk_fail = ((t_close.values - y_close.values) / y_close.values) / 100
        first = np.array([0.])
        risk_fail = np.concatenate((first, risk_fail), axis=0)
        data['risk_fail'] = risk_fail
        data = data.drop(
            ['Unnamed: 0', 'date', 'time', 'mtd', 'pabp', 'pabv', 'pasp', 'pasv', 'absp_diff', 'absp_avg', 'absv_diff',
             'abs_deep'], 1)
        return data

    # 训练集和测试集
    def __calc_model_data(self, data):
        # 买卖盘数据采用最后记录
        time_step = 50
        n = data.shape[0]
        p = data.shape[1]

        data = data.values
        train_start = 0
        train_end = int(np.floor(0.8 * n))

        test_start = train_end + 1
        test_end = n
        data_train = data[np.arange(train_start, train_end), :]
        data_test = data[np.arange(test_start, test_end), :]

        normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)
        # 构建训练 X and Y
        for i in range(len(data_train) - time_step):
            x = normalized_train_data[i:i + time_step, :-1]
            y = normalized_train_data[i:i + time_step, -1, np.newaxis]
            self._x_train.append(x)
            self._y_train.append(y)

        # 构建测试 X and Y
        test_time_step = test_end - test_start - 1
        normalized_test_data = (data_test - np.mean(data_test, axis=0)) / np.std(data_train, axis=0)

        size = (len(normalized_test_data) + test_time_step - 1) // test_time_step
        for i in range(size - 1):
            t_x = normalized_test_data[i * test_time_step:(i + 1) * test_time_step, :-1]
            t_y = normalized_test_data[i * test_time_step:(i + 1) * test_time_step, -1, np.newaxis]
            self._x_test.append(t_x)
            self._y_test.append(t_y)

    # 获取训练集
    def train_batch(self):
        return self._x_train, self._y_train

    # 获取测试集
    def test_batch(self):
        return self._x_test, self._y_test

    def calc_etf(self, filename):
        data = self.__calc_feature(filename)
        self.__calc_model_data(data)


class DataSets(object):
    def __init__(self):
        self.gf_data_sets = []

    def gf_etf(self, dir):
        for path, dirs, fs in os.walk(dir):
            for f in fs:
                data_sets = GFDataSet()
                data_sets.calc_etf(os.path.join(path, f))
                self.gf_data_sets.append(data_sets)
                # print os.path.join(path, f)

    def gf_train_count(self):
        return len(self.gf_data_sets)

    def gf_train_batch(self):
        return iter(self.gf_data_sets)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    data_set = PDataSet()
    data_set.calc_etf('./data/out_dir/ag1606_20160104.csv')
    train_x, train_y = data_set.train_batch()
    print np.array(train_x[1:2])
    print '----------->'
    print np.array(train_y[1:2])
    # print np.array(train_x[0:1])
    # print '--------->'
    # print np.array(train_y[0:1])
    # data_set = GFDataSet()
    # filename = './data/out_dir/ag1606_20160104.csv'
    # data_set.calcu_etf(filename)
    # test_x,test_y = data_set.test_batchs()
    # train_x,train_y = data_set.train_batchs()
    # print np.array(train_x[0:1]).shape
    # print np.array(train_y[0:1]).shape
    # print np.array(test_x[0:1]).shape
    # print np.array(test_y[0:1]).shape
    # print np.squeeze(np.array(test_x[0:1])).shape
    # train_x,train_y = data_set.calcu_etf(filename)
    # print train_x[0:1]
    # print '--------->'
    # print train_y[0:1]
