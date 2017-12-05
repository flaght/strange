# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.
# coding: utf-8
# AD 平衡交易量指标，以当日的收盘价位来估算成交量。用于测量一段时间内个股累积的资金流量
# doc: http://note.youdao.com/noteshare?id=b4c9b888f3623297a67d62cd019733f2

import talib
import numpy as np
class AD:
    def __init__(self):
        print 'AD available'

    def set_price(self, high, low, close, volume):
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    def ad(self):
        return talib.AD(self.high,self.low,self.close,self.volume)

    @classmethod
    def __ta_array_ad(cls, high, low, close, volume):
        high_array = np.array(high)
        low_array = np.array(low)
        close_array = np.array(close)
        volume_array = np.array(volume)

        return talib.AD(high_array, low_array, close_array, volume_array)
    
    @classmethod
    def result(cls, high, low, close, volume):
        return cls.__ta_array_ad(high, low, close, volume)
