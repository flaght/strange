# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.

# coding: utf-8

import numpy as np

#昨天收盘价 当天收盘价 当天最高价 当天最低价
class ACD:
    def __init__(self):
        print 'ACD unavailable'
    
    def set_price(self, lc, close, high, low):
        self.lc = lc
        self.close = close
        self.high = high
        self.low = low
    
    def acd(self):
        if self.close > self.lc:
            dift = min(self.low, self.lc)
        elif self.close < self.lc:
            dift = max(self.high, self.lc)
        else:
            dift = 0
        return self.close - dift
    
    def array_acd(self, lc, close, high, low):
        lc_array = np.array(lc)
        close_array = np.array(close)
        high_array = np.array(high)
        low_array = np.array(low)
        dif_array =  close_array - np.where(close_array > lc_array,np.minimum(low_array,lc_array),np.maximum(high_array,lc_array))
        return float(dif_array.sum() / len(dif_array))
