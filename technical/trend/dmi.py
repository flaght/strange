# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.
# coding: utf-8

import numpy as np
import talib

class DMI:
    def __init__(self):
        print 'DMI available'
        talib.ADX

    def set_array(self,high, low, close, timeperiod=14):
        self.high_array = np.array(high)
        self.low_array = np.array(low)
        self.close_array = np.array(close)
        self.timeperiod = timeperiod

    def adx(self):
        return talib.ADX(self.high_array, self.low_array, self.close_array,self.timeperiod)

    def adxr(self):
        return talib.ADXR(self.high_array, self.low_array, self.close_array, self.timeperiod)
