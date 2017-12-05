# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.
# coding: utf-8
# ATR 显示市场变化率的指标。主要用来衡量价格的波动.能表明价格波动的程度,不能反映价格走向及趋势的稳定性
# doc: http://note.youdao.com/noteshare?id=cde731ed427721ad6219b9ada0541f0f

import numpy as np
import talib
#当日最高价 当日最低价 前一个收盘价

class ATR:
    def __init__(self):
        print 'ATR available'
 
    @classmethod
    def __ta_array_atr(cls,high,low,close):
        high_array = np.array(high)
        low_array = np.array(low)
        close_array = np.array(close)
        return talib.ATR(high_array, low_array, close_array)

    @classmethod
    def result(cls, high, low, close):
        return cls.__ta_array_atr(high, low, close)
