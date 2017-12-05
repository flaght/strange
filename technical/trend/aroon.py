# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.
# coding: utf-8
# 将资金流动情况与价格行为相对比，检测市场中资金流入和流出的情况
# doc: http://note.youdao.com/noteshare?id=b001e9111d8e27fed598df3996abf704

import talib
import numpy as np

class AROON:
    def __init__(self):
        print 'AROON available'

    @classmethod
    def __ta_array_aroon(cls, high, low, timeperiod):
        high_array = np.array(high)
        low_array = np.array(low)
        return talib.AROON(high_array,low_array,timeperiod)

    @classmethod
    def result(cls, high, low, timeperiod = 14):
        return cls.__ta_array_aroon(high, low, timeperiod)
