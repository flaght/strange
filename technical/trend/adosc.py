# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.
# coding: utf-8
# 将资金流动情况与价格行为相对比，检测市场中资金流入和流出的情况
# doc: http://note.youdao.com/noteshare?id=6fe1b1481c8579a91e0376c5f9819bc3

import talib
import numpy as np
class ADOSC:
    def __init__(self):
        print 'ADOSC available'

    @classmethod
    def __ta_array_adosc(cls, high, low, close, volume, fastperiod, slowperiod):
        high_array = np.array(high)
        low_array = np.array(low)
        close_array = np.array(close)
        volume_array = np.array(volume)

        return talib.ADOSC(high_array, low_array, close_array, volume_array,fastperiod,slowperiod)

    @classmethod
    def result(cls, high, low, close, volume, fastperiod = None, slowperiod = None):
        return cls.__ta_array_adosc(high, low, close, volume, fastperiod, slowperiod)
