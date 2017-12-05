# -*- coding: utf-8 -*-

# Copyright 2017 The Strange Authors. All Rights Reserved.

# coding: utf-8

import numpy as np

#低于-0.5时为低风险区，高于+0.5时为高风险区
#开盘价  昨日收盘价 最高价 昨日开盘价

class ADTM:
    def __init__(self):
        print 'ADTM unavailable'

    def set_price(self, open, lc, high, lo):
        self.open = open 
        self.lc = lc
        self.high = high
        self.lo = lo

    def adtm(self):
        if self.open > self.lo:
            self.dtm = max((self.high - self.open),(self.open - self.lo))
        else:
            self.dtm = 0

        if self.open < self.lo:
            self.dbm = max(self.open - self.low, self.lo - self.open)
        else:
            self.dbm = 0

        self.stm = self.dtm
        self.sbm = self.dbm

        if self.stm > self.sbm:
            self.adtm = (self.stm - self.sbm) / self.stm
        elif self.stm < self.sbm:
            self.adtm = (self.stm - self.sbm) / self.sbm
        elif self.stm == self.sbm:
            self.adtm = 0

        return self.adtm

    def array_adtm(self, open, low, high, lo):
        open_array = np.array(open)
        low_array = np.array(low)
        high_array = np.array(high)
        lo_array = np.array(lo)


        dtm_array = np.where(open_array > lo_array, np.maximum(high_array - open_array,open_array - lo_array), 0)

        dbm_array = np.where(open_array < lo_array, np.maximum(open_array - low_array,lo_array - open_array), 0)

        stm = float(dtm_array.sum())
        sbm = float(dbm_array.sum())
        if stm > sbm:
            adtm = float(stm - sbm) / float(stm)
        elif stm < sbm:
            adtm = float(stm - sbm) / float(sbm)
        else:
            adtm = 0
        return adtm
