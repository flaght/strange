# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.
# coding: utf-8
# ASI 指出股价短期的趋向,可以提早一步确定股价的走势
# doc: http://note.youdao.com/noteshare?id=daf0b8c96559145bddd99d2e1726caf6 
import numpy as np

#当日最低价  当日最高价 当日收盘价 当日开盘价  昨日收盘价 昨日最低价 昨日开盘价
class ASI:
    def __init__(self):
        print 'ASI unavailable'
    
    @classmethod
    def array_asi(cls, low, high, close, open, lc, ll, lo):
        low_array = np.array(low)
        high_array = np.array(high)
        close_array = np.array(close)
        open_array = np.array(open)
        lc_array = np.array(lc)
        ll_array = np.array(ll)
        lo_array = np.array(lo)

        aa = abs(high_array - lc_array)
        bb = abs(low_array - lc_array)
        cc = abs(high_array - ll_array)
        dd = abs(lc_array - lo_array)

        abc = np.maximum(cc,np.maximum(aa,bb))
        
        #print np.where(abc == aa,44,np.where(abc==bb,77,88))
        
        rr = np.where(abc==aa, aa + bb/2 + dd/4, np.where(abc==bb,aa/2 + dd/4, cc+dd/4))
        
        ee = close_array - lc_array
        ff = close_array - open_array
        gg = lc_array - lo_array
        xx = ee + ff/2 + gg
        si = np.true_divide(16 * xx,rr) * np.maximum(aa,bb)

        return si.sum()
