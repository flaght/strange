# -*- coding: utf-8 -*-
# Copyright 2017 The Strange Authors. All Rights Reserved.

# coding: utf-8

import numpy as np
class ARBR:
    def __init__(self):
        print 'ARBR available'

    def set_array(self, high, open, low, lc):
        self.high_array = np.array(high)
        self.low_array = np.array(low)
        self.open_array = np.array(open)
        self.lc_array = np.array(lc)

    def array_ar(self):
        hc_array = self.high_array - self.open_array
        ol_array = self.open_array - self.low_array
        return float(hc_array.sum() / ol_array.sum()) * 100

    def array_br(self):
        hlc_array = self.high_array - self.lc_array
        lcl_array = self.lc_array - self.low_array
        return float(hlc_array.sum() / lcl_array.sum()) * 100
