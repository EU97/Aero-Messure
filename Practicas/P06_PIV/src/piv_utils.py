import numpy as np
import openpiv.process as process
import openpiv.validation as validation
import openpiv.filters as filters

def simple_piv(a, b, window=32, overlap=16, dt=0.01, search=64):
    u,v,s2n = process.extended_search_area_piv(a,b, window_size=window, overlap=overlap, dt=dt, search_area_size=search)
    u,v,mask = validation.sig2noise_val(u,v,s2n, threshold=1.3)
    u,v = filters.replace_outliers(u,v, method='localmean', max_iter=3, kernel_size=2)
    return u,v
