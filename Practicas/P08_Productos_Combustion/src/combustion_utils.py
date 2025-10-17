import numpy as np
import pandas as pd

def afr_from_o2(o2_vol):
    # Simplificada: relación con lambda
    return 14.7/(1+1e-6*(20.9 - o2_vol))
