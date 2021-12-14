# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

# + {"code_folding": []}
## deal with dates

#dateM_str = dateM.dt.year.astype(int).astype(str) + \
#             "M" + dateM.dt.month.astype(int).astype(str)
# -

InfCPICMRT.index = pd.DatetimeIndex(dateM_cpic,freq='infer')
InfCPIMRT.index = pd.DatetimeIndex(dateM_cpi,freq='infer')


# + {"code_folding": [0]}
def GetRealTimeData(matrix):
    periods = len(matrix)
    real_time = np.zeros(periods)
    for i in range(periods):
        real_time[i] = matrix.iloc[i,i+1]
    return real_time


# + {"code_folding": []}
## generate real-time series 
matrix_cpic = InfCPICMRT.copy().drop(columns=['date','year','month'])
matrix_cpi = InfCPIMRT.copy().drop(columns=['date','year','month'])

real_time_cpic = pd.Series(GetRealTimeData(matrix_cpic) )
real_time_cpi =  pd.Series(GetRealTimeData(matrix_cpi) ) 
real_time_cpic.index =  pd.DatetimeIndex(dateM_cpic,freq='infer')
real_time_cpi.index = pd.DatetimeIndex(dateM_cpi,freq='infer')
# -
real_time =pd.concat([real_time_cpic,real_time_cpi], join='inner', axis=1)
real_time.columns=['RTCPI','RTCPICore']

real_time
