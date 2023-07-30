# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Real Time Data 
#
# - [link](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/real-time-data-set-full-time-series-history)

import pandas as pd
import numpy as np
import datetime as dt


# + {"code_folding": [0]}
def RT_this_date(real_time_data,
                 date,
                 date_freq='M',
                vintage_freq='Q',
                variable ="EMPLOY"):
    """
    input
    ======
    real_time_date: panda dataframe with row index being the date and column index bing the vintage 
    date: date for which the up-to-that-date real time data is extracted 
    
    output
    ======
    a pandas series of real-time data till the date indexed by date 
    """
    if vintage_freq =='M':
        date = date
    elif vintage_freq =='Q':
        date = date
    print(date)
    vintage_names = [var for var in real_time_data.columns if variable in var]
    vintage_dates = [var.replace(variable, '') for var in vintage_names]
    print('Note: the earliest date is',str(vintage_dates[0]))
    
    
    ## for the date find the vintage date, M or Q

    
    ## look for that date 
    if vintage_freq =='M': 
        date_m_forward = date+pd.DateOffset(months=1)
        month = date_m_forward.month
        year = date_m_forward.year
        year2d = year-year//100*100
        if year2d<10:
            year2d_str = '0'+str(year2d)
        else:
            year2d_str = str(year2d)
        date_str = year2d_str+'M'+str(month)
    elif vintage_freq == 'Q':
        date_m_back = date-pd.DateOffset(months=1) ## move month by 1
        date_q = date_m_back.to_period('Q')+1   ## move quarter forward by 1
        quarter = date_q.quarter
        year = date_q.year
        year2d = year-year//100*100
        if year2d<10:
            year2d_str = '0'+str(year2d)
        else:
            year2d_str = str(year2d)
        date_str = year2d_str+'Q'+str(quarter)
    if date_str in vintage_dates:
        which_vintage = vintage_dates.index(date_str)
    else:
        which_vintage = 0
    which_vintage_name = vintage_names[which_vintage]
    print('vintage:',str(which_vintage_name))
    #print(which_vintage_name)
    real_time_data_this_date = real_time_data[which_vintage_name]
    
    return real_time_data_this_date

# + {"code_folding": []}
## get only the contemparaneous data point (last obs of the real-time data)

if __name__ == "__main__":
    ### CPI 

    CPIMRT = pd.read_excel('../OtherData/RealTimeData/pcpiMvMd.xlsx')  
    #https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcpi 
    CPIMRT = CPIMRT[-CPIMRT.DATE.isnull()]
    dateM_CPI  = pd.to_datetime(CPIMRT['DATE'],
                                format='%Y:%m', 
                               # errors='coerce'
                               )

    CPIMRT.index = dateM_CPI
    CPIMRT = CPIMRT.drop(columns=['DATE'])


    ## get only the contemparaneous data point (last obs of the real-time data)

    std_CPI = pd.Timestamp('1967-01-01')

    CPI_RT_now = pd.DataFrame(columns = ['CPI'], 
                              index= CPIMRT.index[CPIMRT.index>std_CPI])


    for date in CPI_RT_now.index:
        ## get all the historical real-time date till date
        this_RT_till_now = RT_this_date(CPIMRT,
                                       date,
                                      'M',
                                      'M',
                                      'PCPI')
        ## then only get the observation for the date 
        this_RT_now = this_RT_till_now.loc[date]

        ## put it in the RT_now dataframe 
        CPI_RT_now['CPI'].loc[date] = this_RT_now
# -

CPICMRT

# +
## get only the contemparaneous data point (last obs of the real-time data)

if __name__ == "__main__":
    ### CPI Core

    CPICMRT = pd.read_excel('../OtherData/RealTimeData/pcpixMvMd.xlsx')  
    #https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcpix
    CPICMRT = CPICMRT[-CPICMRT.DATE.isnull()]
    dateM_CPIC  = pd.to_datetime(CPICMRT['DATE'],
                                format='%Y:%m', 
                               # errors='coerce'
                               )

    CPICMRT.index = dateM_CPIC
    CPICMRT = CPICMRT.drop(columns=['DATE'])


    ## get only the contemparaneous data point (last obs of the real-time data)

    std_CPIC = pd.Timestamp('1967-01-01')

    CPIC_RT_now = pd.DataFrame(columns = ['CPIC'], 
                              index= CPICMRT.index[CPICMRT.index>std_CPIC])


    for date in CPIC_RT_now.index:
        ## get all the historical real-time date till date
        this_RT_till_now = RT_this_date(CPICMRT,
                                       date,
                                      'M',
                                      'M',
                                      'PCPIX')
        ## then only get the observation for the date 
        this_RT_now = this_RT_till_now.loc[date]

        ## put it in the RT_now dataframe 
        CPIC_RT_now['CPIC'].loc[date] = this_RT_now

# +
## export data
RT_now_all = pd.merge(CPI_RT_now,
                      CPIC_RT_now,
                      left_index = True,
                      right_index = True,
                     how ='outer')


RT_now_all.to_excel('../OtherData/RealTimeData/RealTimeInfQ.xlsx')
# -
RT_now_all

