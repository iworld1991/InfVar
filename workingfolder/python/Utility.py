# ---
# jupyter:
#   jupytext:
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

# +
## not used for now 

def calc_variance_at_low_freq(series,
                             horizon):
    """
    input
    ===== 
    series: data at higher frequency with serial correlation
    horizon: nb of periods between two non-serially correlated data point
    
    out
    ====
    variance: average variance of the non-serially correlated series 
    """
    
    varinace_sum = 0.0
    for cut in range(0,horizon):
        ids = np.arange(cut,len(series),horizon)
        print(ids)
        variance = series[ids].var()
        varinace_sum +=variance
    return varinace_sum/horizon

def newey_west_variance(series,
                        truncate):
    """
    input
    ===== 
    series: data at higher frequency with serial correlation
    truncate: nb of periods between two non-serially correlated data point
    
    out
    ====
    variance: average variance of the non-serially correlated series 
    """
    adjust_term = 1.0
    
    for truc in range(1,truncate):
        weight = 2*(truncate-truc)/truncate
        adjust_term += weight
    return series.var()/adjust_term
