# InfVar 

- the replication repo of the research project of density forecasts: How Do Agents Form Macroeconomic Expectations? Evidence from the Inflation Uncertainty
- Tao Wang
- Bank of Canada 

## [Most recent draft](/InfVar.pdf)

## Data sources
- [__Survey of Professional Forecasters__](https://www.philadelphiafed.org/surveys-and-data/data-files)
- [__Survey of Consumer Expectations__](https://www.newyorkfed.org/microeconomics/sce/background.html)
- [__Real-time data__](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research)
- [__Inflation realizations__](https://fred.stlouisfed.org/)

## Data code

For replication, 
- __either__ mannually download following data and put them into the folder workingfolder/SurveyData/SCE/
    - NYFED_SCE_13_16.dta: https://www.dropbox.com/s/rqzo1ypb08no5by/NYFED_SCE_13_16.dta?dl=0 
    - NYFED_SCE_17_19.dta: https://www.dropbox.com/s/964vt24chjk3yxh/NYFED_SCE_17_19.dta?dl=0 
    - NYFED_SCE_20.dta: https://www.dropbox.com/s/gg2j47c0wwike7b/NYFED_SCE_20.dta?dl=0 

- __or__ run the [Python code](/workingfolder/python/DownloadSCE.ipynb)

## Other Code

- [Stata code](/workingfolder/DoFile)
    - [Cleaning inflation data](/workingfolder/DoFile/Step00_InflationData.do)
    - [Cleaning SCE micro data](/workingfolder/DoFile/Step01_CleaningSCE%26hist.do)
    - [CLeaning SPF micro data](/workingfolder/DoFile/Step02_CleaningSPF.do)
    - [Cleaning SPF population probablistic data](/workingfolder/DoFile/Step02b_CleaningSPF_Prob.do)
    - [Cleaning SPF individual probablistic data](/workingfolder/DoFile/Step02c_CleaningSPF_IndProb.do)
    - [Cleaning SPF summary statistics](/workingfolder/DoFile/Step02d_CleaningSPFSumStat_IndDst.do)
    - [Population regression analysis for SPF](/workingfolder/DoFile/Step03a_PopAnalysisQ.do)
    - [Population regression analysis for SCE](/workingfolder/DoFile/Step03b_PopAnalysisSCEM.do)
    - [Individual regression analysis for SPF](/workingfolder/DoFile/Step05a_IndSPFAnalysis.do)
    - [Individual regression analysis for SCE](/workingfolder/DoFile/Step05b_IndSCEAnalyais.do)
 
- [Python](/workingfolder/python)
  - [Density Estimation of SPF](/workingfolder/python/DoDensityEst.ipynb), which draws the model class from 
     - [DensityEstimation](/workingfolder/python/DensityEst.py)

  - [Preparing real-time inflation data](/workingfolder/python/RealTimeDataAnalytics.ipynb)

  - [Estimating stochastic volatility of inflation (in Matlab)](/workingfolder/python/DoStockWatsonEst.m)
  
  - [Structural estimation](/workingfolder/python/DoSMMEst.ipynb), which imports model classes from 
     - [Model Class](/workingfolder/python/SMMEst.ipynb), which tests if each model could correctly identify model parameters using various moments. 
    - [Structural estimation for extended sample](/workingfolder/python/DoSMMEst-after2022.ipynb)
   

```python

```
