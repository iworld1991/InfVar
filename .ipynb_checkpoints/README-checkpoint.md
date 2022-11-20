# InfVar 

- the working repo of the research project of density forecasts: How Do Agents Form Inflation Expectations? Evidence from the Forecast Uncertainty
- Tao Wang
- Johns Hopkins University 

## [Most recent draft](/InfVar.pdf)
## Data 

For replication, 
- __either__ mannually download following data and put them into the folder workingfolder/SurveyData/SCE/
    - NYFED_SCE_13_16.dta: https://www.dropbox.com/s/rqzo1ypb08no5by/NYFED_SCE_13_16.dta?dl=0 
    - NYFED_SCE_17_19.dta: https://www.dropbox.com/s/964vt24chjk3yxh/NYFED_SCE_17_19.dta?dl=0 
    - NYFED_SCE_20.dta: https://www.dropbox.com/s/gg2j47c0wwike7b/NYFED_SCE_20.dta?dl=0 

- __or__ run the [Python code](/workingfolder/python/DownloadSCE.ipynb)

## Code

- [Stata code](/workingfolder/DoFile)

- [Density Estimation of SPF](/workingfolder/python/DoDensityEst.ipynb), which draws the model class from 
   - [DensityEstimation](/workingfolder/python/DensityEst.py)

- [Python](/workingfolder/python)
  - [SMM estimation](/workingfolder/python/DoSMMEst.ipynb), which imports model classes from 
     - [Model Class](/workingfolder/python/SMMEst.ipynb), which tests if each model could correctly identify model parameters using various moments. 

```python

```
