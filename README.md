# InfVar 

- the replication repo of the research project: How Do Agents Form Macroeconomic Expectations? Evidence from the Inflation Uncertainty
- Tao Wang
- taowang@bank-banque-canada.ca
- Bank of Canada 



# Data sources

Federal Reserve Bank of New York. Survey of Consumer Expectations Complete Microdata (Public). 2013-2016, 2017-2019, 2020-2023. https://www.newyorkfed.org/microeconomics/sce#/ 

Federal Reserve Bank of Philadelphia. Survey of Professional Forecasters (CORECPI): Core CPI Inflation Rate (CORECPI) – Individual responses, 2007-2023. https://www.philadelphiafed.org/surveys-and-data/corecpi 

Federal Reserve Bank of Philadelphia. Survey of Professional Forecasters (COREPCE): Core PCE Inflation Rate (COREPCE) – Individual responses, 2007-2023. https://www.philadelphiafed.org/surveys-and-data/corepce

Federal Reserve Bank of Philadelphia. Real-Time Data Set (CPI), 1967-2022. https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/cpi 

Federal Reserve Bank of Philadelphia. Real-Time Data Set (Core CPI), 1967-2022. https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcpi 

Federal Reserve Bank of Philadelphia. Real-Time Data Set (Core PCE), 1967-2022. https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/pcon 

Federal Reserve Bank of Philadelphia. Survey of Professional Forecasters (Core CPI), 1967-2022. https://www.philadelphiafed.org/surveys-and-data/corecpi 

Federal Reserve Bank of Philadelphia. Survey of Professional Forecasters (Core PCE), 1967-2022. https://www.philadelphiafed.org/surveys-and-data/corepce 

Federal Reserve Bank of St Louis. Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (CPIAUCSL), 2007-2023.  FRED Economic Data. https://fred.stlouisfed.org/series/CPIAUCSL [CPI]

Federal Reserve Bank of St Louis. Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average (CPILFESL), 2007-2023. FRED Economic Data. https://fred.stlouisfed.org/series/CPILFESL [Core CPI]

Federal Reserve Bank of St Louis. Personal Consumption Expenditures Excluding Food and Energy (Chain-Type Price Index) (PCEPILFE), 2007-2023.  FRED Economic Data. https://fred.stlouisfed.org/series/PCEPILFE [Core PCE]


The datasets used for this paper are all publicly accessible without restriction or limitation.

## File structure

- \tables 
  - This folder includes summary tables that report the data information and main results of the paper. They are hand created from the outputs generated from the code.

- \workingfolder
  - This folder contains all the code and data that generate the results of the paper. 

    - \workingfolder\DoFile 
     
        This folder contains the stata do files that clean the data, report the stylized facts about the inflation uncertainty and the reduced-form regression results.  

    - \workingfolder\graphs
        - \workingfolder\graphs\ind
        - \workingfolder\graphs\inflation
        - \workingfolder\graphs\pop
        -  \workingfolder\graphs\pop\hist
    - \workingfolder\ind
    
         This folder contains regressions with individual survey answers.
    
    - \workingfolder\OtherData

        This folder contains all non-survey data used in this paper, including the real-time and historical realizations of various inflation rates. 
        - \workingfolder\OtherData\RealTimeData
        
    - \workingfolder\SurveyData

        This folder contains survey data: SCE and SPF. 
        - \workingfolder\SurveyData\SCE
        - \workingfolder\SurveyData\SPF
        - \workingfolder\SurveyData\SPF\individual
    - \workingfolder\tables
        - \workingfolder\tables\ind
        \workingfolder\tables\reduced_form_results
        - \workingfolder\tables\reduced_form_results\before2019
    - \workingfolder\python 
       
        See below for detailed information.
        - \workingfolder\python\figures
        - \workingfolder\python\tables
        - \workingfolder\python\tables\after2022

## Data code

- For downloading the SCE data run this [Python code](/workingfolder/python/DownloadSCE.ipynb). This Python program downloads the three datasets listed and formats them as .dta files.  

The citations for these data files are as follows: 

    - Federal Reserve Bank of New York. Survey of Consumer Expectations, 2013-2016 Complete Microdata (Public). https://www.newyorkfed.org/medialibrary/Interactives/sce/sce/downloads/data/FRBNY-SCE-Public-Microdata-Complete-13-16.xlsx?sc_lang=en (accessed January 2024).

    - Federal Reserve Bank of New York. Survey of Consumer Expectations, 2017-2019 Complete Microdata (Public). https://www.newyorkfed.org/medialibrary/Interactives/sce/sce/downloads/data/FRBNY-SCE-Public-Microdata-Complete-17-19.xlsx?sc_lang=en (accessed January 2024).

    - Federal Reserve Bank of New York. Survey of Consumer Expectations, 2020-2023 Latest Microdata (Public). https://www.newyorkfed.org/medialibrary/Interactives/sce/sce/downloads/data/FRBNY-SCE-Public-Microdata-Complete-17-19.xlsx?sc_lang=en (accessed January 2024).

- Other data is stored in the \SurveyData\ and \OtherData folder.

## Other Code

- [Stata code](/workingfolder/DoFile)
    - [Cleaning inflation data](/workingfolder/DoFile/Step00_InflationData.do)
    - [Cleaning SCE micro data](/workingfolder/DoFile/Step01_CleaningSCE%26hist.do)
    - [CLeaning SPF micro data](/workingfolder/DoFile/Step02_CleaningSPF.do)
    - [Cleaning SPF population density data](/workingfolder/DoFile/Step02b_CleaningSPF_Prob.do)
    - [Cleaning SPF individual density data](/workingfolder/DoFile/Step02c_CleaningSPF_IndProb.do)
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
