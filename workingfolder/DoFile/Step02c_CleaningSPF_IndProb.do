*******************************************************************************
** This do file imports and cleans the raw data of individual probability forecasts
**  from SPF. It creats the InfExpSPFProbIndQ.dta file to be imported into DensEst.py
**   for density and moment estimation. The output of that is InfExpSPFDstIndQ.dta."
** That data will be processed by the do file next to this. *********************
*note: date SPFmicrodata is downloaded from https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/individual-forecasts
********************************************************************************* 

clear 
set more off 
cd "/Users/Myworld/Dropbox/InfVar/workingfolder/SurveyData/SPF/individual"

/*
foreach var in PRCCPI PRCPCE{
   import excel "/Users/Myworld/Dropbox/InfVar/workingfolder/SurveyData/SPF/individual/SPFmicrodata.xlsx", sheet("`var'") firstrow clear 
   save "`var'.dta",replace 
}

use PRCCPI, clear

merge 1:1 YEAR QUARTER ID INDUSTRY using PRCPCE

rename _merge PRCPCE_merge
rm "PRCPCE.dta"
rm "PRCCPI.dta"


foreach var in PRCCPI1-PRCPCE20{
destring(`var'), replace force 
}

*************************************
*** Prob only available after 2007 **
*************************************

rename YEAR year
rename QUARTER quarter 
keep if year >=2007

gen date2 = string(year)+"Q"+string(quarter)
gen date = quarterly(date2,"YQ")
format date %tq 
drop date2 
order date year quarter 

save InfExpSPFProbIndQ,replace 
*/

use InfExpSPFProbIndQ,clear

**********************
*** Label variables **
**********************

foreach var in PRCCPI PRCPCE{
label var `var'1 "prob `var' >=4% from y-1 to y(Q4 YoY)"
label var `var'2 "prob `var' 3.5%-3.9% from y-1 to y(Q4 YoY)"
label var `var'3 "prob `var' 3.0%-3.4% from y-1 to y(Q4 YoY)"
label var `var'4 "prob `var' 2.5%-2.9% from y-1 to y(Q4 YoY)"
label var `var'5 "prob `var' 2.0%-2.4% from y-1 to y(Q4 YoY)"
label var `var'6 "prob `var' 1.5%-1.9% from y-1 to y(Q4 YoY)"
label var `var'7 "prob `var' 1.0%-1.4% from y-1 to y(Q4 YoY)"
label var `var'8 "prob `var' 0.5%-0.9% from y-1 to y(Q4 YoY)"
label var `var'9 "prob `var' 0%-0.4% from y-1 to y(Q4 YoY)" 
label var `var'10 "prob `var' <0% from y-1 to y"

label var `var'11 "prob `var' >=4% from y to y+1(Q4 YoY)"
label var `var'12 "prob `var' 3.5%-3.9% from y to y+1(Q4 YoY)"
label var `var'13 "prob `var' 3.0%-3.4% from y to y+1(Q4 YoY)"
label var `var'14 "prob `var' 2.5%-2.9% from y to y+1(Q4 YoY)"
label var `var'15 "prob `var' 2.0%-2.4% from y to y+1(Q4 YoY)"
label var `var'16 "prob `var' 1.5%-1.9% from y to y+1(Q4 YoY)"
label var `var'17 "prob `var' 1.0%-1.4% from y to y+1(Q4 YoY)"
label var `var'18 "prob `var' 0.5%-0.9% from y to y+1(Q4 YoY)"
label var `var'19 "prob `var' 0%-0.4% from y to y+1(Q4 YoY)" 
label var `var'20 "prob `var' <0% from y to y+1(Q4 YoY)"
}


save InfExpSPFProbIndQ,replace

** need to estimated density first with this data, which generates InfExpSPFDstIndQ.dta
