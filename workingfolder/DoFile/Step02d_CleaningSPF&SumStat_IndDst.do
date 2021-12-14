*******************************************************************************
** This do file cleans the density/moment estimats of individual SPF 
** InfExpSPFDstIndQ.dta and saves a clean version of it for Individual SPF analysis.
** And then it generates the population moments from SPF. InfExpSPFDstPopQ.dta.
** This data is quarterly. So one may need to convert SCE to quarterly for a 
** comparison of the two. 
********************************************************************************* 

clear 
set more off 
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global datafolder "${mainfolder}/SurveyData/SPF/individual"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${datafolder}

use InfExpSPFDstIndQ,clear

local Moments PRCCPIMean0 PRCCPIMean1 PRCPCEMean0 PRCPCEMean1 /// 
              PRCCPIVar0 PRCCPIVar1 PRCPCEVar0 PRCPCEVar1

*****************************
**   Summary Stats of SPF  **
*****************************
			  
	  
* table 1
tabstat `Moments', st(N min p1 p10 p25 p50 p75 p90 p99 max) save 

return list
mat T = r(StatTotal)'
matlist T
putexcel set "${sum_table_folder}/InfExpSPFDstSum.xlsx", modify  
putexcel B2 = matrix(T), sheet("rawdata") 

foreach var in `Moments'{

      egen `var'p1=pctile(`var'),p(1)
	  egen `var'p99=pctile(`var'),p(99)
	  replace `var' = . if `var' <`var'p1 | (`var' >`var'p99 & `var'!=.)
}

* table 2
tabstat `Moments', st(N min p1 p10 p25 p50 p75 p90 p99 max) save

return list
mat T = r(StatTotal)'
matlist T
putexcel set "${sum_table_folder}/InfExpSPFDstSum.xlsx", modify 
putexcel B2 = matrix(T), sheet("data_winsored") 


** table 3

foreach var in `Moments'{
    tabstat `var' if `var'!=., st(mean median p1 p99) by(year) save
	return list
	mat All = r(Stat1)'
	matlist All
	foreach i of numlist 2/13{
	  matrix All = All\r(Stat`i')'
	}
	matlist All
	putexcel set "${sum_table_folder}/InfExpSPFDstSum.xlsx", modify 
	putexcel B2 = matrix(All), sheet("`var'") 
}


*******************************
**  Set Panel Data Structure **
*******************************

gen date2=string(year)+"Q"+string(quarter)
gen date3= quarterly(date2,"YQ")
format date3 %tq 
drop date2 
rename date3 dateQ

xtset ID dateQ
sort ID dateQ 

drop if ID==ID[_n-1] & INDUSTRY != INDUSTRY[_n-1]


*** Save individual data afterwinsorization 

save InfExpSPFDstIndQClean,replace 

******************************
**   Moments of Moments   ****
******************************

foreach mom in Mean Var{
   foreach var in PRCCPI PRCPCE{
    forvalues i=0/1{
	  egen `var'`mom'`i'p90 =pctile(`var'`mom'`i'),p(90) by(year quarter)
     egen `var'`mom'`i'p75 =pctile(`var'`mom'`i'),p(75) by(year quarter)
	 egen `var'`mom'`i'p25 =pctile(`var'`mom'`i'),p(25) by(year quarter)
	 egen `var'`mom'`i'p10 =pctile(`var'`mom'`i'),p(10) by(year quarter)
	 egen `var'`mom'`i'p50=pctile(`var'`mom'`i'),p(50) by(year quarter)
	 local lb: variable label `var'`mom'`i'
	 label var `var'`mom'`i'p75 "`lb': 75 pctile"
	 label var `var'`mom'`i'p25 "`lb': 25 pctile"
	 label var `var'`mom'`i'p50 "`lb': 50 pctile"
	 label var `var'`mom'`i'p10 "`lb': 10 pctile"
	 label var `var'`mom'`i'p90 "`lb': 10 pctile"
 }
 }
}


******************************
**  Vintages Moments      ****
******************************

** Forward looking 
foreach mom in Mean Var{
   foreach var in PRCCPI PRCPCE{
     gen `var'`mom'f0 =f1.`var'`mom'0
	 local lb: variable label `var'`mom'0
	 label var `var'`mom'f0 "`lb' in q+1"
 }
}


** Backward looking 

foreach mom in Mean Var{
   foreach var in PRCCPI PRCPCE{
     gen `var'`mom'l1 =l1.`var'`mom'1
	 local lb: variable label `var'`mom'1
	 label var `var'`mom'l1 "`lb' in q-1"
 }
}


******************************
**  Generate Revision  ****
******************************


foreach mom in Mean Var{
   foreach var in PRCCPI PRCPCE{
      gen `var'`mom'_rv = `var'`mom'0 - `var'`mom'l1
	  *label var `var'`mom'_rv "Revision of `var'`mom'"
   }
}

 
******************************
**   Labeling for plots   ****
******************************


foreach mom in Mean {
   foreach var in PRCCPI PRCPCE{
	label var `var'`mom'0 "expected inflation from previous year"
	label var `var'`mom'1 "1-year-ahead expected inflation"
 }
}

foreach mom in Var {
   foreach var in PRCCPI PRCPCE{
	label var `var'`mom'0 "uncertainty about inflation from previous year"
	label var `var'`mom'1 "uncertainty of 1-year-ahead expected inflation"
 }
}

***************************************
**   Histograms of Moments  ***********
** Maybe replaced by kernel desntiy **
***************************************

label var PRCCPIMean1 "forecast of CPI"
label var PRCPCEMean1 "forecast of PCE"

/*
** These are charts for paper draft.
foreach mom in Mean{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (kdensity `var'`mom'_rv, n(20) lcolor(red) lwidth(thick)), ///
	       xline(0) ///
	       by(year,title("Distribution of revision in `lb'") note("")) ytitle("Fraction of population") ///
		   xtitle("Revision in mean forecast")
	graph export "${sum_graph_folder}/hist/`var'`mom'01_rv_true_hist", as(png) replace 
 }
}


label var PRCCPIVar1 "uncertainty about CPI"
label var PRCPCEVar1 "uncertainty about PCE"

foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (kdensity `var'`mom'_rv, n(50) lcolor(blue) lwidth(thick)), ///
	       xline(0) ///
	       by(year,title("Distribution of revision in `lb'") note("")) ytitle("Fraction of population") ///
		   xtitle("Revision in uncertainty")
	graph export "${sum_graph_folder}/hist/`var'`mom'01_rv_true_hist", as(png) replace 
 }
}

* Kernal density plot only 
** These are charts for paper draft.
 
foreach mom in Mean{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1
    twoway (kdensity `var'`mom'1,fcolor(none) lcolor(red) lwidth(thick)), ///
	       by(year,title("Distribution of `lb'") note("")) ytitle("Fraction of population") ///
		   xtitle("Mean forecast") ///
		   note("") 
	graph export "${sum_graph_folder}/hist/`var'`mom'1_hist", as(png) replace 
 }
}

foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (kdensity `var'`mom'1, n(50) lcolor(blue) lwidth(thick)), ///
	       by(year,title("Distribution of `lb'") note("")) ytitle("Fraction of population") ///
		   xtitle("Uncertainty") ///
		   note("")
	graph export "${sum_graph_folder}/hist/`var'`mom'1_hist", as(png) replace 
 }
}


foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (kdensity `var'`mom'1, n(50) ) ///
	       (kdensity `var'`mom'f0, n(50) fcolor(ltblue)), ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" )) ///
	       by(year,title("Distribution of `lb'")) ytitle("Fraction of population") ///
		   xtitle("Revision in uncertainty")
	graph export "${sum_graph_folder}/hist/`var'`mom'01_rv_hist", as(png) replace 
 }
}


foreach mom in Mean{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (kdensity `var'`mom'1, n(20) ) ///
	       (kdensity `var'`mom'f0, n(50) lpattern(dash) fcolor(ltblue)), ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" )) ///
	       by(year,title("Distribution of `lb'")) ytitle("Fraction of population")
	graph export "${sum_graph_folder}/hist/`var'`mom'01_rv_hist", as(png) replace 
 }
}


* histograms only 
foreach mom in Mean{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'0
    twoway (histogram `var'`mom'0,bin(10) color(ltblue)) ///
	       (histogram `var'`mom'1,bin(10) fcolor(none) lcolor(red)), by(year,title("Distribution of `lb'")) ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" )) ///
		   xtitle("Mean forecast")
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist", as(png) replace 
 }
}


foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'0
    twoway (histogram `var'`mom'0,bin(20) color(ltblue)) ///
	       (histogram `var'`mom'1,bin(20) fcolor(none) lcolor(red)), by(year,title("Distribution of `lb'")) ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" )) ///
		   xtitle("Uncertainty") 
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist", as(png) replace 
 }
}


* nowcasting and forecasting 

foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'0
    twoway (kdensity  `var'`mom'0, n(30)) ///
	       (kdensity `var'`mom'1, n(30) fcolor(ltblue)), by(year,title("Distribution of `lb'")) ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" ))
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist", as(png) replace 
 }
}
*/

** make quarterly individual data 


** These are moments of moments 
local MomentsMom PRCCPIMean0p25 PRCCPIMean1p25 PRCPCEMean0p25 PRCPCEMean1p25 /// 
              PRCCPIVar0p25 PRCCPIVar1p25 PRCPCEVar0p25 PRCPCEVar1p25 ///
			  PRCCPIMean0p50 PRCCPIMean1p50 PRCPCEMean0p50 PRCPCEMean1p50 /// 
              PRCCPIVar0p50 PRCCPIVar1p50 PRCPCEVar0p50 PRCPCEVar1p50 ///
			  PRCCPIMean0p75 PRCCPIMean1p75 PRCPCEMean0p75 PRCPCEMean1p75 /// 
              PRCCPIVar0p75 PRCCPIVar1p75 PRCPCEVar0p75 PRCPCEVar1p75 ///
			   PRCCPIMean0p10 PRCCPIMean1p10 PRCPCEMean0p10 PRCPCEMean1p10 /// 
              PRCCPIVar0p10 PRCCPIVar1p10 PRCPCEVar0p10 PRCPCEVar1p10 ///
			   PRCCPIMean0p90 PRCCPIMean1p90 PRCPCEMean0p90 PRCPCEMean1p90 /// 
              PRCCPIVar0p90 PRCCPIVar1p90 PRCPCEVar0p90 PRCPCEVar1p90 

local Momentsrv PRCPCEMeanl1 PRCPCEMeanf0 PRCPCEVarl1  PRCPCEVarf0 ///
                PRCCPIMeanl1 PRCCPIMeanf0 PRCCPIVarl1  PRCCPIVarf0 ///
                PRCCPIMean_rv PRCPCEMean_rv PRCCPIVar_rv  PRCPCEVar_rv
				
				
				
** make quarterly population data 
preserve 
collapse (mean) `Moments' `MomentsMom' `Momentsrv', by(date year quarter) 

foreach var in `Moments'{
rename `var' `var'mean
label var `var'mean "Population moments: mean of `var'"
}
save "${mainfolder}/SurveyData/SPF/InfExpSPFDstPopQ1",replace 
restore 

collapse (sd) `Moments', by(date year quarter) 
foreach var in `Moments'{
replace `var' = `var'^2
rename `var' `var'disg
label var `var'disg "Population moments: variance of `var'"
}
merge using "${mainfolder}/SurveyData/SPF/InfExpSPFDstPopQ1"
drop _merge

drop date 
 
save "${mainfolder}/SurveyData/SPF/InfExpSPFDstPopQ",replace 

rm "${mainfolder}/SurveyData/SPF/InfExpSPFDstPopQ1.dta"

