*******************************************************************************
** This do file cleans the density/moment estimats of individual SPF 
** InfExpSPFDstIndQ.dta and saves a clean version of it for Individual SPF analysis.
** And then it generates the population moments from SPF. InfExpSPFDstPopQ.dta.
** This data is quarterly. So one may need to convert SCE to quarterly for a 
** comparison of the two. 
********************************************************************************* 

clear 
set more off 
global mainfolder "/Users/Myworld/Dropbox/InfVar/workingfolder"
global datafolder "${mainfolder}/SurveyData/SPF/individual"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${datafolder}

use InfExpSPFDstIndQ,clear

local Moments PRCCPIMean0 PRCCPIMean1 PRCPCEMean0 PRCPCEMean1 /// 
              PRCCPIVar0 PRCCPIVar1 PRCPCEVar0 PRCPCEVar1
** 0 stands for q4/q4 in current year and 1 stands for q4/q4 for next year

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
     gen `var'`mom'f1 =f1.`var'`mom'0 
	 local lb: variable label `var'`mom'0
	 label var `var'`mom'f1 "`lb' in q+1"
	 
	 gen `var'`mom'f4 =f4.`var'`mom'0 
	 local lb: variable label `var'`mom'0
	 label var `var'`mom'f4 "`lb' in q+4"
 }
}


** Backward looking 

foreach mom in Mean Var{
   foreach var in PRCCPI PRCPCE{
     gen `var'`mom'l1 =l1.`var'`mom'0
	 replace `var'`mom'l1 = l1.`var'`mom'1 if quarter==1 
	 ** q1 needs to be compared with the forecast made in q4 in previous year
	 ** q2, q3, q4 are all compared with the nowcast made in previous quarter of the current year
	 local lb: variable label `var'`mom'0
	 label var `var'`mom'l1 "`lb' in q-1"
	 
	 gen `var'`mom'l4 =l4.`var'`mom'1
	 local lb: variable label `var'`mom'1
	 label var `var'`mom'l4 "`lb' in q-4"
 }
}


******************************
**  Generate Revision  ****
******************************


** There are two types of revisions we calculate here.
*** Type 1 is yearly. 
   ** Regardless of quarter, it is always from q4/q4 ``forecast'' in the previous year to q4/q4 nowcast in the current year. 
* Type 2 is quarterly. It differs depending on the quarter. 
*** q1. forecast from the q4 of the previous year to nowcast q1 in the current year. 
*** q2. nowcast in q1 to nowcast in q2
*** q3. nowcast in q2 to nowcast in q3
*** q4. nowcast in q3 to nowcast in q4. 




** Type 1 yearly revision 

foreach mom in Mean Var{
   foreach var in PRCCPI PRCPCE{
      gen `var'`mom'_rv1y = `var'`mom'0 - `var'`mom'l4
	  ** q4/q4 nowcast for current year at q minus q4/q4 forecast from q-4
	  *label var `var'`mom'_rv "Revision of `var'`mom'"
   }
}


** Type 2 quarterly revision 

foreach mom in Mean Var{
   foreach var in PRCCPI PRCPCE{
      gen `var'`mom'_rv = `var'`mom'0 - `var'`mom'l1
	  ** q4/q4 nowcast for current year at q minus q4/q4 forecast from q-1 
	  *label var `var'`mom'_rv "Revision of `var'`mom'"
   }
}

** 

******************************
**   Labeling for plots   ****
******************************


foreach mom in Mean {
   foreach var in PRCCPI PRCPCE{
	label var `var'`mom'0 "expected q4/q4 inflation in the current year"
	label var `var'`mom'1 "1-year-ahead q4/q4 expected inflation"
 }
}

foreach mom in Var {
   foreach var in PRCCPI PRCPCE{
	label var `var'`mom'0 "uncertainty about q4/q4 inflation in the current year"
	label var `var'`mom'1 "uncertainty of 1-year-ahead q4/q4 expected inflation"
 }
}

***************************************
**   Histograms of Moments  ***********
** Maybe replaced by kernel desntiy **
***************************************

label var PRCCPIMean1 "forecast of CPI"
label var PRCPCEMean1 "forecast of PCE"

** These are charts for paper draft.
foreach mom in Mean{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (histogram `var'`mom'_rv1y, bin(30) color(red) lcolor(red) lwidth(thick)), ///
	       xline(0) ///
	       by(year,title("Distribution of revision in `lb'") note("")) ytitle("Fraction of population") ///
		   xtitle("Revision in mean forecast")
	graph export "${sum_graph_folder}/hist/`var'`mom'01_rv_true_hist.png", as(png) replace 
 }
}


label var PRCCPIVar1 "uncertainty about CPI"
label var PRCPCEVar1 "uncertainty about PCE"

foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (histogram `var'`mom'_rv1y, bin(30) color(blue) lcolor(blue) lwidth(thick)), ///
	       xline(0) ///
	       by(year,title("Distribution of revision in `lb'") note("")) ytitle("Fraction of population") ///
		   xtitle("Revision in uncertainty")
	graph export "${sum_graph_folder}/hist/`var'`mom'01_rv_true_hist.png", as(png) replace 
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
	graph export "${sum_graph_folder}/hist/`var'`mom'1_hist.png", as(png) replace 
 }
}

foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (kdensity `var'`mom'1, n(50) lcolor(blue) lwidth(thick)), ///
	       by(year,title("Distribution of `lb'") note("")) ytitle("Fraction of population") ///
		   xtitle("Uncertainty") ///
		   note("")
	graph export "${sum_graph_folder}/hist/`var'`mom'1_hist.png", as(png) replace 
 }
}

** compare nowcast and forecast direclty 

foreach mom in Mean{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (kdensity `var'`mom'0, n(20) lcolor(red) lwidth(thick)) ///
	       (kdensity `var'`mom'l4, n(50) lpattern(dash) lcolor(black) fcolor(ltblue) lwidth(thick)), ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" )) ///
	       by(year,title("Distribution of `lb'")) ytitle("Fraction of population")
	graph export "${sum_graph_folder}/hist/`var'`mom'01_rv_hist.png", as(png) replace 
 }
}

foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'1 
    twoway (kdensity `var'`mom'0, n(50) lcolor(blue) lwidth(thick) ) ///
	       (kdensity `var'`mom'l4, n(50) fcolor(ltblue) lcolor(black) fcolor(ltblue) lwidth(thick)), ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" )) ///
	       by(year,title("Distribution of `lb'")) ytitle("Fraction of population") ///
		   xtitle("Revision in uncertainty")
	graph export "${sum_graph_folder}/hist/`var'`mom'01_rv_hist.png", as(png) replace 
 }
}


* histograms only 

label var PRCCPIMean0 "expected q4/q4 CPI inflation"
label var PRCPCEMean0 "expected q4/q4 PCE inflation"

label var PRCCPIVar0 "uncertainty about q4/q4 CPI inflation"
label var PRCPCEVar0 "uncertainty about q4/q4 PCE inflation"

foreach mom in Mean{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'0
    twoway (histogram `var'`mom'0,bin(10) color(ltblue)) ///
	       (histogram `var'`mom'l1,bin(10) fcolor(none) lcolor(red)), by(year,title("Distribution of `lb'",size(medium))) ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" )) ///
		   xtitle("Mean forecast")
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist.png", as(png) replace 
 }
}


foreach mom in Var{
   foreach var in PRCCPI PRCPCE{
	local lb: variable label `var'`mom'0
    twoway (histogram `var'`mom'0,bin(20) color(ltblue)) ///
	       (histogram `var'`mom'l1,bin(20) fcolor(none) lcolor(red)), by(year,title("Distribution of `lb'",size(medium))) ///
		   legend(order(1 "Nowcasting" 2 "Forecasting" )) ///
		   xtitle("Uncertainty")
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist.png", as(png) replace 
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

