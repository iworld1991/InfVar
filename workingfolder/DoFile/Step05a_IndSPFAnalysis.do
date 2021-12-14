clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/ind"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/indSPF_log",replace


***************************
**  Clean and Merge Data **
***************************

use "${folder}/SPF/individual/InfExpSPFPointIndQ",clear 

duplicates report year quarter ID 

merge 1:1 year quarter ID using "${folder}/SPF/individual/InfExpSPFDstIndQClean.dta"

rename _merge SPFDst_merge
table year if SPFDst_merge ==3

merge m:1 year quarter using "${mainfolder}/OtherData/InfShocksQClean.dta",keep(match using master)
rename _merge infshocks_merge


*******************************
**  Set Panel Data Structure **
*******************************

xtset ID dateQ
sort ID year quarter 

drop if ID==ID[_n-1] & INDUSTRY != INDUSTRY[_n-1]


*******************************
**  Summary Statistics of SPF **
*******************************

tabstat ID,s(count) by(dateQ) column(statistics)

******************************
*** Computing some measures **
******************************

gen SPFCPI_FE = CPI1y - Inf1y_CPIAU
label var SPFCPI_FE "1-yr-ahead forecast error(SPF CPI)"
gen SPFCCPI_FE = CORECPI1y - Inf1y_CPICore
label var SPFCPI_FE "1-yr-ahead forecast error(SPF core CPI)"
gen SPFPCE_FE = PCE1y - Inf1y_PCE
label var SPFPCE_FE "1-yr-ahead forecast error(SPF PCE)"


gen SPFCPI_FE0 = CPI1y - Inf1y_CPIAU
label var SPFCPI_FE0 "1-yr nowcasting error(SPF CPI)"
gen SPFCCPI_FE0 = CORECPI1y - Inf1y_CPICore
label var SPFCPI_FE0 "1-yr nowcasting error(SPF core CPI)"
gen SPFPCE_FE0 = PCE1y - Inf1y_PCE
label var SPFPCE_FE "1-yr nowcasting error (SPF PCE)"


*****************************************
****  Renaming so that more consistent **
*****************************************


rename CPI1y SPFCPI_Mean
rename PCE1y SPFPCE_Mean
rename COREPCE1y SPFCPCE_Mean
rename CORECPI1y SPFCCPI_Mean

rename PRCCPIMean0 SPFCPI_Mean0
rename PRCPCEMean0 SPFPCE_Mean0

rename PRCPCEVar1 SPFPCE_Var
rename PRCCPIVar1 SPFCPI_Var
rename PRCPCEVar0 SPFPCE_Var0
rename PRCCPIVar0 SPFCPI_Var0

rename SPFCPI_FE SPFCPI_FE
rename SPFPCE_FE SPFPCE_FE


rename CPI_ct50 SPFCPI_ct50
rename PCE_ct50 SPFPCE_ct50

*******************************
**  Generate Variables       **
*******************************

gen InfExp_Mean = .
gen InfExp_Var = .
gen InfExp_FE = .
*gen InfExp_Disg = . 

gen InfExp_Mean_ch = .
gen InfExp_Var_ch = .
gen InfExp_FE_ch = .
*gen InfExp_Disg_ch = . 

gen InfExp_Mean0 = .
gen InfExp_Var0 = .


gen InfExp_Mean_rv = .
gen InfExp_Var_rv = .


************************************************
** Auto Regression of the Individual Moments  **
************************************************

eststo clear

foreach mom in Mean FE Var{
   foreach var in SPFCPI SPFPCE{
    replace InfExp_`mom' = `var'_`mom'
	xtset ID dateQ
    replace InfExp_`mom'_ch = InfExp_`mom'-l1.InfExp_`mom'

	eststo `var'_`mom'lvl: reg InfExp_`mom' l(3/5).InfExp_`mom', vce(cluster dateQ)
    eststo `var'_`mom'diff: reg InfExp_`mom'_ch l(3/5).InfExp_`mom'_ch, vce(cluster dateQ)
  }
}
esttab using "${sum_table_folder}/ind/autoregSPFIndQ.csv", mtitles se  r2 replace
eststo clear


*******************************
*** Unbiasedness Test        **
*******************************
eststo clear

foreach mom in FE{
   foreach var in SPFCPI SPFPCE{
      ttest `var'_`mom'=0
}
}

**********************************************
*** Revision Efficiency Test Using FE       **
**********************************************

eststo clear

foreach mom in FE{
   foreach var in SPFCPI SPFPCE{
   replace InfExp_Mean = `var'_Mean
   replace InfExp_`mom' = `var'_`mom'
   eststo `var'_`mom'_bias: reg InfExp_`mom',robust 
   eststo `var'_`mom'_lag4: reg  InfExp_`mom' l(4).InfExp_Mean, robust
   eststo `var'_`mom'_arlag4: reg InfExp_`mom' l(4).InfExp_`mom',robust
   eststo `var'_`mom'_arlag13: reg  InfExp_`mom' l(1/3).InfExp_`mom', robust

 }
}
esttab using "${sum_table_folder}/ind/FEEfficiencySPFIndQ.csv", mtitles se(%8.3f) scalars(N r2) replace

***************************************************
*** Revision Efficiency Test Using Mean Revision **
***************************************************


***********************************************************
** Create some deviation measures from population median **
***********************************************************
eststo clear

foreach mom in Mean{
  foreach var in SPFCPI SPFPCE{
  gen `var'_dv = `var'_Mean-`var'_ct50
  label var `var'_dv "Deviation from median forecast"
  }
}

eststo clear

foreach var in SPFCPI SPFPCE{
  foreach mom in Mean{
     replace InfExp_`mom'_rv =  `var'_`mom'0 - l1.`var'_`mom'
	 eststo `var'`mom'rvlv0: reg InfExp_`mom'_rv, vce(cluster date)
     eststo `var'`mom'rvlv1: reg InfExp_`mom'_rv l1.InfExp_`mom'_rv `var'_ct50, vce(cluster date)
	 eststo `var'`mom'rvlv2: reg  InfExp_`mom'_rv l(1/2).InfExp_`mom'_rv `var'_ct50, vce(cluster date)
	 eststo `var'`mom'rvlv3: reg  InfExp_`mom'_rv l(1/3).InfExp_`mom'_rv `var'_ct50, vce(cluster date)
 }
}

foreach var in SPFCPI SPFPCE{
  foreach mom in Var{
     replace InfExp_`mom'_rv =  `var'_`mom'0 - l1.`var'_`mom'
	 eststo `var'`mom'rvlv0: reg InfExp_`mom'_rv, vce(cluster date) 
     eststo `var'`mom'rvlv1: reg InfExp_`mom'_rv l1.InfExp_`mom'_rv, vce(cluster date) 
	 eststo `var'`mom'rvlv2: reg  InfExp_`mom'_rv l(1/2).InfExp_`mom'_rv, vce(cluster date) 
	 eststo `var'`mom'rvlv3: reg  InfExp_`mom'_rv l(1/3).InfExp_`mom'_rv, vce(cluster date)
 }
}

esttab using "${sum_table_folder}/ind/RVEfficiencySPFIndQ.csv", mtitles b(%8.3f) se(%8.3f) scalars(N r2) sfmt(%8.3f %8.3f %8.3f) replace



*******************************************************
***  Weak test on changes of forecst and uncertainty **
*******************************************************


eststo clear

foreach var in SPFCPI SPFPCE{
  foreach mom in Mean{
     replace InfExp_`mom'_ch =  `var'_`mom' - l1.`var'_`mom'
	 eststo `var'`mom'diff0: reg InfExp_`mom'_ch, vce(cluster date)
     eststo `var'`mom'diff1: reg InfExp_`mom'_ch l1.InfExp_`mom'_ch, vce(cluster date)
	 eststo `var'`mom'diff2: reg  InfExp_`mom'_ch l(1/2).InfExp_`mom'_ch, vce(cluster date)
	 eststo `var'`mom'diff3: reg  InfExp_`mom'_ch l(1/3).InfExp_`mom'_ch, vce(cluster date)
 }
}

foreach var in SPFCPI SPFPCE{
  foreach mom in Var{
     replace InfExp_`mom'_ch =  `var'_`mom' - l1.`var'_`mom'
	 eststo `var'`mom'diff0: reg InfExp_`mom'_ch, vce(cluster date) 
     eststo `var'`mom'diff1: reg InfExp_`mom'_ch l1.InfExp_`mom'_ch, vce(cluster date) 
	 eststo `var'`mom'diff2: reg  InfExp_`mom'_ch l(1/2).InfExp_`mom'_ch, vce(cluster date) 
	 eststo `var'`mom'diff3: reg  InfExp_`mom'_ch l(1/3).InfExp_`mom'_ch, vce(cluster date)
 }
}

esttab using "${sum_table_folder}/ind/ChEfficiencySPFIndQ.csv", mtitles b(%8.3f) se(%8.3f) scalars(N r2) sfmt(%8.3f %8.3f %8.3f) replace


/*
*****************************************
***  Revesion Efficiency test on level **
*****************************************

eststo clear


foreach var in SPFCPI SPFPCE{
  foreach mom in Mean{
     replace InfExp_`mom' = `var'_`mom'
	 replace InfExp_`mom'0 = `var'_`mom'0
     eststo `var'`mom'rvlv1: reg InfExp_`mom'0 l1.InfExp_`mom' l1.`var'_ct50, vce(cluster date)
	 test _b[l1.InfExp_`mom']=1
	 scalar pvtest= r(p)
	 estadd scalar pvtest
	 eststo `var'`mom'rvlv2: reg InfExp_`mom'0 l(1/2).InfExp_`mom' l(1/2).`var'_ct50, vce(cluster date)
	 test _b[l1.InfExp_`mom']+_b[l2.InfExp_`mom']=1
	 scalar pvtest= r(p)
	 estadd scalar pvtest
	 eststo `var'`mom'rvlv3: reg InfExp_`mom'0 l(1/3).InfExp_`mom' l(1/3).`var'_ct50, vce(cluster date)
     test _b[l1.InfExp_`mom']+_b[l2.InfExp_`mom']+_b[l3.InfExp_`mom']=1
	 scalar pvtest= r(p)
	 estadd scalar pvtest
 }
}


foreach var in SPFCPI SPFPCE{
  foreach mom in Var{
     replace InfExp_`mom' = `var'_`mom'
	 replace InfExp_`mom'0 = `var'_`mom'0
     eststo `var'`mom'rvlv1: reg InfExp_`mom'0 l1.InfExp_`mom', vce(cluster date)
	 test _b[_cons]=0
	 scalar pvtest= r(p)
	 estadd scalar pvtest
	 eststo `var'`mom'rvlv2: reg InfExp_`mom'0 l(1/2).InfExp_`mom', vce(cluster date)
	 test _b[_cons]=0
	 scalar pvtest= r(p)
	 estadd scalar pvtest
	 eststo `var'`mom'rvlv3: reg InfExp_`mom'0 l(1/3).InfExp_`mom', vce(cluster date)
	 test _b[_cons]=0
	 scalar pvtest= r(p)
	 estadd scalar pvtest
 }
}

esttab using "${sum_table_folder}/ind/RVEfficiencySPFIndQ_lvl.csv", mtitles se(%8.3f) scalars(pvtest N r2) sfmt(%8.3f %8.3f %8.3f) replace
*/


/*
******************************************************
** Response  Estimates using individual moments     **
******************************************************

keep if year>=2008


eststo clear

foreach mom in FE{
   foreach var in SPFCPI SPFPCE{
       * shocks 
       capture eststo `var'_`mom': reg `var'_`mom' l(1/2).`var'_`mom' ///
	                  l(0/1).pty_shock l(0/1).op_shock ///
					 l(0/1).mp1ut_shock l(0/1).ED8ut_shock, vce(cluster dateQ)
       }
 }
 
 
foreach mom in Var{
   foreach var in SPFCPI SPFPCE{
       * abs of shocks 
       capture eststo `var'_`mom': reg `var'_`mom' l(1/2).`var'_`mom' ///
	                  l(0/1).pty_abshock l(0/1).op_abshock ///
					 l(0/1).mp1ut_abshock l(0/1).ED8ut_abshock, vce(cluster dateQ)
   }
}


esttab using "${sum_table_folder}/SPF_ind_ashocks.csv", drop(_cons) mtitles se r2 replace

*/




** !!!! Need to find a way to run var for panel data
/*
************************************************
** IRF using individual SPF moments     **
************************************************



foreach mom in FE{
   foreach var in SPFCPI SPFPCE{
       * shocks 
       var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).op_shock ///
					 l(0/1).mp1ut_shock l(0/1).ED8ut_shock)
   set seed 123456
   capture irf create `var', set(`mom') step(10) bsp replace 
}
   * Non-MP shocks plots
   irf graph dm, set(`mom') impulse(pty_shock op_shock) ///
                         byopts(col(2) title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_shocks_nmp", as(png) replace
   
   * MP shocks plots
   capture irf graph dm, set(`mom') impulse(mp1ut_shock ED8ut_shock) ///
                         byopts(col(2) title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_mpshocks", as(png) replace
   
}



****************************************************************
** IRF of SPF moments (all shocks(abs) exl MP at one time)    **
****************************************************************


foreach mom in Var{
   foreach var in SPFCPI SPFPCE{
       * shocks 
        capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).pty_shock l(0/1).mp1ut_shock l(0/1).ED8ut_shock ///
					 l(0/1).pty_abshock l(0/1).op_abshock l(0/1).mp1ut_abshock l(0/1).ED8ut_abshock)
   set seed 123456
   capture irf create `var', set(`mom') step(10) bsp replace 
}
   * Non-MP shocks 
   capture irf graph dm, set(`mom') impulse(pty_abshock op_abshock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_ab_ashocks_nmp", as(png) replace
   
   * Non-MP shocks 
   capture irf graph dm, set(`mom') impulse(mp1ut_abshock ED8ut_abshock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Quarters") 
   capture graph export "${sum_graph_folder}/irf/moments/SPF`mom'_ab_mpshocks", as(png) replace
}
*/


log close 
