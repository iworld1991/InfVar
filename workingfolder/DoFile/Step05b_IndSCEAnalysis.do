clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/ind"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/indSCE_log",replace


***************************
**  Clean and Merge Data **
***************************

use "${folder}/SCE/InfExpSCEProbIndM",clear 

duplicates report year month userid

rename userid ID 

merge m:1 year month using "${mainfolder}/OtherData/InfShocksMClean.dta",keep(match using master)
rename _merge infshocks_merge


*******************************
**  Set Panel Data Structure **
*******************************

xtset ID date
sort ID year month 


*******************************
**  Summary Statistics of SCE **
*******************************

*tabstat ID,s(count) by(date) column(statistics)

******************************
*** Computing some measures **
******************************

gen SCE_FE = Q9_mean - Inf1y_CPIAU
label var SCE_FE "1-yr-ahead forecast error(SCE)"

*****************************************
****  Renaming so that more consistent **
*****************************************

rename Q9_mean SCE_Mean
rename Q9_var SCE_Var

rename Q9c_mean SCE_Mean1
rename Q9c_var SCE_Var1

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
   foreach var in SCE{
    replace InfExp_`mom' = `var'_`mom'
	xtset ID date
    replace InfExp_`mom'_ch = InfExp_`mom'-l1.InfExp_`mom'

	eststo `var'_`mom'lvl: reg InfExp_`mom' l(3/5).InfExp_`mom', vce(cluster date)
    eststo `var'_`mom'diff: reg InfExp_`mom'_ch l(3/5).InfExp_`mom'_ch, vce(cluster date)
  }
}
esttab using "${sum_table_folder}/ind/autoregSCEIndM.csv", mtitles se  r2 replace
eststo clear

*******************************
*** Unbiasedness Test        **
*******************************
eststo clear

foreach mom in FE{
   foreach var in SCE{
      ttest `var'_`mom'=0
}
}

**********************************************
*** Revision Efficiency Test Using FE       **
**********************************************

eststo clear

foreach mom in FE{
   foreach var in SCE{
   replace InfExp_Mean = `var'_Mean
   replace InfExp_`mom' = `var'_`mom'
   eststo `var'_`mom'_bias: reg InfExp_`mom',robust 
   eststo `var'_`mom'_lag1: reg  InfExp_`mom' l(1/3).InfExp_Mean, robust
   eststo `var'_`mom'_lag4: reg  InfExp_`mom' l(4/7).InfExp_Mean, robust
   eststo `var'_`mom'_lag8: reg  InfExp_`mom' l(8/11).InfExp_Mean, robust
   eststo `var'_`mom'_arlag1: reg InfExp_`mom' l(1/3).InfExp_`mom',robust
   eststo `var'_`mom'_arlag4: reg InfExp_`mom' l(4/7).InfExp_`mom',robust
   eststo `var'_`mom'_arlag8: reg  InfExp_`mom' l(8/11).InfExp_`mom', robust

 }
}
esttab using "${sum_table_folder}/ind/FEEfficiencySCEIndM.csv", mtitles se(%8.3f) scalars(N r2) replace


*******************************************************
***  Weak test on changes of forecst and uncertainty **
*******************************************************

** Generate central tendency measures

foreach var in SCE{
foreach mom in Mean{
   egen `var'_`mom'_ct50 = pctile(`var'_`mom'),p(50) by(date)
   label var `var'_`mom'_ct50 "Median 1-year-ahead inflation."
}
}

eststo clear

foreach var in SCE{
  foreach mom in Mean{
     replace InfExp_`mom'_ch =  `var'_`mom' - l1.`var'_`mom'
	 eststo `var'`mom'diff0: reg InfExp_`mom'_ch, vce(cluster date)
     eststo `var'`mom'diff1: reg InfExp_`mom'_ch l1.InfExp_`mom'_ch, vce(cluster date)
	 capture eststo `var'`mom'diff2: reg  InfExp_`mom'_ch l(1/3).InfExp_`mom'_ch, vce(cluster date)
	 capture eststo `var'`mom'diff3: reg  InfExp_`mom'_ch l(1/6).InfExp_`mom'_ch, vce(cluster date)
 }
}

foreach var in SCE{
  foreach mom in Var{
     replace InfExp_`mom'_ch =  `var'_`mom' - l1.`var'_`mom'
	 eststo `var'`mom'diff0: reg InfExp_`mom'_ch, vce(cluster date) 
     eststo `var'`mom'diff1: reg InfExp_`mom'_ch l1.InfExp_`mom'_ch, vce(cluster date) 
	 eststo `var'`mom'diff2: reg  InfExp_`mom'_ch l(1/3).InfExp_`mom'_ch, vce(cluster date) 
	 capture eststo `var'`mom'diff3: reg  InfExp_`mom'_ch l(1/6).InfExp_`mom'_ch, vce(cluster date)
 }
}

esttab using "${sum_table_folder}/ind/ChEfficiencySCEIndQ.csv", mtitles b(%8.3f) se(%8.3f) scalars(N r2) sfmt(%8.3f %8.3f %8.3f) replace



** Almost works for SCE. 
** Use 2-year inflation forecast 10 months ago and 1-year forecast now

***************************************************
*** Revision Efficiency Test Using Mean Revision **
***************************************************


***********************************************************
** Create some deviation measures from population median **
***********************************************************
eststo clear


foreach mom in Mean{
  foreach var in SCE{
  gen `var'_dv = `var'_Mean-`var'_Mean_ct50
  label var `var'_dv "Deviation from median forecast"
  }
}

eststo clear

foreach var in SCE{
  foreach mom in Mean{
     replace InfExp_`mom'_rv =  `var'_`mom' - l10.`var'_`mom'1
	 eststo `var'`mom'rvlv0: reg InfExp_`mom'_rv, vce(cluster date)
     eststo `var'`mom'rvlv1: reg InfExp_`mom'_rv l1.InfExp_`mom'_rv `var'_`mom'_ct50, vce(cluster date)
	 *eststo `var'`mom'rvlv2: reg  InfExp_`mom'_rv l(1/2).InfExp_`mom'_rv `var'_`mom'_ct50, vce(cluster date)
	 *eststo `var'`mom'rvlv3: reg  InfExp_`mom'_rv l(1/3).InfExp_`mom'_rv `var'_`mom'_ct50, vce(cluster date)
 }
}

foreach var in SCE{
  foreach mom in Var{
     replace InfExp_`mom'_rv =  `var'_`mom' - l10.`var'_`mom'1
	 eststo `var'`mom'rvlv0: reg InfExp_`mom'_rv, vce(cluster date) 
     eststo `var'`mom'rvlv1: reg InfExp_`mom'_rv l1.InfExp_`mom'_rv, vce(cluster date) 
	 *eststo `var'`mom'rvlv2: reg  InfExp_`mom'_rv l(1/2).InfExp_`mom'_rv, vce(cluster date) 
	 *eststo `var'`mom'rvlv3: reg  InfExp_`mom'_rv l(1/3).InfExp_`mom'_rv, vce(cluster date)
 }
}

esttab using "${sum_table_folder}/ind/RVEfficiencySCEIndQ.csv", mtitles b(%8.3f) se(%8.3f) scalars(N r2) sfmt(%8.3f %8.3f %8.3f) replace



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
