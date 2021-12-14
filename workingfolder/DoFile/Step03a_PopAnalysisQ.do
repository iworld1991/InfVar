clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/pop_log",replace

****************************************
**** Population Moments Analysis ******
****************************************

***********************
**   Merge other data**
***********************

use "${folder}/SCE/InfExpSCEProbPopM",clear 

merge 1:1 year month using "${mainfolder}/OtherData/RecessionDateM.dta", keep(match)
rename _merge  recession_merge

merge 1:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match using master)
rename _merge inflation_merge 

merge 1:1 year month using "${folder}/MichiganSurvey/InfExpMichM.dta"
rename inf_exp InfExpMichMed 
rename _merge michigan_merge 

***********************************************
** create quarter variable to match with spf
*********************************************

gen quarter = .
replace quarter=1 if month<4 & month >=1
replace quarter=2 if month<7 & month >=4
replace quarter=3 if month<10 & month >=7
replace quarter=4 if month<=12 & month >=10

merge m:1 year quarter using "${folder}/SPF/individual/InfExpSPFPointPopQ.dta",keep(match using master)
rename _merge spf_merge 

merge m:1 year quarter using "${folder}/SPF/InfExpSPFDstPopQ.dta",keep(match using master)
rename _merge spf_dst_merge 


*************************
** Declare time series **
**************************

gen date2=string(year)+"m"+string(month)
gen date3= monthly(date2,"YM")
format date3 %tm 

drop date2 date 
rename date3 date

tsset date
sort year quarter month 


******************************
*** Computing some measures **
******************************

gen SCE_FE = Q9_mean - Inf1yf_CPIAU
label var SCE_FE "1-yr-ahead forecast error"

gen SPFCPI_FE = CORECPI1y - Inf1yf_CPICore
label var SPFCPI_FE "1-yr-ahead forecast error(SPF Core CPI)"

gen SPFPCE_FE = COREPCE1y - Inf1yf_PCECore
label var SPFPCE_FE "1-yr-ahead forecast error(SPF Core PCE)"


***************************
***  Population Moments *** 
***************************
tsset date

estpost tabstat Q9_mean Q9_var Q9_disg Q9_iqr CPI1y PCE1y CORECPI1y COREPCE1y ///
                CPI_disg PCE_disg CORECPI_disg COREPCE_disg PRCCPIVar1mean PRCPCEVar1mean, ///
			    st(mean var median) columns(statistics)
esttab . using "${sum_table_folder}/pop_sum_stats.csv", cells("mean(fmt(a3)) var(fmt(a3)) median(fmt(a3))") replace

eststo clear
foreach var in Q9_mean Q9_var Q9_disg CPI1y PCE1y CORECPI1y COREPCE1y{
gen `var'_ch = `var'-l1.`var'
label var `var'_ch "m to m+1 change of `var'"
eststo: reg `var' l(1/5).`var'
eststo: reg `var'_ch l(1/5).`var'_ch
corrgram `var', lags(5) 
*gen `var'1=`r(ac1)'
*label var `var'1 "Auto-correlation coefficient of `var'"
}
esttab using "${sum_table_folder}/autoreg.csv", se r2 replace
eststo clear
*/


****************************************
**** Quarterly Level Analysis  ******
****************************************

*******************************************
***  Collapse monthly data to quarterly  **
*******************************************

local Infbf    Inf1y_CPIAU Inf1yf_CPIAU Inf1y_CPICore Inf1yf_CPICore Inf1y_PCE ///
               Inf1yf_PCE Inf1y_PCECore Inf1yf_PCECore

local Moments  Q9_mean Q9_var Q9_disg Q9_iqr ///
               Q9_fe_var Q9_fe_atv Q9_atv ///
               CPI1y PCE1y CORECPI1y COREPCE1y InfExpMichMed ///
               CPI_disg PCE_disg CORECPI_disg COREPCE_disg ///
			   CPI_atv PCE_atv CORECPI_atv COREPCE_atv ///
			   CPI_fe_var PCE_fe_var CORECPI_fe_var COREPCE_fe_var ///
			   CPI_fe_atv PCE_fe_atv CORECPI_fe_atv COREPCE_fe_atv ///
			   SCE_FE SPFCPI_FE SPFPCE_FE ///
			   PRCCPIVar1mean PRCPCEVar1mean PRCCPIVar0mean PRCPCEVar0mean 
				
local MomentsRv PRCCPIMean_rv PRCPCEMean_rv  PRCCPIVar_rv PRCPCEVar_rv  ///
                PRCCPIMeanl1  PRCCPIVarl1 PRCCPIMeanf0  PRCCPIVarf0 ///	
				PRCPCEMeanl1  PRCPCEVarl1 PRCPCEMeanf0  PRCPCEVarf0
				
				
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


collapse (mean) `Moments' `MomentsMom' `MomentsRv' `Infbf', ///
				by(year quarter) 

gen date2=string(year)+"Q"+string(quarter)
gen date3= quarterly(date2,"YQ")
format date3 %tq 
drop date2 
rename date3 date

tsset date
sort year quarter  

order date year quarter 

/*
********************************************************
*** Multiple series charts Quarterly (Moments Only)  ***
********************************************************
drop if CPI1y ==. | PCE1y==.


twoway (tsline Q9_mean) (tsline InfExpMichMed, lp("dash_dot")) ///
       (tsline CPI1y, lp("shortdash")) (tsline PCE1y, lp("dash")), ///
						 title("1-yr-ahead Expected Inflation") ///
						 xtitle("Time") ytitle("") ///
						 legend(label(1 "Mean Expectation(SCE)") ///
						        label(2 "Median Expectation(Michigan)") ///
								label(3 "Mean Expectation CPI (SPF)") ///
								label(4 "Mean Expectation PCE (SPF)") col(1))
graph export "${sum_graph_folder}/mean_medQ", as(png) replace


twoway (tsline Q9_disg, ytitle(" ",axis(1))) ///
       (tsline CORECPI_disg,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if CORECPI_disg!=., ///
	   title("Disagreements in 1-yr-ahead Inflation") xtitle("Time") ///
	   legend(label(1 "Disagreements (SCE)") label(2 "Disagreements(SPF)(RHS)"))
graph export "${sum_graph_folder}/disg_disgQ", as(png) replace 


twoway (tsline Q9_var, ytitle(" ",axis(1)) lp("solid") ) ///
       (tsline PRCCPIVar1mean, yaxis(2) ytitle("",axis(2)) lp("shortdash")) ///
	   (tsline PRCPCEVar1mean, yaxis(2) ytitle("",axis(2)) lp("dash_dot")) ///
	   if Q9_var!=., ///
	   title("Uncertainty in 1-yr-ahead Inflation") xtitle("Time") ///
	   legend(label(1 "Uncertainty (SCE)")  /// 
	          label(2 "Uncertainty (SPF CPI)(RHS)") ///
			  label(3 "Uncertainty (SPF PCE)(RHS)") col(1)) 
			  
graph export "${sum_graph_folder}/var_varQ", as(png) replace 


twoway (tsline SCE_FE)  (tsline SPFCPI_FE, yaxis(2) lp("dash")) ///
        (tsline SPFPCE_FE, yaxis(2) lp("dash_dot")) ///
                         if SPFCPI_FE!=., ///
						 title("1-yr-ahead Forecast Errors") ///
						 xtitle("Time") ytitle("") ///
						 legend(col(1) label(1 "SCE") label(2 "SPF CPI (RHS)") ///
						                label(3 "SPF PCE(RHS)"))
graph export "${sum_graph_folder}/fe_feQ", as(png) replace



twoway (tsline SPFCPI_FE, ytitle(" ",axis(1))) ///
       (tsline PRCCPIVar1mean,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if PRCCPIVar1mean!=., ///
	   title("1-yr-ahead Expected Inflation (SPF CPI)") xtitle("Time") ///
	   legend(label(1 "Average Forecast Error") label(2 "Average Uncertainty(RHS)"))
graph export "${sum_graph_folder}/fe_varSPFSPIQ", as(png) replace 


twoway (tsline SPFPCE_FE, ytitle(" ",axis(1))) ///
       (tsline PRCPCEVar1mean,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if PRCPCEVar1mean!=., ///
	   title("1-yr-ahead Expected Inflation (SPF PCE)") xtitle("Time") ///
	   legend(label(1 "Average Forecast Error") label(2 "Average Uncertainty(RHS)"))
graph export "${sum_graph_folder}/fe_varSPFPCEQ", as(png) replace 



twoway (tsline CPI_disg, ytitle(" ",axis(1))) ///
       (tsline PRCCPIVar1mean,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if PRCCPIVar1mean!=., ///
	   title("1-yr-ahead Expected Inflation(SPF CPI)") xtitle("Time") ///
	   legend(label(1 "Disagreements") label(2 "Average Uncertainty(RHS)")) 
graph export "${sum_graph_folder}/var_disgSPFCPIQ", as(png) replace 


twoway (tsline PCE_disg, ytitle(" ",axis(1))) ///
       (tsline PRCPCEVar1mean,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if PRCPCEVar1mean!=., ///
	   title("1-yr-ahead Expected Inflation(SPF PCE)") xtitle("Time") ///
	   legend(label(1 "Disagreements") label(2 "Average Uncertainty(RHS)")) 
graph export "${sum_graph_folder}/var_disgSPFPCEQ", as(png) replace 
*/

/*
*****************************************
** These are the charts for paper draft 
****************************************

twoway (tsline PRCCPIMean1p25, ytitle(" ",axis(1))  lwidth(thick) lp("shortdash")) ///
       (tsline PRCCPIMean1p75, ytitle(" ",axis(1)) lwidth(thick) lp("shortdash")) ///
	   (tsline PRCCPIMean1p50, ytitle(" ",axis(1)) lwidth(thick) lp("solid")) ///
	   if PRCCPIVar1p25!=. , /// 
	   title("SPF(CPI)",size(large)) xtitle("Time") ///
	   legend(label(1 "25 pctile of forecast") label(2 "75 pctile of forecast") ///
	          label(3 "50 pctile of forecast") col(1)) 
graph export "${sum_graph_folder}/IQRmeanCPIQ", as(png) replace 


twoway (tsline PRCPCEMean1p25, ytitle(" ",axis(1))  lwidth(thick) lp("shortdash")) ///
       (tsline PRCPCEMean1p75, ytitle(" ",axis(1)) lwidth(thick) lp("shortdash")) ///
	   (tsline PRCPCEMean1p50, ytitle(" ",axis(1)) lwidth(thick) lp("solid")) ///
	   if PRCPCEVar1p25!=. , /// 
	   title("SPF(PCE)",size(large)) xtitle("Time") ///
	   legend(label(1 "25 pctile of forecast") label(2 "75 pctile of forecast") ///
	          label(3 "50 pctile of forecast") col(1)) 
graph export "${sum_graph_folder}/IQRmeanPCEQ", as(png) replace 


twoway (tsline PRCCPIVar1p25, ytitle(" ",axis(1))  lwidth(thick) lp("shortdash")) ///
       (tsline PRCCPIVar1p75, ytitle(" ",axis(1)) lwidth(thick) lp("shortdash")) ///
	   (tsline PRCCPIVar1p50, ytitle(" ",axis(1)) lwidth(thick) lp("solid")) ///
	   if PRCCPIVar1p25!=. , /// 
	   title("SPF(CPI)",size(large)) xtitle("Time") ///
	   legend(label(1 "25 pctile of uncertainty") label(2 "75 pctile of uncertainty") ///
	          label(3 "50 pctile of uncertainty") col(1)) 
graph export "${sum_graph_folder}/IQRvarCPIQ", as(png) replace 


twoway (tsline PRCPCEVar1p25, ytitle(" ",axis(1)) lwidth(thick) lp("shortdash")) ///
       (tsline PRCPCEVar1p75, ytitle(" ",axis(1)) lwidth(thick) lp("shortdash")) ///
	   (tsline PRCPCEVar1p50, ytitle(" ",axis(1)) lwidth(thick) lp("solid")) ///
	   if PRCPCEVar1p25!=. , ///
	   title("SPF(PCE)",size(large)) xtitle("Time") ///
	   legend(label(1 "25 pctile of uncertainty") label(2 "75 pctile of uncertainty") ///
	          label(3 "50 pctile of uncertainty") col(1)) 
graph export "${sum_graph_folder}/IQRvarPCEQ", as(png) replace 



** generate absolute values of FE for plotting

foreach var in SPFCPI SPFPCE{
gen `var'_abFE = abs(`var'_FE)
label var `var'_abFE "Absolute Val of Average Forecast Error"
}


** temporarily change the name for plotting 

label var Inf1yf_CPIAU "Realized Headline CPI Inflation"
label var Inf1yf_CPICore "Realized Core CPI Inflation"
label var SPFCPI_FE "Average Forecast Error"
label var CPI_disg "Disagreement"
label var CORECPI_disg "Disagreement"

label var PRCCPIVar1mean "Average Uncertainty(RHS)"
label var Inf1yf_PCE "Realized PCE Inflation"
label var Inf1yf_PCECore "Realized Core PCE Inflation"
label var SPFPCE_FE "Average Forecast Error"
label var PCE_disg "Disagreement"
label var COREPCE_disg "Disagreement"

label var PRCPCEVar1mean "Average Uncertainty(RHS)"


foreach var in Inf1yf_CPICore SPFCPI_abFE CORECPI_disg{
pwcorr `var' PRCCPIVar1mean, star(0.05)
local rho: display %4.2f r(rho) 
twoway (tsline `var',ytitle(" ",axis(1)) lp("shortdash") lwidth(thick)) ///
       (tsline PRCCPIVar1mean, yaxis(2) ytitle("",axis(2)) lp("longdash") lwidth(thick)) ///
	   if PRCCPIVar1mean!=., ///
	   title("SPF(CPI)",size(large)) xtitle("Time") ytitle("") ///
	   legend(size(large) col(1)) ///
	   caption("{superscript:Corr Coeff= `rho'}", ///
	   justification(left) position(11) size(vlarge))
graph export "${sum_graph_folder}/`var'_varSPFCPIQ", as(png) replace
}

foreach var in Inf1yf_PCECore SPFPCE_abFE COREPCE_disg{
pwcorr `var' PRCPCEVar1mean, star(0.05)
local rho: display %4.2f r(rho) 
twoway (tsline `var',ytitle(" ",axis(1)) lp("shortdash") lwidth(thick)) ///
       (tsline PRCPCEVar1mean, yaxis(2) ytitle("",axis(2)) lp("longdash") lwidth(thick)) ///
	   if PRCPCEVar1mean!=., ///
	   title("SPF(PCE)",size(large)) xtitle("Time") ytitle("") ///
	   legend(size(large) col(1)) ///
	   caption("{superscript:Corr Coeff= `rho'}", ///
	   justification(left) position(11) size(vlarge))
graph export "${sum_graph_folder}/`var'_varSPFPCEQ", as(png) replace
}

*/

********************************************************************
** Drop the inflation measures temporarily used for plotting ********
*********************************************************************


foreach var in `Infbf'{
  drop `var'
}

********************************
***  Autoregression Quarterly **
*******************************

tsset date 

eststo clear

gen InfExp1y = .
gen InfExpFE1y = .
gen InfExpVar1y=.
gen InfExpDisg1y = .

*****************************************
****  Renaming so that more consistent **
*****************************************


rename Q9_mean SCE_Mean
rename Q9_var SCE_Var
rename Q9_disg SCE_Disg
rename SCE_FE SCE_FE
rename Q9_fe_var SCE_FEVar
rename Q9_fe_atv SCE_FEATV
rename Q9_atv SCE_ATV

rename CPI1y SPFCPINC_Mean
rename PCE1y SPFPCENC_Mean
rename COREPCE1y SPFPCE_Mean
rename CORECPI1y SPFCPI_Mean

rename CPI_disg SPFCPINC_Disg
rename PCE_disg SPFPCENC_Disg 
rename CORECPI_disg SPFCPI_Disg
rename COREPCE_disg SPFPCE_Disg

rename CPI_atv SPFCPINC_ATV
rename PCE_atv SPFPCENC_ATV
rename CORECPI_atv SPFCPI_ATV
rename COREPCE_atv SPFPCE_ATV

rename PRCPCEVar1mean SPFPCE_Var
rename PRCCPIVar1mean SPFCPI_Var

rename SPFCPI_FE SPFCPI_FE
rename SPFPCE_FE SPFPCE_FE

rename CPI_fe_var SPFCPINC_FEVar
rename PCE_fe_var SPFPCENC_FEVar 
rename CORECPI_fe_var SPFCPI_FEVar
rename COREPCE_fe_var SPFPCE_FEVar

rename CPI_fe_atv SPFCPINC_FEATV
rename PCE_fe_atv SPFPCENC_FEATV 
rename CORECPI_fe_atv SPFCPI_FEATV
rename COREPCE_fe_atv SPFPCE_FEATV


rename PRCPCEMean_rv SPFPCE_Mean_rv
rename PRCCPIMean_rv SPFCPI_Mean_rv

rename PRCPCEVar_rv SPFPCE_Var_rv
rename PRCCPIVar_rv SPFCPI_Var_rv

**********************************
rename PRCCPIMeanl1 SPFCPI_Meanl1
rename PRCCPIVarl1 SPFCPI_Varl1

rename PRCCPIMeanf0 SPFCPI_Meanf0
rename PRCCPIVarf0 SPFCPI_Varf0

rename PRCPCEMeanl1 SPFPCE_Meanl1
rename PRCPCEVarl1 SPFPCE_Varl1

rename PRCPCEMeanf0 SPFPCE_Meanf0
rename PRCPCEVarf0 SPFPCE_Varf0

********************************
gen InfExp_Mean = .
gen InfExp_Var = .
gen InfExp_FE = .
gen InfExp_Disg = . 

gen InfExp_Mean_ch = .
gen InfExp_Var_ch = .
gen InfExp_FE_ch = .
gen InfExp_Disg_ch = .

gen InfExp_Mean_rv =.
gen InfExp_Var_rv =.

gen InfExp_Meanl1 =. 
gen InfExp_Varl1 =. 

gen InfExp_Meanf0 =. 
gen InfExp_Varf0 =. 
 

foreach mom in Mean Var{
  foreach var in PRCCPI PRCPCE{
    forval i =0/1{
	local lb=substr("`var'",4,3)
	rename `var'`mom'`i'p25 SPF`lb'`mom'`i'p25
    rename `var'`mom'`i'p50 SPF`lb'`mom'`i'p50
	rename `var'`mom'`i'p75 SPF`lb'`mom'`i'p75
	}
   }
}

*****************************

gen InfExp_Mean0 =.
gen InfExp_Var0 =.

******************************

/*
*******************************************************
**** Autoregression on the levels of population moments
********************************************************


eststo clear 
foreach mom in Mean Var Disg FE{
   foreach var in SCE SPFCPI SPFPCE{
    replace InfExp_`mom' = `var'_`mom'
    eststo `var'_`mom': reg InfExp_`mom' l(3/5).InfExp_`mom', vce(robust)
  } 
}

esttab using "${sum_table_folder}/autoregLvlQ.csv", mtitles drop(_cons) se(%8.3f) scalars(N r2 ar2) replace


**********************************************************************
******** Autoregression on the first difference of population moments
 **********************************************************************

eststo clear
foreach mom in Mean Var Disg FE{
   foreach var in SCE SPFCPI SPFPCE{
    replace InfExp_`mom' = `var'_`mom'
    replace InfExp_`mom'_ch = InfExp_`mom'-l1.InfExp_`mom'
    eststo `var'_`mom': reg InfExp_`mom'_ch l(1/4).InfExp_`mom'_ch, vce(robust)
  }
}
esttab using "${sum_table_folder}/autoregDiffQ.csv", mtitles drop(_cons) se(%8.3f) scalars(N r2 ar2) replace


***************
*** SPF Only **
***************

eststo clear

foreach mom in Mean Var{
   foreach var in SPFCPI SPFPCE{
    replace InfExp_`mom' = `var'_`mom'
    replace InfExp_`mom'_ch = InfExp_`mom'-l1.InfExp_`mom'
	capture replace InfExp_`mom'_rv = `var'_`mom'_rv  /// caputure because FE and Disg has to rev

	eststo `var'_`mom'lvl: reg InfExp_`mom' l(1/2).InfExp_`mom' 
    eststo `var'_`mom'diff: reg InfExp_`mom'_ch l(1/2).InfExp_`mom'_ch  
	capture eststo `var'_`mom'_rv: reg InfExp_`mom'_rv l(1/2).InfExp_`mom'_rv 

  }
}
esttab using "${sum_table_folder}/autoregSPFQ.csv", mtitles drop(_cons) se(%8.3f) scalars(N r2 ar2) replace
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


gen const=1

foreach mom in FE{
   foreach var in SPFCPI SPFPCE{
      
}
}



**********************************************
*** Revision Efficiency Test Using FE       **
**********************************************

gen const=1

foreach mom in FE{
   foreach var in SPFCPI SPFPCE{
   
   replace InfExp_Mean = `var'_Mean
   replace InfExp_`mom' = `var'_`mom'
   eststo `var'_`mom'_const: reg InfExp_`mom'
   eststo `var'_`mom'_lag4: reg  InfExp_`mom' l(4).InfExp_Mean, robust
   eststo `var'_`mom'_arlag4: reg InfExp_`mom' l(4).InfExp_`mom', robust
   eststo `var'_`mom'_arlag13: reg  InfExp_`mom' l(1/3).InfExp_`mom', robust

 }
}
esttab using "${sum_table_folder}/FEEfficiencySPFQ.csv", mtitles se(%8.3f) scalars(N r2)  replace



*********************************************************************
*** Revision Efficiency Test Using Mean Revision Fuhrer's approach **
*********************************************************************


foreach mom in Var{
   foreach var in SPFCPI SPFPCE{
    ttest `var'_`mom'_rv =0
 }
}

eststo clear

foreach var in SPFCPI SPFPCE{
  foreach mom in Mean{
     replace InfExp_`mom' = `var'_`mom'
	 replace InfExp_`mom'l1 = `var'_`mom'l1
     replace InfExp_`mom'f0 = `var'_`mom'f0
     eststo `var'`mom'rvlv1: reg InfExp_`mom'f0 InfExp_`mom'
	 test _b[InfExp_`mom']=1
	  scalar btestpv= r(p)
	 estadd scalar btestpv
	 eststo `var'`mom'rvlv2: reg InfExp_`mom'f0 l(0/1).InfExp_`mom'
	 test _b[InfExp_`mom']=1
	 scalar btestpv= r(p)
	 estadd scalar btestpv
	 eststo `var'`mom'rvlv3: reg InfExp_`mom'f0 l(0/2).InfExp_`mom'
     test _b[InfExp_`mom']=1
	 scalar btestpv= r(p)
	 estadd scalar btestpv
 }
}

foreach var in SPFCPI SPFPCE{
  foreach mom in Var{
     replace InfExp_`mom' = `var'_`mom'
     replace InfExp_`mom'l1 = `var'_`mom'l1
	 replace InfExp_`mom'f0 = `var'_`mom'f0
     eststo `var'`mom'rvlv1: reg InfExp_`mom'f0 InfExp_`mom'
	 test _b[_cons]=0
	 scalar btestpv= r(p)
	 estadd scalar btestpv
	 eststo `var'`mom'rvlv2: reg InfExp_`mom'f0 l(0/1).InfExp_`mom'
	 test _b[_cons]=0
	 scalar btestpv= r(p)
	 estadd scalar btestpv
	 eststo `var'`mom'rvlv3: reg InfExp_`mom'f0 l(0/2).InfExp_`mom'
	 test _b[_cons]=0
	 scalar btestpv= r(p)
	 estadd scalar btestpv
 }
}

esttab using "${sum_table_folder}/RVEfficiencySPFQ.csv", mtitles se(%8.3f) scalars(btestpv N r2) sfmt(%8.3f %8.3f %8.3f) replace
*/


save "${folder}/InfExpQ.dta",replace 


log close 
