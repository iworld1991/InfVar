clear
global mainfolder "/Users/tao/Dropbox/InfVar/workingfolder"
global folder "${mainfolder}/SurveyData/"
global shadowfolder "/Users/tao/Dropbox/InfVar-local/workingfolder/SurveyData/"

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

*merge 1:1 year month using "${mainfolder}/OtherData/RecessionDateM.dta", keep(match master)
*rename _merge  recession_merge

merge 1:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match using master)
rename _merge inflation_merge 

*merge 1:1 year month using "${shadowfolder}/MichiganSurvey/InfExpMichM.dta"
*rename inf_exp InfExpMichMed 
*rename _merge michigan_merge 

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

** sample filter 

*drop if date > monthly("2020m1","YM")

******************************
*** Computing some measures **
******************************

gen SCE_FE = Q9_mean - Inf1yf_CPIAU
label var SCE_FE "1-yr-ahead forecast error"

gen SPFCPI_FE = CORECPI1y - Inf1yf_CPICore
label var SPFCPI_FE "1-yr-ahead forecast error(SPF CPI)"

gen SPFPCE_FE = COREPCE1y - Inf1yf_PCECore
label var SPFPCE_FE "1-yr-ahead forecast error(SPF PCE)"


*****************************
****    Summary Charts ******
*****************************



**************************
** Single series charts **
**************************

foreach var in Q9_mean Q9_var Q9_iqr Q9_cent50 Q9_disg ///
               CPI1y PCE1y CORECPI1y COREPCE1y ///
			   CPI_disg PCE_disg CORECPI_disg COREPCE_disg ///
			   CPI_ct50 PCE_ct50 CORECPI_ct50 COREPCE_ct50{
local var_lb: var label `var'
tsline `var' if `var'!=.,title("`var_lb'") xtitle("Time") ytitle("")
graph export "${sum_graph_folder}/`var'.png", as(png) replace 
}



***************************
** Multiple series charts**
***************************

drop if CPI1y ==. | PCE1y==.

twoway (tsline Q9_mean)  (tsline Q9_cent50,lp("dash")) /// *(tsline InfExpMichMed, lp("dot-dash")) ///
						 (tsline CPI1y, lp("shortdash")), ///
						 title("1-yr-ahead Expected Inflation") ///
						 xtitle("Time") ytitle("") ///
						 legend(label(1 "Mean Expectation(SCE)") ///
						        label(2 "Median Expectation(Michigan)") ///
								label(3 "Mean Expectation(SPF)")position(6))
graph export "${sum_graph_folder}/mean_med.png", as(png) replace

twoway (tsline Q9_mean, ytitle(" ",axis(1))) ///
       (tsline Q9_var,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_mean!=., ///
	   title("1-yr-ahead Expected Inflation (SCE)") xtitle("Time") ///
	   legend(label(1 "Average Expectation") label(2 "Average Uncertainty(RHS)") position(6))
graph export "${sum_graph_folder}/mean_var.png", as(png) replace 



twoway (tsline Q9_mean, ytitle(" ",axis(1))) ///
       (tsline Q9_disg,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_mean!=., ///
	   title("1-yr-ahead Expected Inflation") xtitle("Time") ///
	   legend(label(1 "Average Expectation") label(2 "Disagreements(RHS)") position(6))
graph export "${sum_graph_folder}/mean_disg.png", as(png) replace 


twoway (tsline Q9_disg, ytitle(" ",axis(1))) ///
       (tsline CORECPI_disg,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if CORECPI_disg!=., ///
	   title("Disagreements in 1-yr-ahead Inflation") xtitle("Time") ///
	   legend(label(1 "Disagreements (SCE)") label(2 "Disagreements(SPF)(RHS)") position(6))
graph export "${sum_graph_folder}/disg_disg.png", as(png) replace 


twoway (tsline Q9_var, ytitle(" ",axis(1))) ///
       (tsline PRCCPIVar1mean, yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_var!=., ///
	   title("Uncertainty in 1-yr-ahead Inflation") xtitle("Time") ///
	   legend(label(1 "Uncertainty (SCE)")  /// 
	          label(2 "Uncertainty (SPF CPI)(RHS)") position(6)) 
			  
graph export "${sum_graph_folder}/var_var.png", as(png) replace 



twoway (tsline CORECPI_disg, ytitle(" ",axis(1))) ///
       (tsline PRCCPIVar1mean,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_disg!=., ///
	   title("1-yr-ahead Expected Inflation(SPF)") xtitle("Time") ///
	   legend(label(1 "Disagreements") label(2 "Average Uncertainty(RHS)") position(6)) 
graph export "${sum_graph_folder}/var_disg2.png", as(png) replace 


twoway (tsline Q9_mean) (tsline CORECPI1y,lp("longdash")) ///
       (tsline Inf1yf_PCE,lp("dash")) ///
       (tsline Inf1yf_CPIAU,lp("shortdash")) ///
	   (tsline Inf1yf_CPICore,lp("dash_dot")) ///
	   if Q9_mean!=., ///
	   title("1-yr-ahead Expected Inflation") xtitle("Time") ytitle("") ///
	   legend(label(1 "Mean Forecast(SCE)") label(2 "Mean Forecast(SPF)") label(3 "Realized Inflation(PCE)") ///
	          label(4 "Realized Inflation(Headline CPI)") label(5 "Realized Inflation(Core CPI)") position(6))
graph export "${sum_graph_folder}/mean_true.png", as(png) replace


twoway (tsline Q9_mean)  (tsline Inf1yf_PCE,lp("dash")) ///
       (tsline Inf1y_CPIAU,lp("shortdash")) ///
	   (tsline Inf1y_CPICore,lp("dash_dot")) ///
	   if Q9_mean!=., ///
	   title("1-yr-ahead Expected Inflation") xtitle("Time") ytitle("") ///
	   legend(label(1 "Mean Expectation") label(2 "Past Inflation(PCE)") ///
	          label(3 "Past Inflation(Headline CPI)") label(4 "Past Inflation(Core CPI)") position(6))
graph export "${sum_graph_folder}/mean_past.png", as(png) replace


twoway (tsline SCE_FE,ytitle("",axis(1)))  (tsline SPFCPI_FE, yaxis(2) lp("dash")) ///
       (tsline SPFPCE_FE, yaxis(2) lp("dash_dot"))  ///  
                         if SPFPCE_FE!=., ///
						 title("1-yr-ahead Forecast Errors") ///
						 xtitle("Time") ytitle("") ///
						 legend(label(1 "SCE") label(2 "SPF CPI(RHS)") ///
						 label(3 "SPF PCE(RHS)") position(6))
graph export "${sum_graph_folder}/fe_fe.png", as(png) replace



twoway (tsline Q9_disg, ytitle(" ",axis(1))) ///
       (tsline Q9_var,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_disg!=., ///
	   title("1-yr-ahead Expected Inflation(SCE)") xtitle("Time") ///
	   legend(label(1 "Disagreements") label(2 "Average Uncertainty(RHS)") position(6)) 
graph export "${sum_graph_folder}/var_disgSCEM.png", as(png) replace 


twoway (tsline SCE_FE, ytitle(" ",axis(1))) ///
       (tsline Q9_var,yaxis(2) ytitle("",axis(2)) lp("dash")) ///
	   if Q9_var!=., ///
	   title("1-yr-ahead Expected Inflation(SCE)") xtitle("Time") ///
	   legend(label(1 "Average Forecast Error") label(2 "Average Uncertainty(RHS)") position(6)) 
graph export "${sum_graph_folder}/fe_varSCEM.png", as(png) replace

pwcorr Inf1yf_CPIAU Q9_var, star(0.05)
local rho: display %4.2f r(rho) 
twoway (tsline Inf1yf_CPIAU,ytitle(" ",axis(1))lp("shortdash") lwidth(thick)) ///
       (tsline Q9_var, yaxis(2) ytitle("",axis(2)) lp("longdash") lwidth(thick)) ///
	   if Q9_var!=., ///
	   title("1-yr-ahead Expected Inflation(SCE)",size(large)) xtitle("Time") ytitle("") ///
	   legend(label(1 "Headline CPI Inflation") ///
	          label(2 "Average Uncertainty(RHS)") size(sml) position(6)) ///
	   caption("{superscript:Corr Coeff: `rho'}", ///
	   justification(left) position(11) size(large))
graph export "${sum_graph_folder}/true_varSCEM.png", as(png) replace 
*/


****************************************
**** generate ar 1 real-time prediction
****************************************

gen Inf1yf_CPIAU_ar_pr = .
label var Inf1yf_CPIAU_ar_pr "Real-time ar 1 predicted inflation"

sort date
* Loop over each time period t
forval t = 20/`=_N' {
    * Restrict the sample to the time period before t
    quietly regress Inf1yf_CPIAU L.Inf1yf_CPIAU if _n < `t'
    
    * Predict the inflation value at time t using the AR(1) model
    predict double pred, xb
    
    * Store the predicted value in the new variable
    replace Inf1yf_CPIAU_ar_pr = pred if _n == `t'
    
    * Clean up the temporary variable
    drop pred
}

gen Inf1yf_CPIAU_ar_FE = Inf1yf_CPIAU_ar_pr- Inf1yf_CPIAU 
label var Inf1yf_CPIAU_ar_FE "AR1 Forecast Error"

gen Inf1yf_CPIAU_ar_FE2 = Inf1yf_CPIAU_ar_FE^2
label var Inf1yf_CPIAU_ar_FE2 "AR1 Forecast Error Square"



***********************************************************************************
** These are the charts for paper draft (but need to set the color for the lines) 
********************************************************************************


** generate absolute values of FE for plotting

foreach var in SCE{
gen `var'_abFE = abs(`var'_FE)
label var `var'_abFE "Absolute Val of Average Forecast Error"
}


foreach var in SCE{
gen `var'_FE2 = `var'_FE^2
label var `var'_FE2 "Square of Average Forecast Error"

}

gen  SCE_FE2_rd= Q9_fe_rd^2
label var SCE_FE2_rd "Squared Foreacst Errors Residual"

gen Q9_mean_gap = Q9_mean - 2
label var Q9_mean_gap "Expected Inflation Gap"


gen Q9_mean_gap2 = Q9_mean_gap^2
label var Q9_mean_gap2 "Expected Inflation Gap Square"


** generate moving averages for charts 

tssmooth ma Q9_varmv = Q9_var,  window(1 1 1)

foreach var in Inf1yf_CPIAU Q9_mean SCE_FE SCE_FE2 Q9_disg Inf1yf_CPIAU_ar_FE2 SCE_FE2_rd Q9_mean_gap2{
tssmooth ma `var'mv =`var', window(1 1 1) 
}
label var Q9_meanmv "Expected Headline CPI Inflation"
label var Inf1yf_CPIAUmv "Realized Headline CPI Inflation"
label var SCE_FE "Average Forecast Error"
label var SCE_FE2mv "Squared Average Forecast Error"
label var Q9_disgmv "Disagreement"
label var Q9_varmv "Average Uncertainty (RHS)"
label var Inf1yf_CPIAU_ar_FE2mv "Squared AR1 Forecast Errors"
label var SCE_FE2_rdmv "Squared Forecast Error exl ind FE"
label var Q9_mean_gap2mv "Expected Inflation Gap Square"

foreach var in Inf1yf_CPIAU Q9_mean SCE_FE2 Q9_disg Inf1yf_CPIAU_ar_FE2 Q9_mean_gap2{
pwcorr `var' Q9_var, star(0.05)
local rho: display %4.2f r(rho) 
twoway (tsline `var'mv,ytitle(" ",axis(1)) lcolor(navy) lp("shortdash") lwidth(thick)) ///
       (tsline Q9_varmv, yaxis(2) ytitle("",axis(2)) lcolor(maroon) lp("longdash") lwidth(thick)) ///
	   if Q9_var!=., ///
	   title("SCE",size(large)) xtitle("Time") ytitle("") ///
	   legend(size(large) col(1) position(6)) ///
	   caption("{superscript:Corr Coeff= `rho'}", ///
	   justification(left) position(11) size(huge))
graph export "${sum_graph_folder}/`var'_varSCEM.png", as(png) replace
}


* turn them into scatter plots 

label var Q9_mean "Expected Headline CPI Inflation"
label var Inf1yf_CPIAU "Realized Headline CPI Inflation"
label var SCE_FE2 "Squared Average Forecast Error"
label var Q9_disg "Disagreement"
label var Q9_var "Average Uncertainty"
label var Inf1yf_CPIAU_ar_FE2 "Squared AR1 Forecast Errors"
label var SCE_FE2_rd "Squared Forecast Error exl ind FE"
label var Q9_mean_gap2 "Expected Inflation Gap Square"

foreach var in SCE_FE2 Q9_disg SCE_FE2 Inf1yf_CPIAU_ar_FE2 Q9_mean_gap2{
    pwcorr `var' Q9_var, star(0.05)
    local rho: display %4.2f r(rho)
    // Get the labels for the variables
    local xlabel: variable label Q9_var
    local ylabel: variable label `var'

    // Calculate the range for the 45-degree line
    summarize `var'
    local min1 = r(min)
    local max1 = r(max)

    summarize Q9_var
    local min2 = r(min)
    local max2 = r(max)

    local min = min(`min1', `min2')
    local max = max(`max1', `max2')

    twoway (scatter `var' Q9_var, mcolor(navy) msymbol(oh) mlabel(date)) ///
           (lfit `var' Q9_var, lcolor(maroon) lwidth(medthick)) ///
           (function y=x, range(`min' `max') lcolor(black) lpattern(dash) lwidth(thin)) ///
           if PRCCPIVar1mean!=., ///
           title("SCE", size(large)) xtitle("`xlabel'",size(large)) ytitle("`ylabel'",size(large)) ///
           legend(order(1 "Scatter" 2 "Fitted Line" 3 "45-degree line") ///
                  position(11) size(large) col(1)) ///
           caption("{superscript:Corr Coeff= `rho'}", ///
           justification(left) position(11) size(huge)) ///
           xscale(range(`min' `max')) yscale(range(`min' `max'))
    graph export "${sum_graph_folder}/scatter_`var'_varSCEM.png", as(png) replace
}


foreach var in Inf1yf_CPIAU Q9_mean Q9_mean_gap{
pwcorr `var' Q9_var, star(0.05)
local rho: display %4.2f r(rho) 
 // Get the labels for the variables
    local xlabel: variable label Q9_var
    local ylabel: variable label `var'

    // Calculate the range for the 45-degree line
    summarize `var' Q9_var
    local min = min(r(min), r(min_2))
    local max = max(r(max), r(max_2))
	
 twoway (scatter `var' Q9_var, mcolor(navy) msymbol(oh) mlabel(date)) ///
           (lfit `var' Q9_var, lcolor(maroon) lwidth(medthick)) ///
           if PRCCPIVar1mean!=., ///
           title("SCE", size(large)) xtitle("`xlabel'",size(large)) ytitle("`ylabel'",size(large)) ///
           legend(order(1 "Scatter" 2 "Fitted Line" 3 "45-degree line") ///
                  position(11) size(large) col(1)) ///
           caption("{superscript:Corr Coeff= `rho'}", ///
           justification(left) position(11) size(huge))
graph export "${sum_graph_folder}/scatter_`var'_varSCEM.png", as(png) replace

}

********************
*** using risiduals 
*********************

* did not label correctly before. relabel it.

label var Q9_var_rd  "Average 1-yr-ahead Uncertainty exl ind FE"
label var Q9_disg_rd "Disagreement exl id FE"

foreach var in Q9_disg_rd{
    pwcorr `var' Q9_var, star(0.05)
    local rho: display %4.2f r(rho)
    // Get the labels for the variables
    local xlabel: variable label Q9_var_rd
    local ylabel: variable label `var'

    // Calculate the range for the 45-degree line
    summarize `var'
    local min1 = r(min)
    local max1 = r(max)

    summarize Q9_var_rd
    local min2 = r(min)
    local max2 = r(max)

    local min = min(`min1', `min2')
    local max = max(`max1', `max2')

    twoway (scatter `var' Q9_var, mcolor(navy) msymbol(oh) mlabel(date)) ///
           (lfit `var' Q9_var, lcolor(maroon) lwidth(medthick)) ///
           (function y=x, range(`min' `max') lcolor(black) lpattern(dash) lwidth(thin)) ///
           if PRCCPIVar1mean!=., ///
           title("SCE", size(large)) xtitle("`xlabel'",size(large)) ytitle("`ylabel'",size(large)) ///
           legend(order(1 "Scatter" 2 "Fitted Line" 3 "45-degree line") ///
                  position(11) size(large) col(1)) ///
           caption("{superscript:Corr Coeff= `rho'}", ///
           justification(left) position(11) size(huge)) ///
           xscale(range(`min' `max')) yscale(range(`min' `max'))
    graph export "${sum_graph_folder}/scatter_`var'_varSCEM_rd.png", as(png) replace
}

ddd

** quantile plots 

twoway (tsline Q9_varp25, ytitle(" ",axis(1)) lcolor(navy) lp("shortdash") lwidth(thick)) ///
       (tsline Q9_varp75, ytitle(" ",axis(1)) lcolor(black) lp("shortdash") lwidth(thick)) ///
	   (tsline Q9_varp50, ytitle(" ",axis(1)) lcolor(maroon)  lp("solid") lwidth(thick)) ///
	   if Q9_varp50!=. , /// 
	   title("SCE") xtitle("Time") ///
	   legend(label(1 "25 percentile of uncertainty") label(2 "75 percentile of uncertainty") ///
	          label(3 "50 percentile of uncertainty") col(1)) 
graph export "${sum_graph_folder}/IQRvarSCEM.png", as(png) replace 



twoway (tsline Q9_meanp25, ytitle(" ",axis(1)) lcolor(navy)  lp("shortdash") lwidth(thick)) ///
       (tsline Q9_meanp75, ytitle(" ",axis(1)) lcolor(black) lp("shortdash") lwidth(thick)) ///
	   (tsline Q9_meanp50, ytitle(" ",axis(1)) lcolor(maroon) lp("solid") lwidth(thick)) ///
	   if Q9_varp50!=. , /// 
	   title("SCE") xtitle("Time") ///
	   legend(label(1 "25 percentile of forecast") label(2 "75 percentile of forecast") ///
	          label(3 "50 percentile of forecast") col(1)) 
graph export "${sum_graph_folder}/IQRmeanSCEM.png", as(png) replace 
*/



***************************
***  Population Moments *** 
***************************
tsset date

estpost tabstat Q9_mean Q9_var Q9_disg Q9_iqr CPI1y PCE1y CORECPI1y COREPCE1y ///
                Q9c_mean Q9c_var Q9c_disg Q9c_iqr  /// 
                CPI_disg PCE_disg CORECPI_disg COREPCE_disg PRCCPIVar1mean PRCPCEVar1mean, ///
			    st(mean var median) columns(statistics)
esttab . using "${sum_table_folder}/pop_sum_stats.csv", cells("mean(fmt(a3)) var(fmt(a3)) median(fmt(a3))") replace

eststo clear
foreach var in Q9_mean Q9_var Q9_disg Q9c_mean Q9c_var Q9c_disg CPI1y PCE1y CORECPI1y COREPCE1y{
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
**** Stay at Monthly Analysis  ******
****************************************

*******************************************
***  Collapse monthly data to monthly  **
*******************************************

local Moments Q9_mean Q9_var Q9_disg Q9_iqr CPI1y PCE1y CORECPI1y ///
              Q9_mean_rd Q9_var_rd Q9_disg_rd Q9_fe_rd ///
              Q9c_mean Q9c_var Q9c_disg Q9c_iqr ///
			  Q9_fe_var Q9_fe_atv Q9_atv ///
              COREPCE1y CPI_disg PCE_disg CORECPI_disg COREPCE_disg SCE_FE SCE_FE2 SPFCPI_FE SPFPCE_FE ///
			  CPI_atv PCE_atv CORECPI_atv COREPCE_atv ///
			  CPI_fe_var PCE_fe_var CORECPI_fe_var COREPCE_fe_var ///
			  CPI_fe_atv PCE_fe_atv CORECPI_fe_atv COREPCE_fe_atv ///
		      PRCCPIVar1mean PRCPCEVar1mean PRCCPIVar0mean PRCPCEVar0mean 
				
local MomentsRv PRCCPIMean_rv PRCPCEMean_rv  PRCCPIVar_rv PRCPCEVar_rv  ///
                PRCCPIMeanl1  PRCCPIVarl1 PRCPCEMeanl1  PRCPCEVarl1  
				*PRCCPIMeanf0  PRCCPIVarf0 PRCPCEMeanf0  PRCPCEVarf0
				
				
local MomentsMom PRCCPIMean0p25 PRCCPIMean1p25 PRCPCEMean0p25 PRCPCEMean1p25 /// 
              PRCCPIVar0p25 PRCCPIVar1p25 PRCPCEVar0p25 PRCPCEVar1p25 ///
			  PRCCPIMean0p50 PRCCPIMean1p50 PRCPCEMean0p50 PRCPCEMean1p50 /// 
              PRCCPIVar0p50 PRCCPIVar1p50 PRCPCEVar0p50 PRCPCEVar1p50 ///
			  PRCCPIMean0p75 PRCCPIMean1p75 PRCPCEMean0p75 PRCPCEMean1p75 /// 
              PRCCPIVar0p75 PRCCPIVar1p75 PRCPCEVar0p75 PRCPCEVar1p75  ///
			  Q9_meanp25 Q9_meanp50 Q9_meanp75 Q9_varp25 Q9_varp50 Q9_varp75 ///
			  Q9c_meanp25 Q9c_meanp50 Q9c_meanp75 Q9c_varp25 Q9c_varp50 Q9c_varp75


collapse (mean) `Moments' `MomentsMom' `MomentsRv', ///
				by(date year month) 

tsset date
sort year month 
order date year month 

********************************
***  Autoregression Monthly  **
*******************************

eststo clear

gen InfExp1y = .
gen InfExpFE1y = .
gen InfExpVar1y=.
gen InfExpDisg1y = .

*****************************************
****  Renaming so that more consistent **
*****************************************


rename Q9_mean SCE_Mean
rename Q9_mean_rd SCE_Mean_rd

rename Q9_var SCE_Var
rename Q9_var_rd SCE_Var_rd

rename Q9_disg SCE_Disg
rename Q9_disg_rd SCE_Disg_rd

rename SCE_FE SCE_FE
rename Q9_fe_rd SCE_FE_rd


rename Q9_fe_var SCE_FEVar

rename Q9_fe_atv SCE_FEATV
rename Q9_atv SCE_ATV

rename Q9c_mean SCE_Mean1
rename Q9c_var SCE_Var1
rename Q9c_disg SCE_Disg1

rename CPI1y SPFCPI_Mean
rename PCE1y SPFPCE_Mean
rename COREPCE1y SPFCPCE_Mean
rename CORECPI1y SPFCCPI_Mean

rename CPI_disg SPFCPI_Disg
rename PCE_disg SPFPCE_Disg 
rename CORECPI_disg SPFCCPI_Disg
rename COREPCE_disg SPFCPCE_Disg

rename CPI_atv SPFCPINC_ATV
rename PCE_atv SPFPCENC_ATV
rename CORECPI_atv SPFCPI_ATV
rename COREPCE_atv SPFPCE_ATV

rename PRCPCEVar1mean SPFPCE_Var
rename PRCCPIVar1mean SPFCPI_Var

rename SPFCPI_FE SPFCPI_FE
rename SPFPCE_FE SPFPCE_FE

rename CPI_fe_var SPFCPINC_FEVAR
rename PCE_fe_var SPFPCENC_FEVAR 
rename CORECPI_fe_var SPFCPI_FEVAR
rename COREPCE_fe_var SPFPCE_FEVAR

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

*rename PRCCPIMeanf0 SPFCPI_Meanf0
*rename PRCCPIVarf0 SPFCPI_Varf0

rename PRCPCEMeanl1 SPFPCE_Meanl1
rename PRCPCEVarl1 SPFPCE_Varl1

*rename PRCPCEMeanf0 SPFPCE_Meanf0
*rename PRCPCEVarf0 SPFPCE_Varf0

********************************
gen InfExp_Mean = .
gen InfExp_Var = .
gen InfExp_FE = .
gen InfExp_FE2 = .
gen InfExp_Disg = . 


gen InfExp_Mean_ch = .
gen InfExp_Var_ch = .
gen InfExp_FE_ch = .
gen InfExp_Disg_ch = .

gen InfExp_Mean_rv =.
gen InfExp_Var_rv =.

gen InfExp_Meanl1 =. 
gen InfExp_Varl1 =. 

*gen InfExp_Meanf0 =. 
*gen InfExp_Varf0 =. 
 

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




*******************************************************
**** Autoregression on the levels of population moments
********************************************************


tsset date

eststo clear 
foreach mom in Mean Var Disg FE{
   foreach var in SCE{
	 replace InfExp_`mom' = `var'_`mom'
    eststo `var'_`mom': reg InfExp_`mom' l(1/12).InfExp_`mom', vce(robust)
  } 
}

esttab using "${sum_table_folder}/autoregLvlM.csv", mtitles drop(_cons) se(%8.3f) scalars(N r2 ar2) replace


**********************************************************************
******** Autoregression on the first difference of population moments
 **********************************************************************

eststo clear
foreach mom in Mean Var Disg FE{
   foreach var in SCE{
    replace InfExp_`mom' = `var'_`mom'
    replace InfExp_`mom'_ch = InfExp_`mom'-l1.InfExp_`mom'
    eststo `var'_`mom': reg InfExp_`mom'_ch l(1/4).InfExp_`mom'_ch, vce(robust)
  }
}
esttab using "${sum_table_folder}/autoregDiffM.csv", mtitles drop(_cons) se(%8.3f) scalars(N r2 ar2) replace

***************
*** SCE Only **
***************

eststo clear

foreach mom in Mean Var{
   foreach var in SCE{
    replace InfExp_`mom' = `var'_`mom'
    replace InfExp_`mom'_ch = InfExp_`mom'-l1.InfExp_`mom'
	capture replace InfExp_`mom'_rv = `var'_`mom'_rv  /// caputure because FE and Disg has to rev

	eststo `var'_`mom'lvl: reg InfExp_`mom' l(1/2).InfExp_`mom' 
    eststo `var'_`mom'diff: reg InfExp_`mom'_ch l(1/2).InfExp_`mom'_ch  
	capture eststo `var'_`mom'_rv: reg InfExp_`mom'_rv l(1/2).InfExp_`mom'_rv 

  }
}
esttab using "${sum_table_folder}/autoregSCEM.csv", mtitles drop(_cons) se(%8.3f) scalars(N r2 ar2) replace
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

gen const=1

foreach mom in FE{
   foreach var in SCE{
      reg `var'_`mom' const
}
}

**********************************************
*** Forecast Error Efficiency Test      **
**********************************************

foreach mom in FE{
   foreach var in SCE{
   replace InfExp_Mean = `var'_Mean
   replace InfExp_`mom' = `var'_`mom'
   eststo `var'_`mom'_lag6: reg  InfExp_`mom' l(6/8).InfExp_Mean, robust
   eststo `var'_`mom'_arlag712: reg  InfExp_`mom' l(6/12).InfExp_`mom', robust
   eststo `var'_`mom'_arlag12: reg  InfExp_`mom' l(12).InfExp_`mom', robust
   eststo `var'_`mom'_lag12: reg  InfExp_`mom' l(12).InfExp_Mean, robust
 }
}
esttab using "${sum_table_folder}/FEEfficiencySCEM.csv", mtitles se(%8.3f) scalars(N r2 ar2)  replace
*/

***************************************************
*** Revision Efficiency Test Using Mean Revision **
***************************************************


tsset date

eststo clear

foreach var in SCE{
  foreach mom in Mean{
     replace InfExp_`mom'_rv =  `var'_`mom' - l24.`var'_`mom'1
	 replace InfExp_FE = `var'_FE
	 eststo `var'`mom'rvlv0: reg InfExp_`mom'_rv
     eststo `var'`mom'rvlv1: reg InfExp_`mom'_rv l24.InfExp_`mom'_rv l12.InfExp_FE,robust
	 eststo `var'`mom'rvlv2: reg  InfExp_`mom'_rv l25.InfExp_`mom'_rv l12.InfExp_FE, robust
	 eststo `var'`mom'rvlv3: reg  InfExp_`mom'_rv l26.InfExp_`mom'_rv l12.InfExp_FE, robust
 }
}

foreach var in SCE{
  foreach mom in Var{
     replace InfExp_`mom'_rv =  `var'_`mom' - l24.`var'_`mom'1
	 replace InfExp_FE2 = `var'_FE2

	 eststo `var'`mom'rvlv0: reg InfExp_`mom'_rv
     eststo `var'`mom'rvlv1: reg InfExp_`mom'_rv l24.InfExp_`mom'_rv l12.InfExp_FE2, robust
	 eststo `var'`mom'rvlv2: reg  InfExp_`mom'_rv l25.InfExp_`mom'_rv l12.InfExp_FE2, robust
	 eststo `var'`mom'rvlv3: reg  InfExp_`mom'_rv l26.InfExp_`mom'_rv l12.InfExp_FE2, robust
 }
}

esttab using "${sum_table_folder}/RVEfficiencySCEM.csv", mtitles b(%8.3f) se(%8.3f) scalars(N r2) sfmt(%8.3f %8.3f %8.3f) replace


/*
*******************************************************************
*** Revision Efficiency Test Using Mean Revision Fuhrer's Approach**
********************************************************************

foreach mom in Var{
   foreach var in SCE{
   replace InfExp_`mom'_rv =  `var'_`mom' - l12.`var'_`mom'1
   ttest InfExp_`mom'_rv =0
 }
}

eststo clear

foreach var in SCE{
  foreach mom in Mean{
     replace InfExp_`mom' = `var'_`mom'
     replace InfExp_`mom'l1 = `var'_`mom'l1
	 replace InfExp_`mom'f0 = `var'_`mom'f0
     eststo `var'`mom'rvlv1: reg InfExp_`mom'f0 InfExp_`mom'
	 test _b[InfExp_`mom']=1
	  scalar pvtest= r(p)
	 estadd scalar pvtest
	 eststo `var'`mom'rvlv2: reg InfExp_`mom'f0 l(0/1).InfExp_`mom'
	 test _b[InfExp_`mom']=1
	 scalar pvtest= r(p)
	 estadd scalar pvtest
	 eststo `var'`mom'rvlv3: reg InfExp_`mom'f0 l(0/2).InfExp_`mom'
     test _b[InfExp_`mom']=1
	 scalar pvtest= r(p)
	 estadd scalar pvtest
 }
}


foreach var in SCE{
  foreach mom in Var{
     replace InfExp_`mom' = `var'_`mom'
     replace InfExp_`mom'l1 = `var'_`mom'l1
	 replace InfExp_`mom'f0 = `var'_`mom'f0
     eststo `var'`mom'rvlv1: reg InfExp_`mom'f0 InfExp_`mom'
	 test _b[_cons]=0
	 scalar pvtest= r(p)
	 estadd scalar pvtest
	 eststo `var'`mom'rvlv2: reg InfExp_`mom'f0 l(0/1).InfExp_`mom'
	 test _b[_cons]=0
	 scalar pvtest= r(p)
	 estadd scalar pvtest
	 eststo `var'`mom'rvlv3: reg InfExp_`mom'f0 l(0/2).InfExp_`mom'
	 test _b[_cons]=0
	 scalar pvtest= r(p)
	 estadd scalar pvtest
 }
}

esttab using "${sum_table_folder}/RVEfficiencySCEM.csv", mtitles se(%8.3f) scalars(pvtest N r2) replace
*/

save "${folder}/InfExpM.dta",replace 


log close 
