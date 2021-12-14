*************************
**** Import SPF *********
*************************

clear
set more off
global folder "/Users/Myworld/Dropbox/ExpProject/workingfolder/SurveyData"
global datafolder "/SPF/individual"

cd ${folder}/${datafolder}
 
import excel "Individual_CPI.xlsx", sheet("CPI") firstrow clear 
save SPFCPI,replace

import excel "Individual_PCE.xlsx", sheet("PCE") firstrow clear 
save SPFPCE,replace

import excel "Individual_COREPCE.xlsx", sheet("COREPCE") firstrow clear
save SPFCorePCE,replace

import excel "Individual_CORECPI.xlsx", sheet("CORECPI") firstrow clear 
save SPFCoreCPI,replace

use SPFPCE,clear
merge 1:1 YEAR QUARTER INDUSTRY ID using SPFCPI
rename _merge SPFCPI_merge
merge 1:1 YEAR QUARTER INDUSTRY ID using SPFCorePCE
rename _merge SPFCorePCE_merge
merge 1:1 YEAR QUARTER INDUSTRY ID using SPFCoreCPI
rename _merge SPFCoreCPI_merge

rm "SPFPCE.dta"
rm "SPFCPI.dta"
rm "SPFCorePCE.dta"
rm "SPFCoreCPI.dta"



gen date_str = string(YEAR)+"Q"+string(QUARTER)
gen date = quarterly(date_str,"YQ")
format date %tq
drop date_str
rename YEAR year
rename QUARTER quarter 
rename date dateQ

gen month =.
replace month =1 if quarter==1
replace month =4 if quarter==2
replace month =7 if quarter ==3
replace month =9 if quarter==4

order date dateQ year quarter month
xtset ID date

***********************************
*********Destring and Labels ******
***********************************

foreach var in PCE CPI CORECPI COREPCE{
destring `var'1,force replace 
label var `var'1 "inflation `var' from q-2 to q-1"
destring `var'2,force replace 
label var `var'2 "inflation `var' from q-1 to q"
destring `var'3,force replace 
label var `var'3 "inflation `var' from q to q+1"
destring `var'4,force replace 
label var `var'4 "inflation `var' from q+1 to q+2"
destring `var'5,force replace 
label var `var'5 "inflation `var' from q+2 to q+3"
destring `var'6,force replace 
label var `var'6 "inflation `var' from q+3 to q+4"

destring `var'A,force replace 
label var `var'A "inflation `var' from y-1 to y"

destring `var'B,force replace 
label var `var'B "inflation `var' from y to y+1"

destring `var'C,force replace 
label var `var'C "inflation `var' from y+1 to y+2"

}



***********************************************
********* Computing annualized rate ***********
***********************************************


foreach var in PCE CPI CORECPI COREPCE{
gen `var'1y = 100*(((1+`var'3/100)*(1+`var'4/100)*(1+`var'5/100)*(1+`var'6/100))^0.25-1)
label var `var'1y "inflation from q to q+4"
}


*****************************************
*** Compute Cross-sectional Moments *****
*****************************************

* merge inflation data to compute forecast errors 

merge m:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match master)
rename _merge inflation_merge 


** indivdiual forecast errors 

gen CPI_fe = CPI1y - Inf1yf_CPIAU
label var CPI_fe "1-yr-ahead forecast error(SPF Core CPI)"

gen PCE_fe = PCE1y - Inf1yf_PCE
label var PCE_fe "1-yr-ahead forecast error(SPF Core PCE)"

gen CORECPI_fe = CORECPI1y - Inf1yf_CPICore
label var CORECPI_fe "1-yr-ahead forecast error(SPF Core CPI)"

gen COREPCE_fe = COREPCE1y - Inf1yf_PCECore
label var COREPCE_fe "1-yr-ahead forecast error(SPF Core PCE)"

** Redo the date to quarterly 

drop date 
gen date_str = string(year)+"Q"+string(quarter)
gen date = quarterly(date_str,"YQ")
format date %tq
xtset ID date

** variances and autocovariances of forecast errors

foreach var in CPI PCE CORECPI COREPCE{

** variances
sort date
egen `var'_fe_sd = sd(`var'_fe), by(date)
gen `var'_fe_var = `var'_fe_sd^2
label var `var'_fe_sd "Variances of 1-yr-ahead forecast errors"


** autocovariance

xtset ID date 
gen `var'_fe_l1 = l1.`var'_fe
label var `var'_fe_l1 "lagged forecast error"

sort date 
egen `var'_fe_atv = corr(`var'_fe `var'_fe_l1), covariance by(date)
label var `var'_fe_atv "Auto covariance of forecast errors"

** autocovariance of forecast

sort ID date
gen `var'1y_l1 = l1.`var'1y
label var `var'1y_l1 "lagged forecasts"

sort date 
egen `var'_atv = corr(`var'1y `var'1y_l1), covariance by(date)
label var `var'_atv "Auto covariance of forecasts"

** variances of forecasts, i.e. disagreement
sort ID date

egen `var'_std = sd(`var'1y), by(year quarter)
gen `var'_disg = `var'_std^2 
label var `var'_disg "disagreements of `var'"
egen `var'_ct50 = median(`var'1y), by(year quarter) 
label var `var'_ct50 "Median of `var'"
}


save InfExpSPFPointIndQ,replace

collapse (mean) PCE1y CPI1y CORECPI1y COREPCE1y ///
                PCE_disg CPI_disg CORECPI_disg COREPCE_disg ///
				PCE_atv CPI_atv CORECPI_atv COREPCE_atv ///
				PCE_fe_var CPI_fe_var CORECPI_fe_var COREPCE_fe_var ///
				PCE_fe_atv CPI_fe_atv CORECPI_fe_atv COREPCE_fe_atv ///
				PCE_ct50 CPI_ct50 CORECPI_ct50 COREPCE_ct50, by(year quarter month)

label var PCE1y "1-yr-ahead PCE inflation"
label var CPI1y "1-yr-ahead CPI inflation"
label var COREPCE1y "1-yr-ahead Core PCE inflation"
label var CORECPI1y "1-yr-ahead Core CPI inflation"

label var PCE_disg "Disagreements in 1-yr-ahead PCE inflation"
label var CPI_disg "Disagreements in 1-yr-ahead CPI inflation"
label var COREPCE_disg "Disagreements in 1-yr-ahead Core PCE inflation"
label var CORECPI_disg "Disagreements in 1-yr-ahead Core CPI inflation"


label var PCE_atv "Autocovariance of 1-yr-ahead PCE inflation"
label var CPI_atv "Autocovariance of 1-yr-ahead CPI inflation"
label var COREPCE_atv "Autocovariance of in 1-yr-ahead Core PCE inflation"
label var CORECPI_atv "Autocovariance of in 1-yr-ahead Core CPI inflation"


label var PCE_fe_var "Variance of 1-yr-ahead PCE forecast errors"
label var CPI_fe_var "Variance of 1-yr-ahead CPI forecast errors"
label var COREPCE_fe_var "Variance of in 1-yr-ahead Core PCE forecast errors"
label var CORECPI_fe_var "Variance of in 1-yr-ahead Core CPI forecast errors"


label var PCE_fe_atv "Autocovariance of 1-yr-ahead PCE forecast errors"
label var CPI_fe_atv "Autocovariance of 1-yr-ahead CPI forecast errors"
label var COREPCE_fe_atv "Autocovariance of in 1-yr-ahead Core PCE forecast errors"
label var CORECPI_fe_atv "Autocovariance of in 1-yr-ahead Core CPI forecast errors"

label var PCE_ct50 "Median 1-yr-ahead PCE inflation"
label var CPI_ct50 "Median 1-yr-ahead CPI inflation"
label var COREPCE_ct50 "Median 1-yr-ahead Core PCE inflation"
label var CORECPI_ct50 "Median 1-yr-ahead Core CPI inflation"

save InfExpSPFPointPopQ,replace  
