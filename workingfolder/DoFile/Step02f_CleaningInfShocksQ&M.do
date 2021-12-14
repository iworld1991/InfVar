*******************************************************************************
***  This do file first works with the inflation shock data file, including  **
***  cleaning, relabeling and normalizing shocks. It then saves a clean      **
***  InfShocksClean to be used for individual IR analysis.                   **
***   Then it will be merged with population survey data and plot all kinds of impulse *
***   responses. Be careful with the period filter. 
********************************************************************************


clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/irf_log",replace


**************************************************
*** Clean inflation shock data from Python for  **
*** both quarterly and monthly                 ***
**************************************************


*******************************************
* Generate monthly inflation shock data   *
*******************************************

import excel "${mainfolder}/OtherData/2MPShocksJW.xls", sheet("data") firstrow allstring clear

gen date = daily(Date,"MDY")
drop Date
gen year = year(date)
gen month = month(date)
gen date_str=string(year)+"M"+string(month)
gen dateM = monthly(date_str,"YM")


rename date dateD
rename dateM date
format date %tm

keep year month date dateD MP1 ED4 ED8
sort date dateD year month MP1 ED4 ED8

foreach var in MP1 ED4 ED8{
destring `var',force replace 
egen `var'M = sum(`var'), by(year month) 
label var `var'M "Cumulated sum of `var'"
drop `var'
rename `var'M `var'
}

duplicates drop year month,force 

** Complete all months in the sameple since MP shocks are only recorded for some months

tsset date 

tsfill
replace year = yofd(dofm(date))
replace month = month(dofm(date))

sort date

foreach var in MP1 ED4 ED8{
   replace `var'=0 if `var'==.
}


duplicates report date 

save "${mainfolder}/OtherData/MPShocksM.dta",replace 


*************************
**  Oil Price Shock *****
**************************

clear

import excel "${mainfolder}/OtherData/OilShock.xls", sheet("data") firstrow allstring clear

keep observation_date OPShock

destring(OPShock),force replace 

gen date = daily(observation_date,"MDY")
gen year = year(date) 
gen month =month(date) 

gen date_str=string(year)+"M"+string(month)
gen dateM = monthly(date_str,"YM")

drop date
rename dateM date
tsset date
format date %tm

** Normalize Oil shock

egen OPShock_sd = sd(OPShock)
gen OPShock_nom = OPShock/OPShock_sd
label var OPShock_nom "Oil price shock(normalized)"



save "${mainfolder}/OtherData/OPShocksM.dta",replace 

**************************************
***  Merge Shocks and Inflation *****
**************************************

merge 1:1 year month using "${mainfolder}/OtherData/MPShocksM.dta",keep(master match)
rename _merge MPshock_merge


* Merge with inflation 
merge 1:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match master using)
rename _merge InfM_merge 


** Label and rename

rename OPShock_nom op_shock 
label var op_shock "Oil price shock (Hamilton 1996)"
rename MP1 mp1_shock
label var mp1_shock "Unexpected change in federal funds rate"
label var ED4 "1-year ahead future-implied change in federal funds rate"
label var ED8 "2-year ahead future-implied change in federal funds rate"


** MP-path shock 

foreach ed in ED4 ED8{
  reg `ed' mp1_shock
  predict `ed'_shock, residual
label var `ed'_shock "Unexpected shock to future federal funds rate"
}

** Normorlize MP shcoks

foreach var in mp1 ED4 ED8{
  egen `var'_shock_sd =sd(`var'_shock)
  gen `var'ut_shock = `var'_shock/`var'_shock_sd 
  local lb : var label `var'_shock
  label var `var'ut_shock "Normalized `lb'"
}

** Absolute values of the shocks

foreach var in op mp1ut ED8ut{
gen `var'_abshock = abs(`var'_shock)
local lb: var label `var'_shock
label var `var'_abshock "Absolute value of `lb'"
} 

** Generated unidentified shocks. 

tsset date

eststo clear

foreach Inf in CPIAU CPICore PCE PCECore{ 
   reg Inf1y_`Inf' l(1/4).Inf1y_`Inf' l(0/1).op_shock l(0/1).mp1ut_shock l(0/1).ED8ut_shock
   predict `Inf'_uid_shock, residual
   label var `Inf'_uid_shock "Unidentified shocks to inflation"
 }


*Save a dta file for individual monthly IR analysis **

*******************************
**  Plot Shocks for checking **
********************************

tsset date
format date %tm


foreach var in mp1ut ED4ut ED8ut op{
local lb: variable label `var'_shock
tsline(`var'_shock) if `var'_shock!=., title("`lb'") ytitle("")
graph export "${mainfolder}/graphs/shocks/`var'_shock",as(png) replace 
}

save "${mainfolder}/OtherData/InfShocksMClean.dta",replace
  
rm "${mainfolder}/OtherData/MPShocksM.dta"
rm "${mainfolder}/OtherData/OPShocksM.dta"


***********************************************
* Quarterly. originally generated from Python *
* Will be generated using stata later. *******
***********************************************

use "${mainfolder}/OtherData/InfShocksQ.dta",clear 

drop index 

*** Date 
gen date_str=string(year)+"Q"+string(quarter) 

gen date = quarterly(date_str,"YQ")
format date %tq 

drop date_str

order date year quarter month

** Time series 

tsset date 


* Merge with inflation 
merge 1:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match master)
rename _merge InfM_merge 


** Label and rename

label var pty_shock "Technology shock(Gali 1999)"
label var hours_shock "Non-technology shock(Gali 1999)"
label var inf_shock "Shock to inflation(Gali 1999)"
label var pty_max_shock "Technology shock (Francis etal 2014)"
label var news_shock "News shock(Sims etal.2011)"
rename OPShock_nm op_shock 
label var op_shock "Oil price shock (Hamilton 1996)"
rename MP1 mp1_shock
label var mp1_shock "unexpected change in federal funds rate"
label var ED4 "1-year ahead future-implied change in federal funds rate"
label var ED8 "2-year ahead future-implied change in federal funds rate"

** MP-path shock 

foreach ed in ED4 ED8{
  reg `ed' mp1_shock
  predict `ed'_shock, residual
label var `ed'_shock "unexpected shock to future federal funds rate"
}

** Normorlize MP shcoks

foreach var in mp1 ED4 ED8{
  egen `var'_shock_sd =sd(`var'_shock)
  gen `var'ut_shock = `var'_shock/`var'_shock_sd 
  local lb : var label `var'_shock
  label var `var'ut_shock "Normalized `lb'"
}

** Absolute values of the shocks

foreach var in op pty mp1ut ED8ut{
gen `var'_abshock = abs(`var'_shock)
local lb: var label `var'_shock
label var `var'_abshock "Absolute value of `lb'"
} 



** Generated unidentified shocks. 

tsset date

eststo clear

foreach Inf in CPIAU CPICore PCE PCECore{ 
   reg Inf1y_`Inf' l(1/4).Inf1y_`Inf' l(0/1).pty_shock l(0/1).op_shock l(0/1).mp1ut_shock l(0/1).ED8ut_shock
   predict `Inf'_uid_shock, residual
   label var `Inf'_uid_shock "Unidentified shocks to inflation"
 }

 
*Save a dta file for individual IR analysis **
 
save "${mainfolder}/OtherData/InfShocksQClean.dta",replace 
