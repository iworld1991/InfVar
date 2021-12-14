************************************************
***  This do file is optional. The purpose is   *
**    to redo SVAR with long-run restriction in *
**    in Stata as a robust check for the tech   *
**    shock estimated in python codes. It also  *
**    generates IR plots that can be compared.  *
************************************************ 

clear
global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "${mainfolder}/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 
capture log close
log using "${mainfolder}/svar",replace

use "${mainfolder}/OtherData/LRVar.dta",clear

** clean data

drop index
order year quarter 
gen date_str= string(year)+"Q"+string(quarter)
gen date = quarterly(date_str,"YQ")
order date year quarter 

tsset date 

** Label 

label var DLPROD1 "Log Diff of Labor Productivity"
label var DLHOURS "Log Diff of Total Hours"
label var CPIAU "Headline inflation"
label var CoreCPI "Core CPI Inflation"
label var PCE "PCE Inflation"



** SVAR with long-run restriction 

matrix C = (., 0 ,0\ .,.,0 \.,.,.)

foreach inf in CPIAU CoreCPI PCE{
 svar DLPROD1 DLHOURS `inf', lags(1/4) lreq(C)
 irf create lr`inf', set(lrirf,replace) step(10) bs replace
 irf graph sirf, yline(0,lcolor(black)) xlabel(0(1)10) byopts(yrescale)
 graph export "${sum_graph_folder}/irf/other/LRSVAR_`inf'", as(png) replace
}
log close
