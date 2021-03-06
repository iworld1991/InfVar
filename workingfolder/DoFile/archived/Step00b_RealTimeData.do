

******************************************
** Importing and cleaning inflation series 
******************************************
clear
set more off 

global mainfolder "/Users/Myworld/Dropbox/InfVar/workingfolder"
global folder "${mainfolder}/SurveyData/"
global surveyfolder "NYFEDSurvey"
global otherdatafolder "OtherData"
global sum_graph_folder "${mainfolder}/${otherdatafolder}"



cd "${mainfolder}/${otherdatafolder}"

import excel "${mainfolder}/${otherdatafolder}/RealTimeData/pcpiMvMd.xlsx", sheet("pcpi") firstrow

** get all variables that need to destring 
quietly describe, varlist
local vars `r(varlist)'
local omit DATE
local all_var : list vars - omit

foreach var in `all_var'{
destring `var', force replace 
}


gen year = substr(DATE,1,4)
gen month = substr(DATE,6,2)
gen date_str=year+"m"+ month
gen date2=monthly(date_str,"YM")
format date2 %tm
drop date_str DATE 
rename date2 date 
destring year, replace force
destring month, replace force 
order date year month 

keep if date >= monthly("1998m11","YM")

save InfCPIMRealTime,replace 

clear
import excel "${mainfolder}/${otherdatafolder}/RealTimeData/pcpixMvMd.xlsx", sheet("pcpix") firstrow


** get all variables that need to destring 
quietly describe, varlist
local vars `r(varlist)'
local omit DATE
local all_var : list vars - omit

foreach var in `all_var'{
destring `var', force replace 
}

gen year = substr(DATE,1,4)
gen month = substr(DATE,6,2)
gen date_str=year+"m"+ month
gen date2=monthly(date_str,"YM")
format date2 %tm
drop date_str DATE 
rename date2 date 
destring year, replace force
destring month, replace force 
order date year month 

keep if date >= monthly("1998m11","YM")

save InfCPICMRealTime,replace 

