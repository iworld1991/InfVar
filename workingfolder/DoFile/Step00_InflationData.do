

******************************************
** Importing and cleaning inflation series 
******************************************
clear


global mainfolder "/Users/Myworld/Dropbox/InfVar/workingfolder"
*global folder "${mainfolder}/SurveyData/"
*global surveyfolder "NYFEDSurvey"
global otherdatafolder "OtherData"
global sum_graph_folder "${mainfolder}/${otherdatafolder}"


*** download following series from Fred: CPIAUCSL CPILFESL PCEPI PCEPILFE
*** save it as PriceIndexM.csv

cd "${mainfolder}/${otherdatafolder}"


import delimited "PriceIndexM.csv"

gen year = substr(date,1,4)
gen month= substr(date,6,2)

gen date_str=year+"m"+month 
gen date2=monthly(date_str,"YM")
format date2 %tm
drop date_str date 
rename date2 date 

destring(year),replace force
destring(month),replace force

label var cpiaucsl "CPI index for all urban households(seasonally adjusted)"
label var cpilfesl "CPI index for all urban households excl food and energy (seaonsally adjusted)"
label var pcepi "PCE index: chain-type (sesonally adjusted)"
label var pcepilfe "PCE index: chain-type exl food and energe(sesonally adjusted)"

rename cpiaucsl CPIAU
rename cpilfesl CPICore
rename pcepi PCE
rename pcepilfe PCECore

order date year month

tsset date


foreach var in  CPIAU CPICore PCE PCECore{
   
   ** computing yoy inflation and foreward inflation
   gen Inf1y_`var' = (`var'- l12.`var')*100/l12.`var'
   label var Inf1y_`var' "yoy inflation based on `var'"
   gen Inf1yf_`var' = (f12.`var'- `var')*100/`var'
   label var Inf1yf_`var' "1-year-ahead realized inflation"
}

*********************************
**   Plot Inflation Series *****
*********************************


tsline Inf1y_CPIAU Inf1y_CPICore Inf1y_PCE Inf1y_PCECore, title("Annual Inflation in the U.S.") legend(label(1 "Headline CPI") label(2 "Core CPI") label(3 "PCE") label(4 "Core PCE"))
graph export "${mainfolder}/${otherdatafolder}/Inflation.png", as(png) replace 

save InfM,replace 
