*************************
**** Import SPF *********
*************************

clear
set more off

global mainfolder "/Users/Myworld/Dropbox/ExpProject/workingfolder"
global folder "/Users/Myworld/Dropbox/ExpProject/workingfolder/SurveyData"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"
global datafolder "${folder}/SPF"

cd ${datafolder}


*********************
*** Merge  Data *****
*********************

** Dispersion 


foreach var in NGDP UNEMP CPI CORECPI PCE COREPCE CPI5YR PCE5YR CPI10 PCE10{
import excel "${datafolder}/Dispersion.xlsx", sheet("`var'") cellrange(A10) firstrow clear 
save `var'.dta,replace 
}


clear 

use NGDP 
foreach var in UNEMP CPI CORECPI PCE COREPCE CPI5YR PCE5YR CPI10 PCE10{
merge 1:1 Survey_DateT using `var'
drop _merge 
rm "`var'.dta"
}

rm "NGDP.dta"

drop E-J

gen YEAR = substr(Survey_DateT,1,4)
gen QUARTER= substr(Survey_DateT,6,1)
drop Survey_DateT

foreach var of varlist _all{
destring(`var'),replace force 
}

save SPFDispersionPopQ,replace 


foreach var in "PRGDP" "PRPGDP" "PRUNEMP" "PRCCPI" "PRCPCE" "RECESS"{

import excel "${datafolder}/prob.xlsx", sheet("`var'") firstrow clear 

save `var'.dta,replace 

}

clear 

use PRGDP
merge 1:1 YEAR QUARTER using PRPGDP
drop _merge
merge 1:1 YEAR QUARTER using PRUNEMP 
drop _merge
merge 1:1 YEAR QUARTER using PRCCPI
drop _merge
merge 1:1 YEAR QUARTER using PRCPCE
drop _merge
merge 1:1 YEAR QUARTER using RECESS
drop _merge 
save SPFProbPopQ,replace 

rm "PRGDP.dta"
rm "PRPGDP.dta"
rm "PRUNEMP.dta"
rm "PRCCPI.dta"
rm "PRCPCE.dta"
rm "RECESS.dta"


foreach var in "PRGDP" "PRPGDP" "PRUNEMP" "PRCCPI" "PRCPCE"{

forval i=1/10{
label var `var'`i' "prob of current year `var' falling in a specific range"
}

forval i=11/20{
label var `var'`i' "prob of next year `var' falling in a specific range"
}
}

foreach var of varlist _all{
destring(`var'),replace force 
}

save,replace 

use SPFProbPopQ,clear 

merge 1:1 YEAR QUARTER using SPFDispersionPopQ
rename _merge dispersion_merge  

*********************
** Data structure ***
*********************

rename YEAR year 
rename QUARTER quarter 

gen date2=string(year)+"Q"+string(quarter)
gen date3= quarterly(date2,"YQ")
drop date2 

tsset date3
rename date3 date

order date year quarter 
tsset date,q

save "${datafolder}/InfExpSPFProbPopQ.dta",replace

rm "SPFProbPopQ.dta"
rm "SPFDispersionPopQ.dta"

/*
foreach var in "PRGDP" "PRPGDP" "PRUNEMP" "PRCCPI" "PRCPCE"{
gen `var'_epy0 =. 
gen `var'_epy1 =.
replace `var'_epy0 = -0.01*(`var'1*log(`var'1)+`var'2*log(`var'2) ///
                       +`var'3*log(`var'3)+`var'4*log(`var'4) ///
					   +`var'5*log(`var'5)+`var'6*log(`var'6) ///
					   +`var'7*log(`var'7)+`var'8*log(`var'8) ///
					   +`var'9*log(`var'9)+`var'10*log(`var'10))+log(100)*0.01

label var `var'_epy0 "Entropy Current Year `var'"

replace `var'_epy1 = -0.01*(`var'11*log(`var'11)+`var'12*log(`var'12) /// 
                      +`var'13*log(`var'13)+`var'14*log(`var'14) ///
					  +`var'15*log(`var'15)+`var'16*log(`var'16) ///
					  +`var'17*log(`var'17)+`var'18*log(`var'18) ///
					  +`var'19*log(`var'19)+`var'20*log(`var'20))+log(100)*0.01
label var `var'_epy1 "Entropy Next Year `var'"
}
*/
