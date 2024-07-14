****************************************************
***   This do files cleans SCE individual density **
***   forecasts and moments. It exclude the top and *
***   bottom 5 percentiles of mean and uncertainty. *
***   It also plots histograms of mean forecast and *
***   uncertainty. **********************************
*****************************************************


clear
global mainfolder "/Users/tao/Dropbox/InfVar/workingfolder"
global folder "${mainfolder}/SurveyData/"
global big_data_folder "/Users/tao/Dropbox/InfVar-local/workingfolder/SurveyData/"
global sum_graph_folder "${mainfolder}/graphs/pop"
global sum_table_folder "${mainfolder}/tables"

cd ${folder}
pwd
set more off 

*** import SCE xls to dta

use "${folder}/SCE/NYFED_SCE_13_16.dta",clear
append using "${folder}/SCE/NYFED_SCE_17_19.dta",force
append using  "${folder}/SCE/NYFED_SCE_20.dta",force

** if no data is found, first download SCE data using DownloadSCE.ipynb 

sort date
unique userid

********************
*** Date format ****
********************
gen date_str = string(date)
gen year = substr(date_str,1,4)
gen month = substr(date_str,5,2)
gen date2=year+"m"+month
gen date3= monthly(date2,"YM")
format date3 %tm
drop date_str date2 date
rename date3 date
destring year, replace
destring month, replace 
order date year month
xtset userid date

**********************
** Label variables ***
**********************

label var Q1 "finance better or worser(5 vals) from y-1 to y"
label var Q2 "finance better or worser(5 vals) from y to y+1"
label var Q3 "chance of moving (%)"
label var Q4new "chance of UE higher from y to y+1(0-1)"
label var Q5new "chance of saving interest rate higher from y to y+1(%)"
label var Q6new "chance of stock market up from y to y+1(%)"
label var Q8v2 "inflation or deflation from y to y+1 (1/0)"
label var Q8v2part2 "inflation(or deflation) from y to y+1 (%) "
label var Q9_cent25 "25 percentile of inflation from y to y+1(%)"
label var Q9_cent50 "50 percentile of inflation from y to y+1(%)"
label var Q9_cent75 "75 percentile of inflation from y to y+1(%)"
label var Q9_var "var of inflation from y to y+1"
label var Q9_iqr "25/75 inter-quantile range of inflation from y to y+1(%)"
label var Q9_mean "mean of inflation from y to y+1(%)"
label var Q9_probdeflation "prob of deflation from y to y+1 (0-1)"
label var Q9_bin1 "density: >12% inflation from y to y+1(%)"
label var Q9_bin2 "density: 8%-12% inflation from y to y+1(%)"
label var Q9_bin3 "density: 4%-8% inflation from y to y+1(%)"
label var Q9_bin4 "density: 2%-4% inflation from y to y+1(%)"
label var Q9_bin5 "density: 0%-2% inflation from y to y+1(%)"
label var Q9_bin6 "density: -2%-0% inflation from y to y+1(%)"
label var Q9_bin7 "density: -4%- -2% inflation from y to y+1(%)"
label var Q9_bin8 "density: -8%--4% inflation from y to y+1(%)"
label var Q9_bin9 "density: -12%- -8% inflation from y to y+1(%)"
label var Q9_bin10 "density: <-12% inflation from y to y+1(%)"
label var Q9bv2 "inflation or deflation from y+1 to y+2 (1/0)"
label var Q9bv2part2 "inflation(or deflation) from y+1 to y+2 (%)"
label var Q9c_cent25 "25 percentile of inflation from y+1 to y+2(%)"
label var Q9c_cent50 "50 percentile of inflation from y+1 to y+2(%)"
label var Q9c_cent75 "75 percentile of inflation from y+1 to y+2(%)"
label var Q9c_var "var of inflation from y+1 to y+2"
label var Q9c_mean "mean of inflation from y+1 to y+2(%)"
label var Q9c_iqr "25/75 inter-quantile range of inflation from y+1 to y+2(%)"
label var Q9c_probdeflation "prob of deflation from y+1 to y+2 (0-1)"
label var Q9c_bin1 "density: >12% inflation from y+1 to y+2(%)"
label var Q9c_bin2 "density: 8%-12% inflation from y+1 to y+2(%)"
label var Q9c_bin3 "density: 4%-8% inflation from y+1 to y+2(%)"
label var Q9c_bin4 "density: 2%-4% inflation from y+1 to y+2(%)"
label var Q9c_bin5 "density: 0%-2% inflation from y+1 to y+2(%)"
label var Q9c_bin6 "density: -2%-0% inflation from y+1 to y+2(%)"
label var Q9c_bin7 "density: -4%- -2% inflation from y+1 to y+2(%)"
label var Q9c_bin8 "density: -8%--4% inflation from y+1 to y+2(%)"
label var Q9c_bin9 "density: -12%- -8% inflation from y+1 to y+2(%)"
label var Q9c_bin10 "density: <-12% inflation from y+1 to y+2(%)"
label var Q10_1 "current empsituations:full-time"
label var Q10_2 "current emp situations:part-time"
label var Q10_3 "current emp situations: not working but wants to work"
label var Q10_4 "current emp situations: temporary laid-off"
label var Q10_5 "current emp situations: on sick or other leave"
label var Q10_6 "current emp situations: permanently disabled/unable to work"
label var Q10_7 "current emp situations: retiree or early retiree"
label var Q10_8 "current emp situations: student or in training"
label var Q10_9 "current emp situations: homemaker"
label var Q10_10 "current emp situations: others"
label var Q11 "number of jobs"
label var Q12new "work for someone or self-employed(1/0)"
label var ES1_1 ""
label var ES1_2 ""
label var ES1_3 ""
label var ES1_4 ""
label var ES2 ""
label var ES3new ""
label var ES4 ""
label var ES5 ""
label var Q13new "chance of losing job from y to y+1(%)"
label var Q14new "chance of voluntarily leaving the job from y to y+1(%)"
label var Q15 "currently looking for a job (1/0)"
label var Q16 "duration of unemployment (months)"
label var Q17new "chance of finding and accepting a job from y to y+1(%)"
label var Q18new "chance of finding and accepting a job from m to m+3"
label var Q19 "duration of out of work(%)"
label var Q20new "chance of starting looking for a job from y to y+1(%)"
label var Q21new "chance of starting looking for a job from m to m+3(%)"
label var Q22new "chance of finding a new job if losing the current one within 3 months"
label var Q23v2 "earning increase/decrease from the same job/time/place from y to y+1(%)"
label var Q23v2part2 "change of earning from the same job/time/place from y to y+1(%)"
label var Q24_cent25 "25 percentile of earning growth of same job/time/place from y to y+1(%)"
label var Q24_cent50 "50 percentile of earning growth of same job/time/place from y to y+1(%)"
label var Q24_cent75 "75 percentile of earning growth of same job/time/place from y to y+1(%)"
label var Q24_var "var of earning growth of same job/time/place from y to y+1(%)"
label var Q24_mean "mean of earning growth of same job/time/place from y to y+1(%)"
label var Q24_iqr "25/75 inter-quantile range of earning growth of same job/time/place from y to y+1(%)"
label var Q24_probdeflation "???"
label var Q24_bin1 ">12% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin2 "8%-12% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin3 "4%-8% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin4 "2%-4% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin5 "0-2% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin6 "-2%-0 earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin7 "-4%- -2% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin8 "-8% - -4% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin9 "-12%- -8% earning growth of same job/time/place from y to y+1(%)"
label var Q24_bin10 "<-12% earning growth of same job/time/place from y to y+1(%)"
label var Q25v2 "increase/decrease of total household income from y to y+1(1/0)"
label var Q25v2part2 "increase of total household income from y to y+1(%)"
label var Q26v2 "increase/decrease of total household spending from y to y+1(1/0)"
label var Q26v2part2 "increase of total household spending from y to y+1(%)"
label var Q27v2 "increase/decrease of total household tax payment given the same income (1/0)"
label var Q27v2part2 "increase of total household tax payment given the same income (%)"
label var Q28 "easier/harder to get credit/loan from y-1 to y(1-5)"
label var Q29 "easier/harder to get credit/loan from y to y+1(1-5)"
label var Q30new "chance of non-payment of debt from m to m+3(%)"
label var Q31v2 "increase/decrease of nationwide house price(1/0)"
label var Q31v2part2 "increase of nationwide house price(%)"
label var C1_cent25 "25 percentile of increase of nationwide house price(%)"
label var C1_cent50 "50 percentile of increase of nationwide house price(%)"
label var C1_cent75 "75 percentile of increase of nationwide house price(%)"
label var C1_var "var of increase of nationwide house price"
label var C1_mean "mean of increase of nationwide house price(%)"
label var C1_iqr "25/75 inter-quantile range  of increase of nationwide house price(%)"
label var C1_probdeflation "??"
label var C1_bin1 ">12% increase in nationwide house price(%)"
label var C1_bin2 "8%-12% increase in nationwide house price(%)"
label var C1_bin3 "4%-8% increase in nationwide house price(%)"
label var C1_bin4 "2%-4% increase in nationwide house price(%)"
label var C1_bin5 "0-2% increase in nationwide house price(%)"
label var C1_bin6 "-2%-0% increase in nationwide house price(%)"
label var C1_bin7 "-4%- -2% increase in nationwide house price(%)"
label var C1_bin8 "-8%- -4% increase in nationwide house price(%)"
label var C1_bin9 "-12%- -8% increase in nationwide house price(%)"
label var C1_bin10 "<-12% increase in nationwide house price(%)"
label var C2 "increase/decrease in nationwide house price from y+1 to y+2(1/0)"
label var C2part2 "increase in nationwide house price from y+1 to y+2(%)"
label var C3 "increase/decrease of U.S. gov debt from y to y+1(1/0)"
label var C3part2"increase of U.S. gov debt from y to y+1(%)"
label var C4_1 "increase of a gallon of gas from y to y+1(%)"
label var C4_2 "increase of food price from y to y+1(%)"
label var C4_3 "increase of a medical care from y to y+1(%)"
label var C4_4 "increase of college education from y to y+1(%)"
label var C4_5 "increase of price of renting a typical house/apt from y to y+1(%)"
label var C4_6  "increase of gold price from y to y+1(%)"
label var QNUM1 "num q 1: (correct if ==150)"
label var QNUM2 ""
label var QNUM3 ""
label var QNUM5 ""
label var QNUM6 ""
label var QNUM8 ""
label var QNUM9 ""
label var Q32 "age(in years)"
label var Q33 "gender"
label var Q34 "hispanic/latino/spanish(1/0)"
label var Q35_1 "race: white"
label var Q35_2 "race: black/african american"
label var Q35_3 "race: american indian/alaska native"
label var Q35_4 "race: asian"
label var Q35_5 "race: native hawaiian or other pacific islander"
label var Q35_6 "race: other"
label var Q36 "education (1-8 low to high, 9 other)"
label var Q37 "months of current work"
label var Q38 "living with partner/not (1/0)"
label var HH2_1 "emp of partner: full-time for someone"
label var HH2_2 "emp of partner: part-time for someone"
label var HH2_3 "emp of partner: self-employed"
label var HH2_4 "emp of partner: not working but wants to work"
label var HH2_5 "emp of partner: temporary laid-off"
label var HH2_6 "emp of partner: on sick or other leave"
label var HH2_7 "emp of partner: permanently disabled/unable to work"
label var HH2_8 "emp of partner: retiree or early retiree"
label var HH2_9 "emp of partner: student or in training"
label var HH2_10 "emp of partner: homemaker"
label var HH2_10 "emp of partner: other"
label var _STATE "state (2-digit code)"
label var Q41 "years of living in the current residence"
label var Q42 "years of living in the current states"
label var Q43 "own/rent/other a house(1/2/3)"
label var Q43a "my/spounser/both's name under which current residence is owned/rent"
label var Q44 "own other homes(1/0)"
label var Q45b "health condition(1-5 from good to poor)"
label var Q45new_1 ""
label var Q45new_2 ""
label var Q45new_3 ""
label var Q45new_4 ""
label var Q45new_5 ""
label var Q45new_6 ""
label var Q45new_7 ""
label var Q45new_8 ""
label var Q45new_9 ""
label var Q46 "financial decision making of household(1-5, together to individual)"
label var Q47 "total pre-tax household income from y-1 to y(1-11, low to high)"
label var D1 "same household as last year(1/0)"
label var D3 "date of moving to the current residence(month/year)"
label var D6 "total pre-tax household income from y-1 to y(1-11,low to high)"
label var D2new_1 "memeber in the current resident:spounse/partner"
label var D2new_2 "memeber in the current resident: child >25"
label var D2new_3 "memeber in the current resident: child 18-24"
label var D2new_4 "memeber in the current resident: child 6-17"
label var D2new_5 "memeber in the current resident: child <=5 "
label var D2new_6 "memeber in the current resident: own/sponse's parents"
label var D2new_7 "memeber in the current resident:other relatives"
label var D2new_8 "memeber in the current resident:non-relatives"
label var DSAME "same job as last year in survey"
label var DQ38 "living as a partner or married with some one(1/0)"
label var Q48 "interesting/uninteresting of the questions in the survey(1/0)"


*************************
*** Exclude outliers *****
*************************

local Moments Q9_mean Q9_var Q9c_mean Q9c_var

foreach var in `Moments'{
      egen `var'p5=pctile(`var'),p(5)
	  egen `var'p95=pctile(`var'),p(95)
	  replace `var' = . if `var' <`var'p5 | (`var' >`var'p95 & `var'!=.)
}



*********************************
** demographic variables  ******
*********************************

gen HHinc=D6  
gen age = Q32 
gen gender =Q33
gen age2 = age^2
gen educ = Q36

gen edu_g = . 
replace edu_g = 1 if educ==1
replace edu_g = 2 if educ==2 | educ ==3 | educ == 4
replace edu_g = 3 if educ <=9 & educ>4

label var edu_g "education group"
label define edu_glb 1 "HS dropout" 2 "HS graduate" 3 "College/above"
label value edu_g edu_glb


*****************************************
*** Compute Cross-sectional Moments *****
*****************************************

* merge inflation data to compute forecast errors 

merge m:1 year month using "${mainfolder}/OtherData/InfM.dta",keep(match master)
rename _merge inflation_merge 

** indivdiual forecast errors 

gen Q9_fe = Q9_mean - Inf1yf_CPIAU
label var Q9_fe "Forecast error of 1-yr-ahead inflation"


*** exclude individual fixed effects

gen lQ9_var = log(Q9_var)
** run regresson on the log variance so that the residual is positive

foreach var in Q9_mean Q9_fe lQ9_var{
areg `var', a(userid)
predict `var'_rd, residuals
}

gen Q9_var_rd = exp(lQ9_var_rd)
label var Q9_var_rd "Uncertainty of 1-yr-ahead inflation exl id FE"

** variances of forecast errors 

egen Q9_fe_sd = sd(Q9_fe), by(date)
gen Q9_fe_var = Q9_fe_sd^2
label var Q9_fe_sd "Variances of 1-yr-ahead forecast errors"

egen Q9_fe_rd_sd = sd(Q9_fe_rd), by(date)
gen Q9_fe_rd_var = Q9_fe_rd_sd^2
label var Q9_fe_rd_var "Variances of 1-yr-ahead forecast errors exl id FE"

** autocovariance of forecast errors

sort userid date
gen Q9_fe_l1 = l1.Q9_fe
label var Q9_fe_l1 "lagged forecast error"

sort date 
egen Q9_fe_atv = corr(Q9_fe Q9_fe_l1), covariance by(date)
label var Q9_fe_atv "Auto covariance of forecast errors"


** autocovariance of forecast

sort userid date
gen Q9_mean_l1 = l1.Q9_mean
label var Q9_mean_l1 "lagged forecasts"

sort date 
egen Q9_atv = corr(Q9_mean Q9_mean_l1), covariance by(date)
label var Q9_atv "Auto covariance of forecasts"


** variances of forecasts, i.e. disagreement
sort userid date 
egen Q9_sd = sd(Q9_mean), by(date)
gen Q9_disg = Q9_sd^2
label var Q9_disg "Disagreements of 1-yr-ahead expted inflation"

egen Q9_rd_sd = sd(Q9_mean_rd), by(date)
gen Q9_disg_rd = Q9_rd_sd^2
label var Q9_disg_rd "Disagreements of 1-yr-ahead expted inflation exl id FE"

egen Q9c_sd = sd(Q9c_mean), by(date)
gen Q9c_disg = Q9c_sd^2
label var Q9c_disg "Disagreement of 2-yr-ahead expted inflation"

foreach var in Q9 Q9c{
foreach mom in mean var{
     egen `var'_`mom'p75 =pctile(`var'_`mom'),p(75) by(year month)
	 egen `var'_`mom'p25 =pctile(`var'_`mom'),p(25) by(year month)
	 egen `var'_`mom'p50 =pctile(`var'_`mom'),p(50) by(year month)
	 local lb: variable label `var'_`mom'
	 label var `var'_`mom'p75 "`lb': 75 pctile"
	 label var `var'_`mom'p25 "`lb': 25 pctile"
	 label var `var'_`mom'p50 "`lb': 50 pctile"
}
}

save "${folder}/SCE/InfExpSCEProbIndM",replace 




***************************************
**   Histograms of Moments  ***********
** Maybe replaced by kernel desntiy **
***************************************

* for forecasting

gen SCE_mean = .
gen SCE_var = .

* for nowcasting
gen SCE_mean1 = . 
gen SCE_var1 = .   


* Kernal density plot only 

label var Q9_mean "1-yr-ahead forecast of inflation "
label var Q9c_mean "3-yr-ahead forecast of inflation"

 foreach var in SCE{
 foreach mom in mean{
    replace `var'_`mom' = Q9_`mom'
	local lb: variable label Q9_`mom'
    twoway (kdensity `var'_`mom' if `var'_`mom'!=.,lcolor(red) lwidth(thick) ), ///
	       by(year,title("Distribution of `lb'",size(med)) note("")) xtitle("Mean forecast") ///
		   ytitle("Fraction of population")
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist.png", as(png) replace 
}
}

* Kernal density plot only 


label var Q9_var "1-yr-ahead uncertainty of inflation"
label var Q9c_var "3-yr-ahead uncertainty of inflation"

foreach mom in var{
foreach var in SCE{
    replace `var'_`mom' = Q9_`mom'
	local lb: variable label Q9_`mom'
    twoway (kdensity `var'_`mom' if `var'_`mom'!=.,lcolor(blue) lwidth(thick)), ///
	       by(year,title("Distribution of `lb'",size(med)) note("")) xtitle("Uncertainty") ///
		   ytitle("Fraction of population")
	graph export "${sum_graph_folder}/hist/`var'`mom'_hist.png", as(png) replace 
}
}


** both 1-yr and 3-yr

 foreach var in SCE{
 foreach mom in mean{
    replace `var'_`mom' = Q9_`mom'
	replace `var'_`mom'1 = Q9c_`mom'
	local lb: variable label Q9_`mom'
    twoway (kdensity `var'_`mom' if `var'_`mom'!=.,lcolor(red) lwidth(thick) ) ///
	       (kdensity `var'_`mom'1  if `var'_`mom'1!=.,lcolor(black) lpattern(dash) lwidth(thick)), ///
	       by(year,title("Distribution of Mean Forecast",size(med)) note("")) xtitle("Mean forecast") ///
		   ytitle("Fraction of population") ///
		   legend(label(1 "1-year-ahead") label(2 "3-year-ahead") col(1))
	graph export "${sum_graph_folder}/hist/`var'`mom'01_hist.png", as(png) replace 
}
}

foreach mom in var{
foreach var in SCE{
    replace `var'_`mom' = Q9_`mom'
    replace `var'_`mom'1 = Q9c_`mom'
	local lb: variable label Q9_`mom'
	
    twoway (kdensity `var'_`mom' if `var'_`mom'!=.,lcolor(blue) lwidth(thick)) ///
	       (kdensity `var'_`mom'1 if `var'_`mom'1!=., lcolor(orange) lpattern(dash) lwidth(thick)), ///
	       by(year,title("Distribution of Uncertainty",size(med)) note("")) xtitle("Uncertainty") ///
		   ytitle("Fraction of population") ///
		   legend(label(1 "1-year-ahead") label(2 "3-year-ahead")  col(1))
	graph export "${sum_graph_folder}/hist/`var'`mom'01_hist.png", as(png) replace 
}
}


******************************
**  Generate Revision  ****
******************************

preserve
collapse (mean) SCE_mean SCE_mean1 SCE_var SCE_var1, by(date year month)
tsset date 


label var SCE_mean "expected inflation (SCE)"
label var SCE_var "inflation uncertainty (SCE)"

foreach mom in mean{
foreach var in SCE{
gen `var'_`mom'_rv = `var'_`mom' - l24.`var'_`mom'1
count if `var'_`mom'_rv !=.


local lb: variable label `var'_`mom' 
twoway (histogram `var'_`mom'_rv if `var'_`mom'_rv!=., bin(10) color(red) lcolor(red) lwidth(thick)), ///
 xline(0) ///
 by(year,title("Distribution of revision in `lb'") note("")) ytitle("Fraction of population") ///
 xtitle("Revision in mean forecast")
 graph export "${sum_graph_folder}/hist/`var'`mom'_rv_true_hist.png", as(png) replace 
}
}


foreach mom in var{
foreach var in SCE{
gen `var'_`mom'_rv = `var'_`mom' - l24.`var'_`mom'1
count if `var'_`mom'_rv !=.


local lb: variable label `var'_`mom' 
twoway (histogram `var'_`mom'_rv if `var'_`mom'_rv!=., bin(10) color(blue) lcolor(blue) lwidth(thick)), ///
 xline(0) ///
 by(year,title("Distribution of revision in `lb'") note("")) ytitle("Fraction of population") ///
 xtitle("Revision in uncertainty")
 graph export "${sum_graph_folder}/hist/`var'`mom'_rv_true_hist.png", as(png) replace 
}
}
restore 

*/

*************************
*** Population SCE ******
*************************


local Moments Q9_mean  Q9_var Q9_iqr Q9_cent50 Q9_disg ///
              Q9_mean_rd Q9_disg_rd Q9_var_rd Q9_fe_rd ///
              Q9c_mean Q9c_var Q9c_iqr Q9c_cent50 Q9c_disg ///
              Q9_atv Q9_fe_var  Q9_fe_rd_var Q9_fe_atv
			  
			  
local MomentsMom Q9_meanp25 Q9_meanp50 Q9_meanp75 Q9_varp25 Q9_varp50 Q9_varp75 ///
                 Q9c_meanp25 Q9c_meanp50 Q9c_meanp75 Q9c_varp25 Q9c_varp50 Q9c_varp75


collapse (mean) `Moments' `MomentsMom', by(date year month)

label var Q9_mean "Average 1-yr-ahead Expected Inflation(%)"
label var Q9_mean_rd "Average 1-year-ahead Expected Inflation(%) exl id FE"
label var Q9_var "Average Uncertainty of 1-yr-ahead Expected Inflation"
label var Q9_var_rd "Average 1-year-ahead Uncertainty exl id FE"
label var Q9_iqr "Average 25/75 IQR of 1-yr-ahead Expected Inflation(%)"
label var Q9_cent50 "Average Median of 1-yr-ahead Expected Inflation(%)"
label var Q9_disg "Disagreements of 1-yr-ahead Expected Inflation"
label var Q9_disg_rd "Disagreements of 1-yr-ahead Expected Inflation exl id FE"
label var Q9_fe_rd "Average Forecast Errors exl id FE"

label var Q9_atv "Autocovariances of 1-yr-ahead Forecasts"
label var Q9_fe_var "Variances of 1-yr-ahead Forecast Errors"
label var Q9_fe_atv "Autocovariances of 1-yr-ahead Forecast Errors"
label var Q9_fe_var "Variance of 1-yr-ahead Forecast Errors exl id FE"

label var Q9c_mean "Average 2-yr-ahead Expected Inflation(%)"
label var Q9c_var "Average Uncertainty of 2-yr-ahead Expected Inflation"
label var Q9c_iqr "Average 25/75 IQR of 2-yr-ahead Expected Inflation(%)"
label var Q9c_cent50 "Average Median of 2-yr-ahead Expected Inflation(%)"
label var Q9c_disg "Disagreements of 2-yr-ahead Expected Inflation"


save "${folder}/SCE/InfExpSCEProbPopM",replace 





