*******************************************************************************
***  This do file starts with  a clean quarterly version of                  **
***  InfShocksClean to be used for individual IR analysis.                   **
***   Then it merges with population survey data and plot all kinds of impulse *
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
log using "${mainfolder}/SCEirf_log",replace



*****************************
***      IR Analysis     ****
*****************************

use "${mainfolder}/OtherData/InfShocksMClean.dta",clear  

merge 1:1 year month using "${folder}/InfExpM.dta",keep(match using master)
rename _merge InfExp_merge

drop if month==. 


** Period filter   
** i.e. Coibion et al2012. 1976-2007. But Density data is only avaiable after 2007.

*keep if year > 2007
*keep if year>=1976 & year <= 2007
tsset date

 
/*
** Plot all shocks for checking 

twoway (tsline op_shock) ///
        (tsline mp1ut_shock) (tsline ED8ut_shock) ///
		(tsline CPIAU_uid_shock) if mp1ut_shock!=., ///
		title("Shocks to Inflation",size(4)) ///
		xtitle("Time") ytitle("") ///
		legend(cols(1)) 
		
graph export "${sum_graph_folder}/inf_shocksM", as(png) replace


** First-run of inflation 

eststo clear
foreach sk in op mp1ut ED4ut ED8ut{
  foreach Inf in CPIAU CPICore PCEPI{ 
   eststo `Inf'_`sk': reg Inf1y_`Inf' l(0/1).`sk'_shock, robust
   eststo `Inf'_uid: reg Inf1y_`Inf' l(0/1).`Inf'_uid_shock,robust 
 }
}
esttab using "${sum_table_folder}/IRFM.csv", mtitles se r2 replace


** IRF of inflation (one shock each time) 

eststo clear
foreach sk in op mp1ut ED4ut ED8ut{
  foreach Inf in CPIAU CPICore PCEPI{ 
   var Inf1y_`Inf', lags(1/4) exo(l(0/1).`sk'_shock)
   set seed 123456
   irf create irf1, set(irf,replace) step(10) bsp
   irf graph dm, impulse(`sk'_shock)
   graph export "${sum_graph_folder}/irf/`Inf'_`sk'M", as(png) replace
 }
}



***********************************************
** IRF of inflation (MP shocks at one time) **
***********************************************

eststo clear

foreach Inf in CPIAU PCEPI{ 
   var Inf1y_`Inf', lags(1/4) ///
                     exo(l(0/1).op_shock ///
					 l(0/1).mp1ut_shock l(0/1).ED8ut_shock)   
   set seed 123456
   irf create irf1, set(irf,replace) step(10) bsp replace 
   irf graph dm, impulse(mp1ut_shock ED8ut_shock) ///
                 byopts(title("`mom'") yrescale xrescale note("")) ///
                 legend(col(2) order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall)) ///
				 xtitle("Month") 
   graph export "${sum_graph_folder}/irf/`Inf'_ashocksM", as(png) replace

}



***********************************************
** IRF of inflation (all shocks exl MP at one time) **
***********************************************


eststo clear

foreach Inf in CPIAU CPICore PCEPI{ 
   var Inf1y_`Inf', lags(1/4) ///
                     exo(l(0/1).op_shock)   
   set seed 123456
   irf create irf1, set(irf,replace) step(10) bsp replace 
   irf graph dm, impulse(op_shock) ///
                 byopts(title("`mom'") yrescale xrescale note("")) ///
                 legend(col(2) order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall)) ///
				 xtitle("Month") 
   graph export "${sum_graph_folder}/irf/`Inf'_ashocks_nmpM", as(png) replace

}



****************************************************
** IRF of SCE moments (MP shocks at one time)    **
****************************************************


foreach mom in FE{
   foreach var in SCE{
       * shocks 
       capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).op_shock ///
					 l(0/1).mp1ut_shock l(0/1).ED8ut_shock)
   set seed 123456
   capture irf create `var', set(`mom',replace) step(10) bsp replace 
}
 
   capture irf graph dm, impulse(mp1ut_shock ED8ut_shock) ///
                         byopts(col(2) title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Month") 
   capture graph export "${sum_graph_folder}/irf/moments/SCE`mom'_ashocksM", as(png) replace
}


*********************************************************
** IRF of SPF moments (MP shocks(abs) at one time)    **
*********************************************************


foreach mom in Disg Var{
   foreach var in SCE{
       * shocks 
       capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).op_shock l(0/1).mp1ut_shock l(0/1).ED8ut_shock ///
					 l(0/1).op_abshock l(0/1).mp1ut_abshock l(0/1).ED8ut_abshock)
   set seed 123456
   capture irf create `var', set(`mom',replace) step(10) bsp replace 
}
 
   capture irf graph dm, impulse(mp1ut_abshock ED8ut_abshock) ///
                         byopts(col(2) title("`mom'") yrescale /// 
						 xrescale note("") ) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Month") xtick(0(1)10)
   capture graph export "${sum_graph_folder}/irf/moments/SCE`mom'_ab_ashocksM", as(png) replace
}
*/

***********************************************************
** IRF of SPF moments (all shocks exl MP at one time)    **
***********************************************************


foreach mom in FE{
   foreach var in SCE{
	   capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).op_shock) 
   set seed 123456
   capture irf create `var'_nmp, set(`mom'_nmp,replace) step(10) bsp replace  
}
   capture irf graph dm, impulse(op_shock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Month") 
   capture graph export "${sum_graph_folder}/irf/moments/SCE`mom'_ashocks_nmpM", as(png) replace
}


****************************************************************
** IRF of SPF moments (all shocks(abs) exl MP at one time)    **
****************************************************************


foreach mom in Disg Var{
   foreach var in SCE{
       * shocks 
       capture var `var'_`mom', lags(1/4) ///
                     exo(l(0/1).op_abshock)
   set seed 123456
   capture irf create `var'_nmp, set(`mom'_nmp,replace) step(10) bsp replace 
}
 
   capture irf graph dm, impulse(op_abshock) ///
                         byopts(title("`mom'") yrescale /// 
						 xrescale note("")) legend(col(2) /// 
						 order(1 "95% CI" 2 "IRF") symx(*.5) size(vsmall))  ///
						 xtitle("Month") 
   capture graph export "${sum_graph_folder}/irf/moments/SCE`mom'_ab_ashocks_nmpM", as(png) replace
}

*/


log close 
