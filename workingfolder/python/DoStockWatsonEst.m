%%%%%%%%%%%%%%%%%%%%

clc;
clear;

% Quarterly 
%% Set up the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 2);

% Specify sheet and range
opts.Sheet = "Sheet1";
opts.DataRange = "A2:B114";

% Specify column names and types
opts.VariableNames = ["date", "Inf1y_CPICore"];
opts.VariableTypes = ["datetime", "double"];

% Specify variable properties
opts = setvaropts(opts, "date", "InputFormat", "");

% Import the data
CPICQ = readtable("../OtherData/CPICQ.xlsx", opts, "UseExcel", false);
CPICQ = rmmissing(CPICQ);

%% Clear temporary variables
clear opts

% get the series 

InfQ = CPICQ.Inf1y_CPICore;

%drop nan
InfQ = InfQ(~isnan(InfQ));

% do estimation 
var_esp_min = 0.01;
var_eta_min = 0.01;
[sd_eps,sd_eta,tau] = stockwatson(InfQ,0.01,0.01);

%% merge it back 
estQ = table(sd_eps,sd_eta,tau);
UCSVestQ = [CPICQ,estQ];

%% export to excel 

writetable(UCSVestQ,'../OtherData/UCSVestQ.xlsx');

% Monthly 
clear

%% Set up the Import Options and import the data
opts = spreadsheetImportOptions("NumVariables", 2);

% Specify sheet and range
opts.Sheet = "Sheet1";
opts.DataRange = "A2:B340";

% Specify column names and types
opts.VariableNames = ["date", "Inf1y_CPIAU"];
opts.VariableTypes = ["datetime", "double"];

% Specify variable properties
opts = setvaropts(opts, "date", "InputFormat", "");

% Import the data
CPIM = readtable("../OtherData/CPIM.xlsx", opts, "UseExcel", false);
CPIM = rmmissing(CPIM);

%% Clear temporary variables
clear opts

% get the series 

InfM = CPIM.Inf1y_CPIAU;

% dropna 
InfM = InfM(~isnan(InfM));

% do estimation 
var_esp_min = 0.01;
var_eta_min = 0.01;
[sd_eps,sd_eta,tau] = stockwatson(InfM,0.01,0.01);

%% merge it back 
estM = table(sd_eps,sd_eta,tau);
UCSVestM = [CPIM,estM];

%% export to excel 

writetable(UCSVestM,'../OtherData/UCSVestM.xlsx');

