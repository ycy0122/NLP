clc;clear;close all
ReportTimeAll = {};
reportsencellarrayAll = {};

%% combine all clinical notes
load('NLP_700_clinical_SC_updated.mat')
ReportTimeAll = [ReportTimeAll;ReportTime];
reportsencellarrayAll = [reportsencellarrayAll; reportsencellarray];
clear ReportTime reportsencellarray

load('NLP_700_clinical_SC2_updated.mat')
ReportTimeAll = [ReportTimeAll,ReportTime(8501:17000)];
reportsencellarrayAll(8501:17000) =  reportsencellarray(8501:17000);
clear ReportTime reportsencellarray

load('NLP_700_clinical_SC_YH.mat')
ReportTimeAll = [ReportTimeAll,ReportTime];
reportsencellarrayAll(17001:25500) =  reportsencellarray(17001:25500);
clear ReportTime reportsencellarray


