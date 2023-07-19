clc;clear;close all
%% read html file into matlab (SGLT2 folder)
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei'))


url = ['P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei\NLP_700_clinical.html'];
tic
    Outputtable = fread_html_reports(url);
toc
writetable(Outputtable,['NLP_700_clinical.xlsx'])


