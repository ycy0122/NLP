clc;clear;close all
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\AlgorithmUtilities'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\AlgorithmUtilities\custom_func\Random-Forest-Matlab-master\lib'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\MatlabNLP-master\funcs\funcs'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NLP_MGUS'))

save_name = 'Final_110122_MLNoRule';
%% load texts as input for bag-of-words(BOW) algorithms
nminFeatures = 10000;  % minimum # of appearances for this term to be a feature
removeStopWords = 1; 
doStem = 0;

tic
load('NLP_700_clinical_processed.mat','clinicaldata','reportcell','N_report','ReportTime') 
[featureVector,headers] = featurizeTrainReports(reportcell, nminFeatures, removeStopWords, doStem);
save(['NLP_700_clinical_featureVector',num2str(nminFeatures),'.mat'],'featureVector','headers')
BOWtime = toc

%% 
load(['NLP_700_clinical_featureVector',num2str(nminFeatures),'.mat'],'featureVector','headers')
pid = unique(table2array(clinicaldata(:,1)));
ReportTime = strtrim(ReportTime);

new_id_list = [];
% pid = 283160217
for i = 1:length(pid)
    this_pid = pid(i);
    pid_loc = find(table2array(clinicaldata(:,1)) == this_pid);
    time_emp_id = cellfun(@isempty,ReportTime(1,pid_loc).','UniformOutput',false);
    [~, idx_date] = sort(cell2mat(cellfun(@datenum,ReportTime(1,pid_loc).','UniformOutput',false)), 1, 'ascend');
    
    if sum(cellfun(@sum, time_emp_id)) ~= 0 % empty cells... delete
        idx = find(cellfun(@sum, time_emp_id)==1);
        idx_date_fix = setdiff(1:length(pid_loc),idx).';
        idx_date = idx_date_fix;
    end

    new_id_list = [new_id_list;pid_loc(idx_date)];
end
clinicaldata = clinicaldata(new_id_list,:);
ReportTime = ReportTime(1,new_id_list);        
N_report = size(clinicaldata,1);

% make the feature matrix in terms of patients...
TEST_filename = "True_MGUSPROG.xlsx";
TESTtable = readtable(TEST_filename);
true_train_label = table2array(TESTtable(:,3));

train_true_filenamemm = "True_MGUSPROG.xlsx";
true_train_labelmm = readtable(train_true_filenamemm);
true_train_labelmm = table2array(true_train_labelmm(:,4));

patients_lables =  true_train_label+1;
patients_lablesmm =  true_train_labelmm+1;
PatientVector = zeros(length(true_train_label),size(featureVector,2));

for i = 1:length(pid)
    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    PatientVector(i,:) = sum(featureVector(this_pid(1):this_pid(end),:));
end

% transform bow to bow_to_tfidf
tfidf_vector = full(bow_to_tfidf(PatientVector));
%% specificity + sensitivity
load('Results_clinical_700_newplasma080922.mat')

sensitivity = sum(MGUS_label == 1 & true_train_label == 1)./ ( sum(MGUS_label == 1 & true_train_label == 1) +  sum(MGUS_label == 0 & true_train_label == 1))
specificity = sum(MGUS_label == 0 & true_train_label == 0)./ ( sum(MGUS_label == 0 & true_train_label == 0) +  sum(MGUS_label == 1 & true_train_label == 0))
PPV =  sum(MGUS_label == 1 & true_train_label == 1)./ ( sum(MGUS_label == 1 & true_train_label == 1) +  sum(MGUS_label == 1 & true_train_label == 0))
NPV =  sum(MGUS_label == 0 & true_train_label == 0)./ ( sum(MGUS_label == 0 & true_train_label == 0) +  sum(MGUS_label == 0 & true_train_label == 1))
Accuracy = sum(MGUS_label==true_train_label)/numel(true_train_label)
F1_score = 2/(1/sensitivity + 1/PPV)

%% MM performance
sensitivitymm = sum(MM_label == 1 & true_train_labelmm == 1)./ ( sum(MM_label == 1 & true_train_labelmm == 1) +  sum(MM_label == 0 & true_train_labelmm == 1))
specificitymm = sum(MM_label == 0 & true_train_labelmm == 0)./ ( sum(MM_label == 0 & true_train_labelmm == 0) +  sum(MM_label == 1 & true_train_labelmm == 0))
PPVmm =  sum(MM_label == 1 & true_train_labelmm == 1)./ ( sum(MM_label == 1 & true_train_labelmm == 1) +  sum(MM_label == 1 & true_train_labelmm == 0))
NPVmm =  sum(MM_label == 0 & true_train_labelmm == 0)./ ( sum(MM_label == 0 & true_train_labelmm == 0) +  sum(MM_label == 0 & true_train_labelmm == 1))
Accuracymm = sum(MM_label==true_train_labelmm)/numel(true_train_labelmm)
F1_scoremm = 2/(1/sensitivitymm + 1/PPVmm)

%% Form new ML dataset for training and testing
sumMGUS_vector = []; 
sumProtein_vector = []; 
sumNotMM_vector = []; 
sumMM_vector = []; 
sumSMM_vector = [];
sumTreat_vector = [];

for i = 1:length(pid)
    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    sumMGUS_vector(i,1) = sum(ind_MGUS(this_pid(1):this_pid(end)));
    sumMM_vector(i,1) = sum(ind_MM(this_pid(1):this_pid(end)));
    sumNotMM_vector(i,1) = sum(ind_NotMM(this_pid(1):this_pid(end)));
    sumProtein_vector(i,1) = sum(ind_protein(this_pid(1):this_pid(end)));
    sumSMM_vector(i,1) = sum(ind_SMM(this_pid(1):this_pid(end)));
    sumTreat_vector(i,1) = sum(ind_Treat(this_pid(1):this_pid(end)));
end

train_Matrix = full(bow_to_tfidf([PatientVector]));

% train_Matrix = full(bow_to_tfidf([PatientVector, sumMGUS_vector, ...
%     sumNotMM_vector, sumProtein_vector, sumSMM_vector, sumTreat_vector]));
% 

%% normalize(?)
MatrixN = bsxfun(@rdivide,bsxfun(@minus,train_Matrix,min(train_Matrix,[],2)), max(train_Matrix,[],2));

%% SVM
opts= struct;
opts.C= 1e-1;
opts.polyOrder= 2;
opts.rbfScale= 1;
opts.type = 3;

tic;
modelSVM_MGUS = svmTrain(train_Matrix, true_train_label+1, opts); % train
timetrain= toc;
yhatMGUS_svm = svmTest(modelSVM_MGUS, train_Matrix) - 1;
[sen_mgus_svm,spe_mgus_svm,PPV_mgus_svm,NPV_mgus_svm,Accu_mgus_svm,F1_score_mgus_svm] =...
    CalcPerformance(yhatMGUS_svm,true_train_label);

tic;
modelSVM_MM = svmTrain(train_Matrix, true_train_labelmm+1, opts); % train
timetrain= toc;
yhatMM_svm = svmTest(modelSVM_MM, train_Matrix) - 1;
[sen_mm_svm,spe_mm_svm,PPV_mm_svm,NPV_mm_svm,Accu_mm_svm,F1_score_mm_svm] =...
    CalcPerformance(yhatMM_svm,true_train_labelmm);

%% random forest
optsRF= struct;
optsRF.depth= 15;
optsRF.numTrees = 200;
optsRF.numSplits= 20;
optsRF.verbose= true;
optsRF.classifierID = [1,2]; % weak learners to use. Can be an array for mix of weak learners too

train_MatrixN= bsxfun(@rdivide, bsxfun(@minus, train_Matrix, mean(train_Matrix)), var(train_Matrix) + 1e-10);
tic
modelRF_MGUS= forestTrain(train_MatrixN, true_train_label+1, optsRF); % train
timetrain= toc;
yhatMGUS_rf = forestTest(modelRF_MGUS, train_MatrixN) - 1;
[sen_mgus_rf,spe_mgus_rf,PPV_mgus_rf,NPV_mgus_rf,Accu_mgus_rf,F1_score_mgus_rf] =...
    CalcPerformance(yhatMGUS_rf,true_train_label);
tic
modelRF_MM= forestTrain(train_MatrixN, true_train_labelmm+1, optsRF); % train
timetrain= toc;
yhatMM_rf = forestTest(modelRF_MM, train_MatrixN) - 1;
[sen_mm_rf,spe_mm_rf,PPV_mm_rf,NPV_mm_rf,Accu_mm_rf,F1_score_mm_rf] =...
    CalcPerformance(yhatMM_rf,true_train_labelmm);
% Logistic regression
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\logistic'))
%compute cost and gradient
iter = 50000; % No. of iterations for weight updation
theta=zeros(size(train_Matrix,2),1); % Initial weights

al=0.2; % Learning parameter
tic
[J grad h thMGUS]=cost(theta,train_MatrixN,true_train_label,al,iter); % Cost funtion
toc
yhatMGUS_lg=train_MatrixN*thMGUS; %target prediction

% probability calculation
[hp]=sigmoid(yhatMGUS_lg); % Hypothesis Function
yhatMGUS_lg(hp>=0.5)=1;
yhatMGUS_lg(hp<0.5)=0;

[sens_mgus_lr,spe_mgus_lr,PPV_mgus_lr,NPV_mgus_lr,Accuracy_mgus_lr,F1_score_mgus_lr] =...
    CalcPerformance(yhatMGUS_lg,true_train_label);

tic
[J grad h thMM]=cost(theta,train_MatrixN,true_train_labelmm,al,iter); % Cost funtion
toc
yhatMM_lg=train_MatrixN*thMM; %target prediction

% probability calculation
[hp]=sigmoid(yhatMM_lg); % Hypothesis Function
yhatMM_lg(hp>=0.5)=1;
yhatMM_lg(hp<0.5)=0;

[sens_mm_lr,spe_mm_lr,PPV_mm_lr,NPV_mm_lr,Accuracy_mm_lr,F1_score_mm_lr] =...
    CalcPerformance(yhatMM_lg,true_train_labelmm);

%% save results
% train
T = readtable('NLP_performance_092922.xlsx','Sheet','Details','Range','J1:J701');
train_cohort = strcmp(table2array(T),'TR');
test_cohort = strcmp(table2array(T),'TE');
valid_cohort = strcmp(table2array(T),'VA');

%% SVM
Est_SVM_TR = zeros(6,1);
[Est_SVM_TR(1),Est_SVM_TR(2),Est_SVM_TR(3),Est_SVM_TR(4),Est_SVM_TR(5),Est_SVM_TR(6)] ...
    = CalcPerformance(true_train_label(train_cohort),yhatMGUS_svm(train_cohort));

Est_SVM_TE = zeros(6,1);
[Est_SVM_TE(1),Est_SVM_TE(2),Est_SVM_TE(3),Est_SVM_TE(4),Est_SVM_TE(5),Est_SVM_TE(6)] ...
    = CalcPerformance(true_train_label(test_cohort),yhatMGUS_svm(test_cohort));

Est_SVM_VA = zeros(6,1);
[Est_SVM_VA(1),Est_SVM_VA(2),Est_SVM_VA(3),Est_SVM_VA(4),Est_SVM_VA(5),Est_SVM_VA(6)] ...
    = CalcPerformance(true_train_label(valid_cohort),yhatMGUS_svm(valid_cohort));

sensitivity = sum(yhatMGUS_svm == 1 & true_train_label == 1)./ ( sum(yhatMGUS_svm == 1 & true_train_label == 1) +  sum(yhatMGUS_svm == 0 & true_train_label == 1))
specificity = sum(yhatMGUS_svm == 0 & true_train_label == 0)./ ( sum(yhatMGUS_svm == 0 & true_train_label == 0) +  sum(yhatMGUS_svm == 1 & true_train_label == 0))
PPV =  sum(yhatMGUS_svm == 1 & true_train_label == 1)./ ( sum(yhatMGUS_svm == 1 & true_train_label == 1) +  sum(yhatMGUS_svm == 1 & true_train_label == 0))
NPV =  sum(yhatMGUS_svm == 0 & true_train_label == 0)./ ( sum(yhatMGUS_svm == 0 & true_train_label == 0) +  sum(yhatMGUS_svm == 0 & true_train_label == 1))
Accuracy = sum(yhatMGUS_svm==true_train_label)/numel(true_train_label)
F1_score = 2/(1/sensitivity + 1/PPV)

Est_SVM_TR = zeros(6,1);
[Est_SVM_TR(1),Est_SVM_TR(2),Est_SVM_TR(3),Est_SVM_TR(4),Est_SVM_TR(5),Est_SVM_TR(6)] ...
    = CalcPerformance(true_train_labelmm(train_cohort),yhatMM_svm(train_cohort));

Est_SVM_TE = zeros(6,1);
[Est_SVM_TE(1),Est_SVM_TE(2),Est_SVM_TE(3),Est_SVM_TE(4),Est_SVM_TE(5),Est_SVM_TE(6)] ...
    = CalcPerformance(true_train_labelmm(test_cohort),yhatMM_svm(test_cohort));

Est_SVM_VA = zeros(6,1);
[Est_SVM_VA(1),Est_SVM_VA(2),Est_SVM_VA(3),Est_SVM_VA(4),Est_SVM_VA(5),Est_SVM_VA(6)] ...
    = CalcPerformance(true_train_labelmm(valid_cohort),yhatMM_svm(valid_cohort));

sensitivity = sum(yhatMM_svm == 1 & true_train_labelmm == 1)./ ( sum(yhatMM_svm == 1 & true_train_labelmm == 1) +  sum(yhatMM_svm == 0 & true_train_labelmm == 1))
specificity = sum(yhatMM_svm == 0 & true_train_labelmm == 0)./ ( sum(yhatMM_svm == 0 & true_train_labelmm == 0) +  sum(yhatMM_svm == 1 & true_train_labelmm == 0))
PPV =  sum(yhatMM_svm == 1 & true_train_labelmm == 1)./ ( sum(yhatMM_svm == 1 & true_train_labelmm == 1) +  sum(yhatMM_svm == 1 & true_train_labelmm == 0))
NPV =  sum(yhatMM_svm == 0 & true_train_labelmm == 0)./ ( sum(yhatMM_svm == 0 & true_train_labelmm == 0) +  sum(yhatMM_svm == 0 & true_train_labelmm == 1))
Accuracy = sum(yhatMM_svm==true_train_labelmm)/numel(true_train_labelmm)
F1_score = 2/(1/sensitivity + 1/PPV)

%% RF
Est_RF_TR = zeros(6,1);
[Est_RF_TR(1),Est_RF_TR(2),Est_RF_TR(3),Est_RF_TR(4),Est_RF_TR(5),Est_RF_TR(6)] ...
    = CalcPerformance(true_train_label(train_cohort),yhatMGUS_rf(train_cohort));

Est_RF_TE = zeros(6,1);
[Est_RF_TE(1),Est_RF_TE(2),Est_RF_TE(3),Est_RF_TE(4),Est_RF_TE(5),Est_RF_TE(6)] ...
    = CalcPerformance(true_train_label(test_cohort),yhatMGUS_rf(test_cohort));

Est_RF_VA = zeros(6,1);
[Est_RF_VA(1),Est_RF_VA(2),Est_RF_VA(3),Est_RF_VA(4),Est_RF_VA(5),Est_RF_VA(6)] ...
    = CalcPerformance(true_train_label(valid_cohort),yhatMGUS_rf(valid_cohort));

sensitivity = sum(yhatMGUS_rf == 1 & true_train_label == 1)./ ( sum(yhatMGUS_rf == 1 & true_train_label == 1) +  sum(yhatMGUS_rf == 0 & true_train_label == 1))
specificity = sum(yhatMGUS_rf == 0 & true_train_label == 0)./ ( sum(yhatMGUS_rf == 0 & true_train_label == 0) +  sum(yhatMGUS_rf == 1 & true_train_label == 0))
PPV =  sum(yhatMGUS_rf == 1 & true_train_label == 1)./ ( sum(yhatMGUS_rf == 1 & true_train_label == 1) +  sum(yhatMGUS_rf == 1 & true_train_label == 0))
NPV =  sum(yhatMGUS_rf == 0 & true_train_label == 0)./ ( sum(yhatMGUS_rf == 0 & true_train_label == 0) +  sum(yhatMGUS_rf == 0 & true_train_label == 1))
Accuracy = sum(yhatMGUS_rf==true_train_label)/numel(true_train_label)
F1_score = 2/(1/sensitivity + 1/PPV)

Est_RF_TR = zeros(6,1);
[Est_RF_TR(1),Est_RF_TR(2),Est_RF_TR(3),Est_RF_TR(4),Est_RF_TR(5),Est_RF_TR(6)] ...
    = CalcPerformance(true_train_labelmm(train_cohort),yhatMM_rf(train_cohort));

Est_RF_TE = zeros(6,1);
[Est_RF_TE(1),Est_RF_TE(2),Est_RF_TE(3),Est_RF_TE(4),Est_RF_TE(5),Est_RF_TE(6)] ...
    = CalcPerformance(true_train_labelmm(test_cohort),yhatMM_rf(test_cohort));

Est_RF_VA = zeros(6,1);
[Est_RF_VA(1),Est_RF_VA(2),Est_RF_VA(3),Est_RF_VA(4),Est_RF_VA(5),Est_RF_VA(6)] ...
    = CalcPerformance(true_train_labelmm(valid_cohort),yhatMM_rf(valid_cohort));

sensitivity = sum(yhatMM_rf == 1 & true_train_labelmm == 1)./ ( sum(yhatMM_rf == 1 & true_train_labelmm == 1) +  sum(yhatMM_rf == 0 & true_train_labelmm == 1))
specificity = sum(yhatMM_rf == 0 & true_train_labelmm == 0)./ ( sum(yhatMM_rf == 0 & true_train_labelmm == 0) +  sum(yhatMM_rf == 1 & true_train_labelmm == 0))
PPV =  sum(yhatMM_rf == 1 & true_train_labelmm == 1)./ ( sum(yhatMM_rf == 1 & true_train_labelmm == 1) +  sum(yhatMM_rf == 1 & true_train_labelmm == 0))
NPV =  sum(yhatMM_rf == 0 & true_train_labelmm == 0)./ ( sum(yhatMM_rf == 0 & true_train_labelmm == 0) +  sum(yhatMM_rf == 0 & true_train_labelmm == 1))
Accuracy = sum(yhatMM_rf==true_train_labelmm)/numel(true_train_labelmm)
F1_score = 2/(1/sensitivity + 1/PPV)

%% LR
Est_LR_TR = zeros(6,1);
[Est_LR_TR(1),Est_LR_TR(2),Est_LR_TR(3),Est_LR_TR(4),Est_LR_TR(5),Est_LR_TR(6)] ...
    = CalcPerformance(true_train_label(train_cohort),yhatMGUS_lg(train_cohort));

Est_LR_TE = zeros(6,1);
[Est_LR_TE(1),Est_LR_TE(2),Est_LR_TE(3),Est_LR_TE(4),Est_LR_TE(5),Est_LR_TE(6)] ...
    = CalcPerformance(true_train_label(test_cohort),yhatMGUS_lg(test_cohort));

Est_LR_VA = zeros(6,1);
[Est_LR_VA(1),Est_LR_VA(2),Est_LR_VA(3),Est_LR_VA(4),Est_LR_VA(5),Est_LR_VA(6)] ...
    = CalcPerformance(true_train_label(valid_cohort),yhatMGUS_lg(valid_cohort));

sensitivity = sum(yhatMGUS_lg == 1 & true_train_label == 1)./ ( sum(yhatMGUS_lg == 1 & true_train_label == 1) +  sum(yhatMGUS_lg == 0 & true_train_label == 1))
specificity = sum(yhatMGUS_lg == 0 & true_train_label == 0)./ ( sum(yhatMGUS_lg == 0 & true_train_label == 0) +  sum(yhatMGUS_lg == 1 & true_train_label == 0))
PPV =  sum(yhatMGUS_lg == 1 & true_train_label == 1)./ ( sum(yhatMGUS_lg == 1 & true_train_label == 1) +  sum(yhatMGUS_lg == 1 & true_train_label == 0))
NPV =  sum(yhatMGUS_lg == 0 & true_train_label == 0)./ ( sum(yhatMGUS_lg == 0 & true_train_label == 0) +  sum(yhatMGUS_lg == 0 & true_train_label == 1))
Accuracy = sum(yhatMGUS_lg==true_train_label)/numel(true_train_label)
F1_score = 2/(1/sensitivity + 1/PPV)

Est_LR_TR = zeros(6,1);
[Est_LR_TR(1),Est_LR_TR(2),Est_LR_TR(3),Est_LR_TR(4),Est_LR_TR(5),Est_LR_TR(6)] ...
    = CalcPerformance(true_train_labelmm(train_cohort),yhatMM_lg(train_cohort));

Est_LR_TE = zeros(6,1);
[Est_LR_TE(1),Est_LR_TE(2),Est_LR_TE(3),Est_LR_TE(4),Est_LR_TE(5),Est_LR_TE(6)] ...
    = CalcPerformance(true_train_labelmm(test_cohort),yhatMM_lg(test_cohort));

Est_LR_VA = zeros(6,1);
[Est_LR_VA(1),Est_LR_VA(2),Est_LR_VA(3),Est_LR_VA(4),Est_LR_VA(5),Est_LR_VA(6)] ...
    = CalcPerformance(true_train_labelmm(valid_cohort),yhatMM_lg(valid_cohort));

sensitivity = sum(yhatMM_lg == 1 & true_train_labelmm == 1)./ ( sum(yhatMM_lg == 1 & true_train_labelmm == 1) +  sum(yhatMM_lg == 0 & true_train_labelmm == 1))
specificity = sum(yhatMM_lg == 0 & true_train_labelmm == 0)./ ( sum(yhatMM_lg == 0 & true_train_labelmm == 0) +  sum(yhatMM_lg == 1 & true_train_labelmm == 0))
PPV =  sum(yhatMM_lg == 1 & true_train_labelmm == 1)./ ( sum(yhatMM_lg == 1 & true_train_labelmm == 1) +  sum(yhatMM_lg == 1 & true_train_labelmm == 0))
NPV =  sum(yhatMM_lg == 0 & true_train_labelmm == 0)./ ( sum(yhatMM_lg == 0 & true_train_labelmm == 0) +  sum(yhatMM_lg == 0 & true_train_labelmm == 1))
Accuracy = sum(yhatMM_lg==true_train_labelmm)/numel(true_train_labelmm)
F1_score = 2/(1/sensitivity + 1/PPV)

%%

save(['ML_Only_No_Rule',num2str(nminFeatures),'.mat'],...
    'sen_mgus_svm','spe_mgus_svm','PPV_mgus_svm','NPV_mgus_svm','Accu_mgus_svm','F1_score_mgus_svm',...
    'sen_mm_svm','spe_mm_svm','PPV_mm_svm','NPV_mm_svm','Accu_mm_svm','F1_score_mm_svm',...
    'sen_mgus_rf','spe_mgus_rf','PPV_mgus_rf','NPV_mgus_rf','Accu_mgus_rf','F1_score_mgus_rf',...
    'sen_mm_rf','spe_mm_rf','PPV_mm_rf','NPV_mm_rf','Accu_mm_rf','F1_score_mm_rf',...
    'sens_mgus_lr','spe_mgus_lr','PPV_mgus_lr','NPV_mgus_lr','Accuracy_mgus_lr','F1_score_mgus_lr',...
    'sens_mm_lr','spe_mm_lr','PPV_mm_lr','NPV_mm_lr','Accuracy_mm_lr','F1_score_mm_lr')