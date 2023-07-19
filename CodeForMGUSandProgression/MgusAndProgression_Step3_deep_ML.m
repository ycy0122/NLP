clc;clear;close all
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\cohort1\Finished'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\AlgorithmUtilities'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\AlgorithmUtilities\custom_func\Random-Forest-Matlab-master\lib'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\MatlabNLP-master\funcs\funcs'))
% addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NLP_MGUS'))

save_name = 'Final_02032023_deepML';

%% universal data & variables
mspike_filename = "NLP_M_Protein.xlsx";
mspike = readtable(mspike_filename);
MK_PatientSSN = str2double(mspike.PatientSSN);
M_Spike = mspike.mspike;
mspike_date = mspike.LabChemCompleteDate;
ind_mspike_mgus = unique(MK_PatientSSN(M_Spike<3 & M_Spike >= 0.1 ));
ind_mspike_mm = unique(MK_PatientSSN(M_Spike>=3));
ind1mspike = M_Spike<3 & M_Spike >= 0.1 ;
ind2mspike = M_Spike>=3;
mspike_date = datetime(mspike_date);
%% KL ratio for train
klratio_filename = "NLP_KL_ratio.xlsx";
klratio = readtable(klratio_filename);
KL_PatientSSN = str2double(klratio.PatientSSN);
KL_ratio = klratio.klratio;
KL_date = klratio.LabChemCompleteDate;
ind_klratio = unique(KL_PatientSSN(KL_ratio>=100));
ind2klratio = KL_ratio>=100;
KL_date = datetime(KL_date);
%% plasma cell
plasmacell_filename = "NLP_Plasma_cell.xlsx";
plasmacell = readtable(plasmacell_filename);
PCPatientSSN = str2double(plasmacell.PatientSSN);
PC = plasmacell.plasma_cell;
% PC = cellfun(@str2num,(plasmacell.x_OfPlasma));
PC_date = plasmacell.SpecimenTakenDate;
ind_PC = unique(PCPatientSSN);
%%
MK_unique = unique(MK_PatientSSN);
KL_unique = unique(KL_PatientSSN);
PC_unique = unique(PCPatientSSN);


%% load texts as input for bag-of-words(BOW) algorithms
nminFeatures = 5000;  % minimum # of appearances for this term to be a feature
removeStopWords = 1; 
doStem = 0;
% 
% tic
% load('NLP_700_clinical_processed.mat','clinicaldata','reportcell','N_report','ReportTime') 
% [featureVector,headers] = featurizeTrainReports(reportcell, nminFeatures, removeStopWords, doStem);
% save(['NLP_700_clinical_featureVector',num2str(nminFeatures),'.mat'],'featureVector','headers')
% BOWtime = toc

%% 
load('NLP_700_clinical_processed.mat','clinicaldata','reportcell','N_report','ReportTime') 
load(['NLP_700_clinical_featureVector',num2str(nminFeatures),'_updated.mat'],'featureVector','headers')
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

addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei\True'))
% make the feature matrix in terms of patients...
True_filename = "True_MGUSPROG.xlsx";
Truetable = readtable(True_filename);
true_train_label = table2array(Truetable(:,3));
true_train_date = table2array(Truetable(:,2));

true_train_labelmm =  table2array(Truetable(:,4));
true_train_labelmm_date =  table2array(Truetable(:,5));


%% treatment file 
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\Code\HSICLasso'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\Code\OtherMLutilities'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NLP_MGUS'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\AlgorithmUtilities'))

Treat_records = readtable('myeloma_drug_use_history.xlsx');

patients_lables =  true_train_label+1;
patients_lablesmm =  true_train_labelmm+1;
PatientVector = zeros(length(true_train_label),size(featureVector,2));

for i = 1:length(pid)
    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    PatientVector(i,:) = sum(featureVector(this_pid(1):this_pid(end),:));
end

% transform bow to bow_to_tfidf
tfidf_vector = full(bow_to_tfidf(PatientVector));
%% specificity + sensitivity == Rule only
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

 % add mspike, PC, and Treatment vector
 KL_vecotr = zeros(length(true_train_label),1);
 MS1_vecotr = zeros(length(true_train_label),1);
 MS2_vecotr = zeros(length(true_train_label),1);
 PC_vecotr1 = zeros(length(true_train_label),1);
 PC_vecotr2 = zeros(length(true_train_label),1);
 Treat_vecotr = zeros(length(true_train_label),1);
for i = 1:length(pid)
    this_Drug_T = pid(i) == table2array(Treat_records(:,1));
    
    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    KL_vecotr(i) = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_klratio));
    MS1_vecotr(i) = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mgus));
    MS2_vecotr(i) = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mm));
    
    this_PC = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_PC));
    PC_vecotr1(i) = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_PC));
    if this_PC ~= 0
        pc_id = find(pid(i) == PCPatientSSN);
        PC_val = PC(pc_id);
        if length(PC_val) >  1
            PC_val;
        end
        PC_vecotr2(i) = any((PC_val >= 10));
    end
%     this_Treat = any(ind_Treat(this_pid) == 1);          %  C4
    if sum(this_Drug_T) ~= 0 
        Treat_vecotr(i) = 1;
    end
end
 


%% train, test, validation id & pid
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\Code\HSICLasso'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\Code\OtherMLutilities'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NLP_MGUS'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\AlgorithmUtilities'))
% load(['NLP_700_clinical_processed_test.mat'],'clinicaldata')
[pid, ic, ia] = unique(table2array(clinicaldata(:,1)));

T = readtable('NLP_performance_092922.xlsx','Sheet','Details','Range','J1:J701');
train_cohort = strcmp(table2array(T),'TR');
test_cohort = strcmp(table2array(T),'TE');
valid_cohort = strcmp(table2array(T),'VA');

%%
train_Matrix = [PatientVector(train_cohort,:), sumMGUS_vector(train_cohort,:), ...
    sumNotMM_vector(train_cohort,:), sumProtein_vector(train_cohort,:), sumSMM_vector(train_cohort,:), ...
    KL_vecotr(train_cohort,:), MS1_vecotr(train_cohort,:), MS2_vecotr(train_cohort,:),... 
    PC_vecotr1(train_cohort,:), PC_vecotr2(train_cohort,:), Treat_vecotr(train_cohort,:),...
    MGUS_label(train_cohort,:), MM_label(train_cohort,:)];

test_Matrix = [PatientVector(test_cohort,:), sumMGUS_vector(test_cohort,:), ...
    sumNotMM_vector(test_cohort,:), sumProtein_vector(test_cohort,:), sumSMM_vector(test_cohort,:), ...
    KL_vecotr(test_cohort,:), MS1_vecotr(test_cohort,:), MS2_vecotr(test_cohort,:),... 
    PC_vecotr1(test_cohort,:), PC_vecotr2(test_cohort,:), Treat_vecotr(test_cohort,:),...
    MGUS_label(test_cohort,:), MM_label(test_cohort,:)];


valid_Matrix = [PatientVector(valid_cohort,:), sumMGUS_vector(valid_cohort,:), ...
    sumNotMM_vector(valid_cohort,:), sumProtein_vector(valid_cohort,:), sumSMM_vector(valid_cohort,:), ...
    KL_vecotr(valid_cohort,:), MS1_vecotr(valid_cohort,:), MS2_vecotr(valid_cohort,:),... 
    PC_vecotr1(valid_cohort,:), PC_vecotr2(valid_cohort,:), Treat_vecotr(valid_cohort,:),...
    MGUS_label(valid_cohort,:), MM_label(valid_cohort,:)];


%% normalize(?)
trainMatrixN = bsxfun(@rdivide,bsxfun(@minus,train_Matrix,min(train_Matrix,[],2)), max(train_Matrix,[],2));
testMatrixN = bsxfun(@rdivide,bsxfun(@minus,test_Matrix,min(test_Matrix,[],2)), max(test_Matrix,[],2));
validMatrixN = bsxfun(@rdivide,bsxfun(@minus,valid_Matrix,min(valid_Matrix,[],2)), max(valid_Matrix,[],2));

%% deep NN - MGUS
trainX = train_Matrix;
validX = valid_Matrix;
testX = test_Matrix;
% trainX = num2cell(train_Matrix,2);
% trainX = cellfun(@transpose,trainX,'UniformOutput',false);
x = categorical(true_train_label(train_cohort));
trainY = categorical(true_train_label(train_cohort));
testY = categorical(true_train_label(test_cohort));
validY = categorical(true_train_label(valid_cohort));

addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))

numFeatures = size(train_Matrix,2);
numClasses = 2;
miniBatchSize = 20;
classes = unique(trainY).';

for i = 1:numClasses
    classFrequency(i) = sum(categorical(trainY(:)) == classes(i));
    classWeights(i) = numel(trainY(:))/(numClasses*classFrequency(i));
end

layers = [
    featureInputLayer(numFeatures,"Name","featureinput","Normalization","zscore")
    fullyConnectedLayer(50,"Name","fc1")
    reluLayer("Name","relu")
    fullyConnectedLayer(50,"Name","fc2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(50,"Name","fc3")
    reluLayer("Name","relu3")
    fullyConnectedLayer(numClasses,"Name","fc4")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput", "Classes", classes, "ClassWeights",classWeights)];

options = trainingOptions('adam',...
    'MaxEpochs',100,...
    'MiniBatchSize',miniBatchSize,...
    'InitialLearnRate',1e-4,...
    "Shuffle","every-epoch",...
    'Verbose',false,...
    'Plots','training-progress');

netMGUS = trainNetwork(trainX,trainY,layers,options);
Ypred_train = classify(netMGUS,trainX);
Ypred_train_num = zeros(size(Ypred_train,1),1);
for j = 1:sum(train_cohort)
    if Ypred_train(j) == classes(1)
        Ypred_train_num(j) = 0;
    elseif Ypred_train(j) == classes(2)
        Ypred_train_num(j) = 1;
    end
end


Ypred_test = classify(netMGUS,testX);
Ypred_test_num = zeros(size(Ypred_test,1),1);
for j = 1:sum(test_cohort)
    if Ypred_test(j) == classes(1)
        Ypred_test_num(j) = 0;
    elseif Ypred_test(j) == classes(2)
        Ypred_test_num(j) = 1;
    end
end

[sensitivity_tr_dnn,specificity_tr_dnn,...
    PPV_tr_dnn,NPV_tr_dnn,...
    Accuracy_tr_dnn,F1_score_tr_dnn] =...
    CalcPerformance(Ypred_train_num,true_train_label(train_cohort));

[sensitivity_te_dnn,specificity_te_dnn,...
    PPV_te_dnn,NPV_te_dnn,...
    Accuracy_te_dnn,F1_score_te_dnn] =...
    CalcPerformance(Ypred_test_num,true_train_label(test_cohort));

Ypred_valid = classify(netMGUS,validX);
Ypred_valid_num = zeros(size(Ypred_valid,1),1);
for j = 1:sum(valid_cohort)
    if Ypred_valid(j) == classes(1)
        Ypred_valid_num(j) = 0;
    elseif Ypred_valid(j) == classes(2)
        Ypred_valid_num(j) = 1;
    end
end

[sensitivity_va_dnn,specificity_va_dnn,...
    PPV_va_dnn,NPV_va_dnn,...
    Accuracy_va_dnn,F1_score_va_dnn] =...
    CalcPerformance(Ypred_valid_num,true_train_label(valid_cohort));

yhatMGUS_DNN = zeros(size(true_train_label,1),1);
yhatMGUS_DNN(train_cohort) = Ypred_train_num;
yhatMGUS_DNN(test_cohort) = Ypred_test_num;
yhatMGUS_DNN(valid_cohort) = Ypred_valid_num;

%% ROC
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))

[FPRdnn_tr,TPRdnn_tr,Thrednn_tr,AUCdnn_tr] = perfcurve(true_train_label(train_cohort) == 1,Ypred_train_num,1,'XVals',[0:0.05:1]);
[TPRdnn_tr,FPRdnn_tr] = roc([true_train_label(train_cohort) == 1,Ypred_train_num]);
% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(train_cohort) == 1,Ypred_train_num])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(train_cohort) == 1;
yy = Ypred_train_num;
tic;
for i = 1:nsamp
   ind = randi([1,length(yy)],[nsamp,1]);
   y = yy(ind);
   t = tt(ind);
   [A_tr(i),Aci_tr(i,:)] = auc([t,y],alpha,'hanley');
   [TPR_tr(:,i),FPR_tr(:,i)] = roc([y,t]);
   [SEN_tr_CI(i),SPE_tr_CI(i),...
    PPV_tr_CI(i),NPV_tr_CI(i),...
    Accu_tr_CI(i),F1_score_tr_CI(i)] =...
    CalcPerformance(y,t);
    % plot([0; tpr], [1 ; prec], style);
    [prec_tr(:,i), tpr_tr(:,i), ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end    

[FPRdnn_te,TPRdnn_te,Thrednn_te,AUCdnn_te] = perfcurve(true_train_label(test_cohort) == 1,Ypred_test_num,1,'XVals',[0:0.05:1]);
[TPRdnn_te,FPRdnn_te] = roc([true_train_label(test_cohort) == 1,Ypred_test_num]);
% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(test_cohort) == 1,Ypred_test_num])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(test_cohort) == 1;
yy = Ypred_test_num;
tic;
for i = 1:nsamp
   ind = randi([1,length(yy)],[nsamp,1]);
   y = yy(ind);
   t = tt(ind);
   [A_te(i),Aci_te(i,:)] = auc([t,y],alpha,'hanley');
   [TPR_te(:,i),FPR_te(:,i)] = roc([y,t]);
   [SEN_te_CI(i),SPE_te_CI(i),...
    PPV_te_CI(i),NPV_te_CI(i),...
    Accu_te_CI(i),F1_score_te_CI(i)] =...
    CalcPerformance(y,t);
   [prec_te(:,i), tpr_te(:,i), ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end 

[FPRdnn_va,TPRdnn_va,Thrednn_va,AUCdnn_va] = perfcurve(true_train_label(valid_cohort) == 1,Ypred_valid_num,1,'XVals',[0:0.05:1]);
[TPRdnn_va,FPRdnn_va] = roc([true_train_label(valid_cohort) == 1,Ypred_valid_num]);
% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(valid_cohort) == 1,Ypred_valid_num])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(valid_cohort) == 1;
yy = Ypred_valid_num;
tic;
for i = 1:nsamp
   ind = randi([1,length(yy)],[nsamp,1]);
   y = yy(ind);
   t = tt(ind);
   [A_va(i),Aci_va(i,:)] = auc([t,y],alpha,'hanley');
   [TPR_va(:,i),FPR_va(:,i)] = roc([y,t]);
   [SEN_va_CI(i),SPE_va_CI(i),...
    PPV_va_CI(i),NPV_va_CI(i),...
    Accu_va_CI(i),F1_score_va_CI(i)] =...
    CalcPerformance(y,t);
   [prec_va(:,i), tpr_va(:,i), ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end 

A = [A_tr; A_te; A_va];
Aci = [Aci_tr; Aci_te; Aci_va];

CI_fun = @(x) [mean(x),mean(x) - 1.96.*(std(x)./sqrt(numel(x))), ...
               mean(x) + 1.96.*(std(x)./sqrt(numel(x)))];
CI_AUC_tr_dnn = CI_fun(A_tr);
CI_PPV_tr_dnn = CI_fun(PPV_tr_CI);
CI_SEN_tr_dnn = CI_fun(SEN_tr_CI);
CI_ACC_tr_dnn = CI_fun(Accu_tr_CI);
CI_F1_tr_dnn = CI_fun(F1_score_tr_CI);
CI_tr_DNN = [CI_AUC_tr_dnn;CI_PPV_tr_dnn;CI_SEN_tr_dnn;CI_ACC_tr_dnn;CI_F1_tr_dnn];

CI_AUC_te_dnn = CI_fun(A_te);
CI_PPV_te_dnn = CI_fun(PPV_te_CI);
CI_SEN_te_dnn = CI_fun(SEN_te_CI);
CI_ACC_te_dnn = CI_fun(Accu_te_CI);
CI_F1_te_dnn = CI_fun(F1_score_te_CI);
CI_te_DNN = [CI_AUC_te_dnn;CI_PPV_te_dnn;CI_SEN_te_dnn;CI_ACC_te_dnn;CI_F1_te_dnn];

CI_AUC_va_dnn = CI_fun(A_va);
CI_PPV_va_dnn = CI_fun(PPV_va_CI);
CI_SEN_va_dnn = CI_fun(SEN_va_CI);
CI_ACC_va_dnn = CI_fun(Accu_va_CI);
CI_F1_va_dnn = CI_fun(F1_score_va_CI);
CI_va_DNN = [CI_AUC_va_dnn;CI_PPV_va_dnn;CI_SEN_va_dnn;CI_ACC_va_dnn;CI_F1_va_dnn];

save('DeepNetMGUSAUC_plot.mat','A','Aci',...
    "CI_va_DNN","CI_te_DNN","CI_tr_DNN",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va");

%% deep NN - MM
trainX = train_Matrix;
validX = valid_Matrix;
testX = test_Matrix;

trainY = categorical(true_train_labelmm(train_cohort));
testY = categorical(true_train_labelmm(test_cohort));
validY = categorical(true_train_labelmm(valid_cohort));

numFeatures = size(train_Matrix,2);
numClasses = 2;
miniBatchSize = 20;
classes = unique(trainY).';

for i = 1:numClasses
    classFrequency(i) = sum(categorical(trainY(:)) == classes(i));
    classWeights(i) = numel(trainY(:))/(numClasses*classFrequency(i));
end

layers = [
    featureInputLayer(numFeatures,"Name","featureinput","Normalization","zscore")
    fullyConnectedLayer(80,"Name","fc1")
    batchNormalizationLayer
    reluLayer("Name","relu")
    fullyConnectedLayer(80,"Name","fc2")
    reluLayer("Name","relu2")
    fullyConnectedLayer(80,"Name","fc3")
    batchNormalizationLayer
    reluLayer("Name","relu3")
    fullyConnectedLayer(80,"Name","fc4")
    reluLayer("Name","relu4")
    fullyConnectedLayer(80,"Name","fc5")
    reluLayer("Name","relu5")
    fullyConnectedLayer(numClasses,"Name","fc6")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput", "Classes", classes, "ClassWeights",classWeights)];

options = trainingOptions('sgdm',...
    'MaxEpochs',200,...
    'MiniBatchSize',miniBatchSize,...
    'InitialLearnRate',1e-4,...
    "Shuffle","every-epoch",...
    'Verbose',false,...
    'Plots','training-progress');

netMM = trainNetwork(trainX,trainY,layers,options);
Ypred_train = classify(netMM,trainX);
Ypred_train_num = zeros(size(Ypred_train,1),1);
for j = 1:sum(train_cohort)
    if Ypred_train(j) == classes(1)
        Ypred_train_num(j) = 0;
    elseif Ypred_train(j) == classes(2)
        Ypred_train_num(j) = 1;
    end
end

Ypred_test = classify(netMM,testX);
Ypred_test_num = zeros(size(Ypred_test,1),1);
for j = 1:sum(test_cohort)
    if Ypred_test(j) == classes(1)
        Ypred_test_num(j) = 0;
    elseif Ypred_test(j) == classes(2)
        Ypred_test_num(j) = 1;
    end
end

sensitivitymm_te_dnn = sum(Ypred_test_num == 1 & true_train_label(test_cohort) == 1)./ ( sum(Ypred_test_num == 1 & true_train_label(test_cohort) == 1) +  sum(Ypred_test_num == 0 & true_train_label(test_cohort) == 1))
specificitymm_te_dnn = sum(Ypred_test_num == 0 & true_train_label(test_cohort) == 0)./ ( sum(Ypred_test_num == 0 & true_train_label(test_cohort) == 0) +  sum(Ypred_test_num == 1 & true_train_label(test_cohort) == 0))
PPVmm_te_dnn =  sum(Ypred_test_num == 1 & true_train_label(test_cohort) == 1)./ ( sum(Ypred_test_num == 1 & true_train_label(test_cohort) == 1) +  sum(Ypred_test_num == 1 & true_train_label(test_cohort) == 0))
NPVmm_te_dnn =  sum(Ypred_test_num == 0 & true_train_label(test_cohort) == 0)./ ( sum(Ypred_test_num == 0 & true_train_label(test_cohort) == 0) +  sum(Ypred_test_num == 0 & true_train_label(test_cohort) == 1))
Accuracymm_te_dnn = sum(Ypred_test_num==true_train_label(test_cohort))/numel(true_train_label(test_cohort))
F1_scoremm_te_dnn = 2/(1/sensitivity + 1/PPV)

Ypred_valid = classify(netMM,validX);
Ypred_valid_num = zeros(size(Ypred_valid,1),1);
for j = 1:sum(valid_cohort)
    if Ypred_valid(j) == classes(1)
        Ypred_valid_num(j) = 0;
    elseif Ypred_valid(j) == classes(2)
        Ypred_valid_num(j) = 1;
    end
end
sensitivitymm_va_dnn = sum(Ypred_valid_num == 1 & true_train_label(valid_cohort) == 1)./ ( sum(Ypred_valid_num == 1 & true_train_label(valid_cohort) == 1) +  sum(Ypred_valid_num == 0 & true_train_label(valid_cohort) == 1))
specificitymm_va_dnn = sum(Ypred_valid_num == 0 & true_train_label(valid_cohort) == 0)./ ( sum(Ypred_valid_num == 0 & true_train_label(valid_cohort) == 0) +  sum(Ypred_valid_num == 1 & true_train_label(valid_cohort) == 0))
PPVmm_va_dnn =  sum(Ypred_valid_num == 1 & true_train_label(valid_cohort) == 1)./ ( sum(Ypred_valid_num == 1 & true_train_label(valid_cohort) == 1) +  sum(Ypred_valid_num == 1 & true_train_label(valid_cohort) == 0))
NPVmm_va_dnn =  sum(Ypred_valid_num == 0 & true_train_label(valid_cohort) == 0)./ ( sum(Ypred_valid_num == 0 & true_train_label(valid_cohort) == 0) +  sum(Ypred_valid_num == 0 & true_train_label(valid_cohort) == 1))
Accuracymm_va_dnn = sum(Ypred_valid_num==true_train_label(valid_cohort))/numel(true_train_label(valid_cohort))
F1_scoremm_va_dnn = 2/(1/sensitivity + 1/PPV)

yhatMM_DNN = zeros(size(true_train_label,1),1);
yhatMM_DNN(train_cohort) = Ypred_train_num;
yhatMM_DNN(test_cohort) = Ypred_test_num;
yhatMM_DNN(valid_cohort) = Ypred_valid_num;

%% ROC
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))
clear TPR_tr FPR_tr TPR_te FPR_te TPR_va FPR_va prec_tr tpr_tr prec_te tpr_te prec_va tpr_va

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(train_cohort) == 1,Ypred_train_num])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(train_cohort) == 1;
yy = Ypred_train_num;
tic;
for i = 1:nsamp
   ind = randi([1,length(yy)],[nsamp,1]);
   y = yy(ind);
   t = tt(ind);
   [A_tr(i),Aci_tr(i,:)] = auc([t,y],alpha,'hanley');
   [TPR_tr(:,i),FPR_tr(:,i)] = roc([y,t]);
   [SEN_tr_CI(i),SPE_tr_CI(i),...
    PPV_tr_CI(i),NPV_tr_CI(i),...
    Accu_tr_CI(i),F1_score_tr_CI(i)] =...
    CalcPerformance(y,t);
   [prec_tr(:,i), tpr_tr(:,i), ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end    

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(test_cohort) == 1,Ypred_test_num])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(test_cohort) == 1;
yy = Ypred_test_num;
tic;
for i = 1:nsamp
   ind = randi([1,length(yy)],[nsamp,1]);
   y = yy(ind);
   t = tt(ind);
   [A_te(i),Aci_te(i,:)] = auc([t,y],alpha,'hanley');
   [TPR_te(:,i),FPR_te(:,i)] = roc([y,t]);
   [SEN_te_CI(i),SPE_te_CI(i),...
    PPV_te_CI(i),NPV_te_CI(i),...
    Accu_te_CI(i),F1_score_te_CI(i)] =...
    CalcPerformance(y,t);
   [prec_te(:,i), tpr_te(:,i), ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end 

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(valid_cohort) == 1,Ypred_valid_num])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(valid_cohort) == 1;
yy = Ypred_valid_num;
tic;
for i = 1:nsamp
   ind = randi([1,length(yy)],[nsamp,1]);
   y = yy(ind);
   t = tt(ind);
   [A_va(i),Aci_va(i,:)] = auc([t,y],alpha,'hanley');
   [TPR_va(:,i),FPR_va(:,i)] = roc([y,t]);
   [SEN_va_CI(i),SPE_va_CI(i),...
    PPV_va_CI(i),NPV_va_CI(i),...
    Accu_va_CI(i),F1_score_va_CI(i)] =...
    CalcPerformance(y,t);
   [prec_va(:,i), tpr_va(:,i), ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end 

A = [A_tr; A_te; A_va];
Aci = [Aci_tr; Aci_te; Aci_va];

CI_fun = @(x) [mean(x),mean(x) - 1.96.*(std(x)./sqrt(numel(x))), ...
               mean(x) + 1.96.*(std(x)./sqrt(numel(x)))];
CI_AUC_tr_dnn = CI_fun(A_tr);
CI_PPV_tr_dnn = CI_fun(PPV_tr_CI);
CI_SEN_tr_dnn = CI_fun(SEN_tr_CI);
CI_ACC_tr_dnn = CI_fun(Accu_tr_CI);
CI_F1_tr_dnn = CI_fun(F1_score_tr_CI);
CI_tr_DNN = [CI_AUC_tr_dnn;CI_PPV_tr_dnn;CI_SEN_tr_dnn;CI_ACC_tr_dnn;CI_F1_tr_dnn];

CI_AUC_te_dnn = CI_fun(A_te);
CI_PPV_te_dnn = CI_fun(PPV_te_CI);
CI_SEN_te_dnn = CI_fun(SEN_te_CI);
CI_ACC_te_dnn = CI_fun(Accu_te_CI);
CI_F1_te_dnn = CI_fun(F1_score_te_CI);
CI_te_DNN = [CI_AUC_te_dnn;CI_PPV_te_dnn;CI_SEN_te_dnn;CI_ACC_te_dnn;CI_F1_te_dnn];

CI_AUC_va_dnn = CI_fun(A_va);
CI_PPV_va_dnn = CI_fun(PPV_va_CI);
CI_SEN_va_dnn = CI_fun(SEN_va_CI);
CI_ACC_va_dnn = CI_fun(Accu_va_CI);
CI_F1_va_dnn = CI_fun(F1_score_va_CI);
CI_va_DNN = [CI_AUC_va_dnn;CI_PPV_va_dnn;CI_SEN_va_dnn;CI_ACC_va_dnn;CI_F1_va_dnn];

save('DeepNetMMAUC_plot.mat','A','Aci',...
    "CI_va_DNN","CI_te_DNN","CI_tr_DNN",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va");

%% DNN
Est_DNN_TR = zeros(6,1);
[Est_DNN_TR(1),Est_DNN_TR(2),Est_DNN_TR(3),Est_DNN_TR(4),Est_DNN_TR(5),Est_DNN_TR(6)] ...
    = CalcPerformance(true_train_label(train_cohort),yhatMGUS_DNN(train_cohort));

Est_DNN_TE = zeros(6,1);
[Est_DNN_TE(1),Est_DNN_TE(2),Est_DNN_TE(3),Est_DNN_TE(4),Est_DNN_TE(5),Est_DNN_TE(6)] ...
    = CalcPerformance(true_train_label(test_cohort),yhatMGUS_DNN(test_cohort));

Est_DNN_VA = zeros(6,1);
[Est_DNN_VA(1),Est_DNN_VA(2),Est_DNN_VA(3),Est_DNN_VA(4),Est_DNN_VA(5),Est_DNN_VA(6)] ...
    = CalcPerformance(true_train_label(valid_cohort),yhatMGUS_DNN(valid_cohort));

sensitivity = sum(yhatMGUS_DNN == 1 & true_train_label == 1)./ ( sum(yhatMGUS_DNN == 1 & true_train_label == 1) +  sum(yhatMGUS_DNN == 0 & true_train_label == 1))
specificity = sum(yhatMGUS_DNN == 0 & true_train_label == 0)./ ( sum(yhatMGUS_DNN == 0 & true_train_label == 0) +  sum(yhatMGUS_DNN == 1 & true_train_label == 0))
PPV =  sum(yhatMGUS_DNN == 1 & true_train_label == 1)./ ( sum(yhatMGUS_DNN == 1 & true_train_label == 1) +  sum(yhatMGUS_DNN == 1 & true_train_label == 0))
NPV =  sum(yhatMGUS_DNN == 0 & true_train_label == 0)./ ( sum(yhatMGUS_DNN == 0 & true_train_label == 0) +  sum(yhatMGUS_DNN == 0 & true_train_label == 1))
Accuracy = sum(yhatMGUS_DNN==true_train_label)/numel(true_train_label)
F1_score = 2/(1/sensitivity + 1/PPV)

writetable(array2table([Est_DNN_TR, Est_DNN_TE, Est_DNN_VA ]),"DNN_results.xlsx",'Sheet',"MGUS")
Est_DNN_TR = zeros(6,1);
[Est_DNN_TR(1),Est_DNN_TR(2),Est_DNN_TR(3),Est_DNN_TR(4),Est_DNN_TR(5),Est_DNN_TR(6)] ...
    = CalcPerformance(true_train_labelmm(train_cohort),yhatMM_DNN(train_cohort));

Est_DNN_TE = zeros(6,1);
[Est_DNN_TE(1),Est_DNN_TE(2),Est_DNN_TE(3),Est_DNN_TE(4),Est_DNN_TE(5),Est_DNN_TE(6)] ...
    = CalcPerformance(true_train_labelmm(test_cohort),yhatMM_DNN(test_cohort));

Est_DNN_VA = zeros(6,1);
[Est_DNN_VA(1),Est_DNN_VA(2),Est_DNN_VA(3),Est_DNN_VA(4),Est_DNN_VA(5),Est_DNN_VA(6)] ...
    = CalcPerformance(true_train_labelmm(valid_cohort),yhatMM_DNN(valid_cohort));

sensitivity = sum(yhatMM_DNN == 1 & true_train_labelmm == 1)./ ( sum(yhatMM_DNN == 1 & true_train_labelmm == 1) +  sum(yhatMM_DNN == 0 & true_train_labelmm == 1))
specificity = sum(yhatMM_DNN == 0 & true_train_labelmm == 0)./ ( sum(yhatMM_DNN == 0 & true_train_labelmm == 0) +  sum(yhatMM_DNN == 1 & true_train_labelmm == 0))
PPV =  sum(yhatMM_DNN == 1 & true_train_labelmm == 1)./ ( sum(yhatMM_DNN == 1 & true_train_labelmm == 1) +  sum(yhatMM_DNN == 1 & true_train_labelmm == 0))
NPV =  sum(yhatMM_DNN == 0 & true_train_labelmm == 0)./ ( sum(yhatMM_DNN == 0 & true_train_labelmm == 0) +  sum(yhatMM_DNN == 0 & true_train_labelmm == 1))
Accuracy = sum(yhatMM_DNN==true_train_labelmm)/numel(true_train_labelmm)
F1_score = 2/(1/sensitivity + 1/PPV)

writetable(array2table([Est_DNN_TR, Est_DNN_TE, Est_DNN_VA ]),"DNN_results.xlsx",'Sheet',"MM")


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