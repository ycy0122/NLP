clc;clear;close all
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\cohort1\Finished'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\cohort1\Lab'))
read_reflists = true;
save_name = 'Final_011123_Rule';
read_reflists = 1;
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

%% load word lists
if read_reflists == 1
    % Negation words
    fid = fopen('NegationModifiers.txt');
    negwords = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        negwords{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,negwords);    % remove empty cell
    negwords = negwords(~emptye_r_cell);
    fclose(fid);
    
    % MGUS word lists
    fid = fopen('DiseaseReferenceSetA.txt');
    refwordsMGUSA = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMGUSA{end+1} = buffer;
%         refwordsMGUSA{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMGUSA);    % remove empty cell
    refwordsMGUSA = refwordsMGUSA(~emptye_r_cell);
    fclose(fid);
    
    fid = fopen('DiseaseReferenceSetB.txt');
    refwordsMGUSB = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMGUSB{end+1} = buffer;
%         refwordsMGUSB{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMGUSB);    % remove empty cell
    refwordsMGUSB = refwordsMGUSB(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetC.txt');
    refwordsMGUSC = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMGUSC{end+1} = buffer;
%         refwordsMGUSC{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMGUSC);    % remove empty cell
    refwordsMGUSC = refwordsMGUSC(~emptye_r_cell);
    fclose(fid);

    % MM word lists
    fid = fopen('DiseaseReferenceSetMMA.txt');
    refwordsMMA = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMMA{end+1} = buffer;
%         refwordsMMA{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMMA);    % remove empty cell
    refwordsMMA = refwordsMMA(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetMMB.txt');
    refwordsMMB = {};
    while ~feof(fid)
        buffer = fgetl(fid);
%         refwordsMMB{end+1} = buffer;
        refwordsMMB{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMMB);    % remove empty cell
    refwordsMMB = refwordsMMB(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetMMC.txt');
    refwordsMMC = {};
    while ~feof(fid)
        buffer = fgetl(fid);
%         refwordsMMC{end+1} = strcat(" ",buffer," ");
        refwordsMMC{end+1} = buffer;
    end
    emptye_r_cell = cellfun(@isempty,refwordsMMC);    % remove empty cell
    refwordsMMC = refwordsMMC(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetMMD.txt');
    refwordsMMD = {};
    while ~feof(fid)
        buffer = fgetl(fid);
%         refwordsMMD{end+1} = buffer;
        refwordsMMD{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMMD);    % remove empty cell
    refwordsMMD = refwordsMMD(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetMME.txt');
    refwordsMME = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMME{end+1} = strcat(" ",buffer," ");
%         refwordsMME{end+1} = buffer;
    end
    emptye_r_cell = cellfun(@isempty,refwordsMME);    % remove empty cell
    refwordsMME = refwordsMME(~emptye_r_cell);
    fclose(fid);
    

    save('ReflistsMGUS.mat','refwordsMGUSA','refwordsMGUSB','refwordsMGUSC','negwords')
    save('ReflistsMM.mat','refwordsMMA','refwordsMMB','refwordsMMC','refwordsMMD','refwordsMME','negwords')
%     load('Reflists.mat','refwordsC')
else
    load('ReflistsMGUS.mat','refwordsMGUSA','refwordsMGUSB','refwordsMGUSC','negwords')
    load('ReflistsMM.mat','refwordsMMA','refwordsMMB','refwordsMMC','refwordsMMD','refwordsMME','negwords')
%     load('Reflists.mat','refwordsC')
end

%%
load(['NLP_700_clinical_processed_test.mat'],'ReportTime','clinicaldata','reportsencellarray','N_report','reportcell')

ReportTime = strtrim(ReportTime);
[pid, ic, ia] = unique(table2array(clinicaldata(:,1)));
N_report = size(clinicaldata,1);

TEST_filename = "True_MGUSPROG.xlsx";
TESTtable = readtable(TEST_filename);
true_train_label = table2array(TESTtable(:,3));

new_id_list = [];
%  pid = table2array(readtable('FP_MMmspk_new2.txt'));

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
reportsencellarray = reportsencellarray(1,new_id_list);
ReportTime = ReportTime(1,new_id_list);        
N_report = size(clinicaldata,1);

% Get True MM labels
% Get True MM labels
train_true_filenamemm = "True_MGUSPROG.xlsx";
true_train_labelmm = readtable(train_true_filenamemm);
% true_train_labelmm_id = cellfun(@str2num,table2array(true_train_labelmm(:,1)));
% [iaamm, ibbmm, iccmm] = intersect(pid,true_train_labelmm_id);
true_train_labelmm = table2array(true_train_labelmm(:,4));

%% treatment file 
Treat_records = readtable('myeloma_drug_use_history.xlsx');
%%
tic
% reportsencellarray = reportsencellarray.';
    for i = 1:N_report
        disp(sprintf('progress: %d out of %d reports', i, N_report));
        if ~isempty(reportsencellarray{1,i})
            [ind_MGUS(i),ind_protein(i),ind_MM(i),ind_SMM(i),ind_Treat(i)] = ...
                featurizeALL(reportsencellarray{1,i}.', refwordsMGUSA, refwordsMGUSB, refwordsMMC, refwordsMMA, refwordsMME,  refwordsMMD, negwords);
%             ind_NotMM(i) = ~ind_MM(i) || ~ind_SMM(i);
            [ind_NotMM(i),~,~] = featurizeC(reportsencellarray{1,i}.', refwordsMGUSC, negwords);


        else
            ind_MGUS(i) = 0;
            ind_protein(i) = 0;
            ind_NotMM(i) = 0;
            ind_SMM(i) = 0;
            ind_MM(i) = 0;
            ind_Treat(i) = 0;
        end
    end
TotalRunTime = toc;
%% condiions:
% clinicaldata = clinicaldata(new_id_list,:);
% C1, C2, C3, C1+C2, C1+C3, C1+C2+C3,  C2+C3, C2+C4, C2+C3+C4, C3+C4
ConditionMGUS_matrix = zeros(length(pid),3);
ConditionMM_matrix = zeros(length(pid),5);
CsMGUS_matrix = zeros(length(pid),4);
CsMGUS = zeros(1,4);
CsMM_matrix = zeros(length(pid),5);
CsMM = zeros(length(pid),5);

% ML_condtions
CDs_MGUS = zeros(length(pid),4);
CDs_MM = zeros(length(pid),5);

MM_set1 = [];
MM_report = cell(length(pid),1);
SMM_report = cell(length(pid),1);
mspk_report = cell(length(pid),1);
kl_report = cell(length(pid),1);
Treat_report = cell(length(pid),1);
PC_report = cell(length(pid),1);

MGUS_set1 = [];
MGUS_report = cell(length(pid),1);
Protein_report = cell(length(pid),1);
NotMM_report = cell(length(pid),1);
earliest_timeMM = cell(length(pid),1);

combsMM = dec2base(0:power(2,5)-1,2) - '0';
combsMM = combsMM(2:end,:);

combsMGUS = dec2base(0:power(2,3)-1,2) - '0';
combsMGUS = combsMGUS(2:end,:);  

count = 1;
Treat_matrix = [];
MMsmm_matrix = [];

%%
for i = 1:length(pid)
    PC_time = []; MSPK_time = [];   KLRT_time = []; Treat_time = [];
    MM_time = []; SMM_time = [];

    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    this_MGUS = any(ind_MGUS(this_pid) == 1);           %  C1
    this_PROP = any(ind_protein(this_pid) == 1);        %  C2
    this_MSPK = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mgus)); 
    this_NoMM = any(ind_NotMM(this_pid) == 1);          %  C4
    
    this_MGUS_MK = MK_PatientSSN==table2array(clinicaldata(this_pid(1),1));
    this_MK_DATE = mspike_date(this_MGUS_MK & ind1mspike);
    
    min_id = min(this_pid);
    max_id = max(this_pid);

    MGUS_report_tmp = min_id + (find(ind_MGUS(min_id:max_id)==1))-1;
    Protein_report_tmp = min_id + (find(ind_protein(min_id:max_id)==1))-1;
    NotMM_report_tmp = min_id + (find(ind_NotMM(min_id:max_id)==1))-1;

    MGUS_report{i} = MGUS_report_tmp;
    Protein_report{i} = Protein_report_tmp;
    NotMM_report{i} = NotMM_report_tmp;
    mspk_report{i} = this_MSPK;

    % For MGUS
    CDs_MGUS(i,:) = [this_MGUS, this_PROP, this_MSPK, this_NoMM];
    % C1
    ConditionMGUS_matrix(i,1) =  (this_PROP ~= 0 && this_NoMM~=0);
    % C2
    ConditionMGUS_matrix(i,2) =  (this_MSPK~=0 && this_NoMM~=0);
    % C3 
    ConditionMGUS_matrix(i,3) =  (this_MGUS ~= 0 && this_PROP~=0);

    % find the earliest time
    this_MM_T = (ind_MM(this_pid) == 1);           %  C1
    MM_time = datetime(strtrim(ReportTime(this_pid(this_MM_T))));

    this_SMM_T = (ind_SMM(this_pid) == 1);   
    SMM_time =  datetime(strtrim(ReportTime(this_pid(this_SMM_T))));

    this_KLRT_T = (intersect(table2array(clinicaldata(this_pid(1),1)),ind_klratio));
    if ~isempty(this_KLRT_T)
        KLRT_time_id = find(this_KLRT_T == KL_PatientSSN);
        KLRT_time_id2 = ind2klratio(KLRT_time_id);
        KLRT_time = KL_date(KLRT_time_id(KLRT_time_id2));
    end
    this_MSPK_T = (intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mm));
    if ~isempty(this_MSPK_T)
        MSPK_time_id = find(this_MSPK_T == MK_PatientSSN);
        MSPK_time_id2 = ind2mspike(MSPK_time_id);
        MSPK_time = mspike_date(MSPK_time_id(MSPK_time_id2));
    end
    this_PC_T = (intersect(table2array(clinicaldata(this_pid(1),1)),ind_PC));
    if ~isempty(this_PC_T)
        pc_id = find(pid(i) == PCPatientSSN);
        PC_val = PC(pc_id);
        if any(PC_val>=10)
            PC_time = PC_date(pc_id);
        else
            PC_time = [];
        end
    end
        
    %% New C4
    this_Drug_T = pid(i) == table2array(Treat_records(:,1));
    if sum(this_Drug_T) == 0
        this_Treat = 0;
        Treat_time = [];
        this_Treat_T = [];
    else
%     Treat_time =  datetime(strtrim(ReportTime(this_pid(this_Treat_T))));
        this_Treat = 1;
        Treat_time = sort(datetime(table2array(Treat_records(this_Drug_T,3))),'ascend');
        this_Treat_T = Treat_time;
    end
    
%     this_Treat_T = (ind_Treat(this_pid) == 1);          %  C4
    this_MM = any(ind_MM(this_pid) == 1);           %  C1
    this_SMM = any(ind_SMM(this_pid) == 1);         
%         this_CRAB = any(ind_crab(this_pid) == 1);        %  C2
    this_KLRT = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_klratio));
    this_MSPK = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mm));
    this_PC = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_PC));
%     this_Treat = any(ind_Treat(this_pid) == 1);          %  C4
    Treat_matrix(i,1) = this_Treat;
    MMsmm_matrix(i,1) = this_SMM | this_MM;
    
    if this_PC ~= 0
        pc_id = find(pid(i) == PCPatientSSN);
        PC_val = PC(pc_id);
        if length(PC_val) >  1
            PC_val;
        end
        S2 = (this_SMM ~= 0) &&  any((PC_val >= 10));
        S4 = (this_MM ~= 0) && any(PC_val >= 10);
    else
        S2 = 0;
        S4 = 0;
    end

    S1 = (this_SMM ~= 0) && (this_MSPK ~=0);
    S3 = (this_MM ~= 0) && (this_KLRT ~=0);
    S5 = (this_MM ~= 0) && (this_Treat~=0);
    
    CsMM(i,:) = [this_MM, this_SMM, this_MSPK, this_KLRT, this_Treat];
    CsMM_matrix(i,:) = [S1 S2 S3 S4 S5];
    CDs_MM(i,:) = [this_MM, this_SMM, this_MSPK, this_KLRT, this_Treat];
    
    min_id = min(this_pid);
    max_id = max(this_pid);

    MM_report_tmp = min_id + (find(ind_MM(min_id:max_id)==1))-1;
    SMM_report_tmp = min_id + (find(ind_SMM(min_id:max_id)==1))-1;
    treat_report_tmp = min_id + (find(ind_Treat(min_id:max_id)==1))-1;

    MM_report{i} = MM_report_tmp;
    SMM_report{i} = SMM_report_tmp;
    Treat_report{i} = treat_report_tmp;


    % Condition1: S1
    ConditionMM_matrix(i,1) =  S1;
    % Condition2: S2
    ConditionMM_matrix(i,2) =  S2;
    % Condition3: S3
    ConditionMM_matrix(i,3) =  S3;
    % Condition3: S4
    ConditionMM_matrix(i,4) =  S4;
    % Condition3: S5
    ConditionMM_matrix(i,5) =  S5;


    MM_set1(i) = any(ConditionMM_matrix(i,logical(combsMM(end,:))),2);


    if  MM_set1(i) == 1
        %find earliest report time
        if ~isempty(this_PC_T)
            earliest_timeMM{i} = min(PC_time);
        end
        if ~isempty(MSPK_time) & ~isempty(Treat_time) & isempty(earliest_timeMM{i})
            clear id
            for j = 1:length(MSPK_time)
                id(j,:) = Treat_time > MSPK_time(j);
            end
            earliest_id = find(sum(id)>0,1);
            earliest_timeMM{i} = Treat_time(earliest_id);
        end
        if ~isempty(MSPK_time) & isempty(Treat_time) & isempty(earliest_timeMM{i})
            earliest_timeMM{i} = min(MSPK_time);
        end
        if ~isempty(KLRT_time) & ~isempty(Treat_time) & isempty(earliest_timeMM{i})
            clear id
            for j = 1:length(KLRT_time)
                id(j,:) = Treat_time > KLRT_time(j);
            end
            earliest_id = find(sum(id)>0,1);
            earliest_timeMM{i} = Treat_time(earliest_id);
        end
        if ~isempty(KLRT_time) & isempty(Treat_time) & isempty(earliest_timeMM{i})
            earliest_timeMM{i} = min(KLRT_time);
        end
        if ~isempty(this_Treat_T) & isempty(earliest_timeMM{i})
            earliest_timeMM{i} = min(Treat_time);
        end

%         earliest_time{i} = min([PC_time.',MSPK_time.',KLRT_time.',Treat_time,MM_time,SMM_time]);
    else
        earliest_timeMM{i} = [];
    end
    
%     % check if mspike time is after the diagnosis time of MM
%     if  ~isempty(earliest_timeMM{i}) && ~isempty(this_MK_DATE)
%         if all(earliest_timeMM{i} <= this_MK_DATE)
%             ConditionMGUS_matrix(i,2) = 0 ;
%         end
%     end


    
    % check if 1st treat time is after the diagnosis time of MM
    if  (this_Treat ~= 0 && ~isempty(this_MK_DATE))
        Tp(count,1) = table2array(clinicaldata(this_pid(1),1));
        FirstTreat(count,1) = Treat_time(1);
        Firstmspike(count,1) = this_MK_DATE(1);
        if all(Treat_time(1) <= this_MK_DATE)
            ConditionMGUS_matrix(i,2) = 0 ;
            CsMGUS_matrix(i,3) = 0;
        end
        CsMGUS(count,:) = CsMGUS_matrix(i,:);
        count = count + 1;
    end

end

%% Patientws without labs/Treatment
% [i_mk,~] = ismember(pid, MK_unique);
% [i_kl,~] = ismember(pid, KL_unique);
% [i_pc,~] = ismember(pid, PC_unique);
% ids_to_remove = ~i_mk & ~i_kl & ~ i_pc & logical(MMsmm_matrix) & ~logical(Treat_matrix);
% pids_to_remove = pid(ids_to_remove);
% writetable(array2table(pids_to_remove),'Pids_to_remove.txt')
%% specificity + sensitivity
% take 400 only
MGUS_set1 = [];
MGUS_set1 = any(ConditionMGUS_matrix(:,logical(combsMGUS(end,:))),2);
MGUS_label = ismember(pid,(pid(MGUS_set1)));

classify_label = ismember(pid,(pid(MGUS_set1)));
sensitivity = sum(classify_label == 1 & true_train_label == 1)./ ( sum(classify_label == 1 & true_train_label == 1) +  sum(classify_label == 0 & true_train_label == 1))
specificity = sum(classify_label == 0 & true_train_label == 0)./ ( sum(classify_label == 0 & true_train_label == 0) +  sum(classify_label == 1 & true_train_label == 0))
PPV =  sum(classify_label == 1 & true_train_label == 1)./ ( sum(classify_label == 1 & true_train_label == 1) +  sum(classify_label == 1 & true_train_label == 0))
NPV =  sum(classify_label == 0 & true_train_label == 0)./ ( sum(classify_label == 0 & true_train_label == 0) +  sum(classify_label == 0 & true_train_label == 1))
Accuracy = sum(classify_label==true_train_label)/numel(true_train_label)
F1_score = 2/(1/sensitivity + 1/PPV)
%%
FP_MGUS = classify_label == 1 & true_train_label == 0;
FP_MGUS_pid = pid(FP_MGUS);
writetable(array2table(FP_MGUS_pid),['FP_MGUS',save_name,'.txt'])
FN_MGUS = classify_label == 0 & true_train_label == 1;
FN_MGUS_pid = pid(FN_MGUS);
writetable(array2table(FN_MGUS_pid),['FN_MGUS',save_name,'.txt'])


%% MGUS performance
classify_labelmm = ismember(pid,(pid(logical(MM_set1.'))));
% classify_labelmm = MM_label;
sensitivitymm = sum(classify_labelmm == 1 & true_train_labelmm == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm == 1) +  sum(classify_labelmm == 0 & true_train_labelmm == 1))
specificitymm = sum(classify_labelmm == 0 & true_train_labelmm == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm == 0) +  sum(classify_labelmm == 1 & true_train_labelmm == 0))
PPVmm =  sum(classify_labelmm == 1 & true_train_labelmm == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm == 1) +  sum(classify_labelmm == 1 & true_train_labelmm == 0))
NPVmm =  sum(classify_labelmm == 0 & true_train_labelmm == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm == 0) +  sum(classify_labelmm == 0 & true_train_labelmm == 1))
Accuracymm = sum(classify_labelmm==true_train_labelmm)/numel(true_train_labelmm)
F1_scoremm = 2/(1/sensitivitymm + 1/PPVmm)

%% train, test, validation id & pid
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\Code\HSICLasso'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\Code\OtherMLutilities'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NLP_MGUS'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\AlgorithmUtilities'))

T = readtable('NLP_performance_092922.xlsx','Sheet','Details','Range','J1:J701');
train_cohort = strcmp(table2array(T),'TR');
test_cohort = strcmp(table2array(T),'TE');
valid_cohort = strcmp(table2array(T),'VA');

% ROC
clear TPR_tr FPR_tr  TPR_te TPR_va FPR_va FPR_te prec_tr tpr_tr prec_te tpr_te prec_va tpr_va
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(train_cohort),MGUS_label(train_cohort)])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(train_cohort);
yy = MGUS_label(train_cohort);
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
p = auc_bootstrap([true_train_label(test_cohort),MGUS_label(test_cohort)])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(test_cohort);
yy = MGUS_label(test_cohort);
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
p = auc_bootstrap([true_train_label(valid_cohort),MGUS_label(valid_cohort)])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(valid_cohort);
yy = MGUS_label(valid_cohort);
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
CI_AUC_tr_rule = CI_fun(A_tr);
CI_PPV_tr_rule = CI_fun(PPV_tr_CI);
CI_SEN_tr_rule = CI_fun(SEN_tr_CI);
CI_ACC_tr_rule= CI_fun(Accu_tr_CI);
CI_F1_tr_rule = CI_fun(F1_score_tr_CI);
CI_tr_RULE = [CI_AUC_tr_rule;CI_PPV_tr_rule;CI_SEN_tr_rule;CI_ACC_tr_rule;CI_F1_tr_rule];

CI_AUC_te_rule = CI_fun(A_te);
CI_PPV_te_rule = CI_fun(PPV_te_CI);
CI_SEN_te_rule = CI_fun(SEN_te_CI);
CI_ACC_te_rule = CI_fun(Accu_te_CI);
CI_F1_te_rule = CI_fun(F1_score_te_CI);
CI_te_RULE = [CI_AUC_te_rule;CI_PPV_te_rule;CI_SEN_te_rule;CI_ACC_te_rule;CI_F1_te_rule];

CI_AUC_va_rule = CI_fun(A_va);
CI_PPV_va_rule = CI_fun(PPV_va_CI);
CI_SEN_va_rule = CI_fun(SEN_va_CI);
CI_ACC_va_rule = CI_fun(Accu_va_CI);
CI_F1_va_rule = CI_fun(F1_score_va_CI);
CI_va_RULE = [CI_AUC_va_rule;CI_PPV_va_rule;CI_SEN_va_rule;CI_ACC_va_rule;CI_F1_va_rule];

save('RuleMGUSAUC_plot.mat','A','Aci',...
    "CI_va_RULE","CI_te_RULE","CI_tr_RULE",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va");


%% RULE MM
% ROC
MM_label = classify_labelmm;

addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))
clear TPR_tr FPR_tr  TPR_te TPR_va FPR_va FPR_te prec_tr tpr_tr prec_te tpr_te prec_va tpr_va 
% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(train_cohort),MM_label(train_cohort)])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(train_cohort);
yy = MM_label(train_cohort);
tic;
for i = 1:nsamp
%    ind = randi([1,length(yy)],[nsamp,1]);
   ind = randperm([length(yy)],[nsamp]);
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
p = auc_bootstrap([true_train_labelmm(test_cohort),MM_label(test_cohort)])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(test_cohort);
yy = MM_label(test_cohort);
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
    [prec_te(:,i), tpr_te(:,i), ~, ~] = prec_rec(y, t,'instanceCount',0.1.*ones(1,nsamp));
end 

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(valid_cohort),MM_label(valid_cohort)])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(valid_cohort);
yy = MM_label(valid_cohort);
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
CI_AUC_tr_rule = CI_fun(A_tr);
CI_PPV_tr_rule = CI_fun(PPV_tr_CI);
CI_SEN_tr_rule = CI_fun(SEN_tr_CI);
CI_ACC_tr_rule= CI_fun(Accu_tr_CI);
CI_F1_tr_rule = CI_fun(F1_score_tr_CI);
CI_tr_RULE = [CI_AUC_tr_rule;CI_PPV_tr_rule;CI_SEN_tr_rule;CI_ACC_tr_rule;CI_F1_tr_rule];

CI_AUC_te_rule = CI_fun(A_te);
CI_PPV_te_rule = CI_fun(PPV_te_CI);
CI_SEN_te_rule = CI_fun(SEN_te_CI);
CI_ACC_te_rule = CI_fun(Accu_te_CI);
CI_F1_te_rule = CI_fun(F1_score_te_CI);
CI_te_RULE = [CI_AUC_te_rule;CI_PPV_te_rule;CI_SEN_te_rule;CI_ACC_te_rule;CI_F1_te_rule];

CI_AUC_va_rule = CI_fun(A_va);
CI_PPV_va_rule = CI_fun(PPV_va_CI);
CI_SEN_va_rule = CI_fun(SEN_va_CI);
CI_ACC_va_rule = CI_fun(Accu_va_CI);
CI_F1_va_rule = CI_fun(F1_score_va_CI);
CI_va_RULE = [CI_AUC_va_rule;CI_PPV_va_rule;CI_SEN_va_rule;CI_ACC_va_rule;CI_F1_va_rule];

save('RuleMMAUC_plot.mat','A','Aci',...
    "CI_va_RULE","CI_te_RULE","CI_tr_RULE",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va");

%% TFIDF - ML algorithms
%% load texts as input for bag-of-words(BOW) algorithms
nminFeatures = 5000;  % minimum # of appearances for this term to be a feature
removeStopWords = 1; 
doStem = 0;

addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\AlgorithmUtilities\custom_func\Random-Forest-Matlab-master\lib'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\MatlabNLP-master\funcs\funcs'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NLP_MGUS'))

% tic
% [featureVector,headers] = featurizeTrainReports(reportcell, nminFeatures, removeStopWords, doStem);
% save('NLP_700_clinical_featureVector5000_updated.mat','featureVector','headers')
% BOWtime = toc
load('NLP_700_clinical_featureVector5000_updated.mat','featureVector','headers')
%%
patients_lables =  true_train_label+1;
patients_lablesmm =  true_train_labelmm+1;
PatientVector = zeros(length(true_train_label),size(featureVector,2));

for i = 1:length(pid)
    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    PatientVector(i,:) = sum(featureVector(this_pid(1):this_pid(end),:));
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
 

for i = 1:length(pid)
    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    sumMGUS_vector(i,1) = sum(ind_MGUS(this_pid(1):this_pid(end)));
    sumMM_vector(i,1) = sum(ind_MM(this_pid(1):this_pid(end)));
    sumNotMM_vector(i,1) = sum(ind_NotMM(this_pid(1):this_pid(end)));
    sumProtein_vector(i,1) = sum(ind_protein(this_pid(1):this_pid(end)));
    sumSMM_vector(i,1) = sum(ind_SMM(this_pid(1):this_pid(end)));
%     sumTreat_vector(i,1) = sum(ind_Treat(this_pid(1):this_pid(end)));
end

% transform bow to bow_to_tfidf
tfidf_vector = full(bow_to_tfidf(PatientVector));
% train_Matrix = [full(bow_to_tfidf([PatientVector])), sumMGUS_vector, ...
%     sumNotMM_vector, sumProtein_vector, sumSMM_vector, ...
%     KL_vecotr, MS1_vecotr, MS2_vecotr, PC_vecotr1, PC_vecotr2, Treat_vecotr, MGUS_set1, MM_set1.'];
train_Matrix = [full(bow_to_tfidf([PatientVector, sumMGUS_vector, ...
    sumNotMM_vector, sumProtein_vector, sumSMM_vector])), sumMGUS_vector, ...
    sumNotMM_vector, sumProtein_vector, sumSMM_vector, ...
    KL_vecotr, MS1_vecotr, MS2_vecotr, PC_vecotr1, PC_vecotr2, Treat_vecotr,...
    MGUS_set1, MM_set1.'];
% train_Matrix = [full(bow_to_tfidf([PatientVector, sumMGUS_vector, ...
%     sumNotMM_vector, sumProtein_vector, sumSMM_vector])), ...
%     KL_vecotr, MS1_vecotr, MS2_vecotr, PC_vecotr1, PC_vecotr2, Treat_vecotr,MGUS_set1, MM_set1.'];


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

lambda = 0.00001;
% for k = 1: 20
%     id = k:20:size(train_Matrix(train_cohort,:),1);
    [alpha_mgus(:,1)] =HSICLasso(train_Matrix(train_cohort,:).',(true_train_label(train_cohort)+1).',2,lambda);
    [alpha_mm(:,1)] =HSICLasso(train_Matrix(train_cohort,:).',(true_train_labelmm(train_cohort)+1).',2,lambda);
% end
alpha_mgus = full(alpha_mgus(:,1));
alpha_mm = full(alpha_mm(:,1));

alpha_mgus_ind = alpha_mgus~=0;
alpha_mm_ind = alpha_mm~=0;
save('HSIC_param.mat','alpha_mgus','alpha_mm','alpha_mgus_ind','alpha_mm_ind');

%%
train_MatrixN= bsxfun(@rdivide, bsxfun(@minus, train_Matrix(train_cohort,:),...
    mean(train_Matrix(train_cohort,:))), var(train_Matrix(train_cohort,:)) + 1e-10);
test_MatrixN= bsxfun(@rdivide, bsxfun(@minus, train_Matrix(test_cohort,:),...
    mean(train_Matrix(test_cohort,:))), var(train_Matrix(test_cohort,:)) + 1e-10);
valid_MatrixN= bsxfun(@rdivide, bsxfun(@minus, train_Matrix(valid_cohort,:),...
    mean(train_Matrix(valid_cohort,:))), var(train_Matrix(valid_cohort,:)) + 1e-10);
%% SVM - MGUS
opts= struct;
opts.C= 1;
opts.polyOrder= 2;
opts.rbfScale= 0.65;
opts.type = 3;

tic;
modelSVM_MGUS = svmTrain(train_Matrix(train_cohort,alpha_mgus_ind ), true_train_label(train_cohort)+1, opts); % train
timetrain= toc;
yhatMGUS_svm_tr = svmTest(modelSVM_MGUS, train_Matrix(train_cohort,alpha_mgus_ind )) - 1;

yhatMGUS_svm_te = svmTest(modelSVM_MGUS, train_Matrix(test_cohort,alpha_mgus_ind )) - 1;
% yhatMGUS_svm = svmTest(modelSVM_MGUS, train_Matrix) - 1;
[sen_mgus_svm_te,spe_mgus_svm_te,PPV_mgus_svm_te,NPV_mgus_svm_te,Accu_mgus_svm_te,F1_score_mgus_svm_te] =...
    CalcPerformance(yhatMGUS_svm_te,true_train_label(test_cohort))

yhatMGUS_svm_va = svmTest(modelSVM_MGUS, train_Matrix(valid_cohort,alpha_mgus_ind )) - 1;
[sen_mgus_svm_va,spe_mgus_svm_va,PPV_mgus_svm_va,NPV_mgus_svm_va,Accu_mgus_svm_va,F1_score_mgus_svm_va] =...
    CalcPerformance(yhatMGUS_svm_va,true_train_label(valid_cohort))

[XX,YY,TT,AUC] = perfcurve(true_train_label(test_cohort),yhatMGUS_svm_te,1)
% ROC
% ROC
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))
clear TPR_tr FPR_tr  TPR_te TPR_va FPR_va FPR_te prec_tr tpr_tr prec_te tpr_te prec_va tpr_va
% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(train_cohort),yhatMGUS_svm_tr])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(train_cohort);
yy = yhatMGUS_svm_tr;
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
p = auc_bootstrap([true_train_label(test_cohort),yhatMGUS_svm_te])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(test_cohort);
yy = yhatMGUS_svm_te;
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
p = auc_bootstrap([true_train_label(valid_cohort),yhatMGUS_svm_va])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(valid_cohort);
yy = yhatMGUS_svm_va;
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
CI_AUC_tr_svm = CI_fun(A_tr);
CI_PPV_tr_svm = CI_fun(PPV_tr_CI);
CI_SEN_tr_svm = CI_fun(SEN_tr_CI);
CI_ACC_tr_svm= CI_fun(Accu_tr_CI);
CI_F1_tr_svm = CI_fun(F1_score_tr_CI);
CI_tr_SVM = [CI_AUC_tr_svm;CI_PPV_tr_svm;CI_SEN_tr_svm;CI_ACC_tr_svm;CI_F1_tr_svm];

CI_AUC_te_svm = CI_fun(A_te);
CI_PPV_te_svm = CI_fun(PPV_te_CI);
CI_SEN_te_svm = CI_fun(SEN_te_CI);
CI_ACC_te_svm = CI_fun(Accu_te_CI);
CI_F1_te_svm = CI_fun(F1_score_te_CI);
CI_te_SVM = [CI_AUC_te_svm;CI_PPV_te_svm;CI_SEN_te_svm;CI_ACC_te_svm;CI_F1_te_svm];

CI_AUC_va_svm = CI_fun(A_va);
CI_PPV_va_svm = CI_fun(PPV_va_CI);
CI_SEN_va_svm = CI_fun(SEN_va_CI);
CI_ACC_va_svm = CI_fun(Accu_va_CI);
CI_F1_va_svm = CI_fun(F1_score_va_CI);
CI_va_SVM = [CI_AUC_va_svm;CI_PPV_va_svm;CI_SEN_va_svm;CI_ACC_va_svm;CI_F1_va_svm];

save('SVMAUCMGUS_plot.mat','A','Aci',...
    "CI_va_SVM","CI_te_SVM","CI_tr_SVM",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va");



% %% SVM - MM
% clc
% c_list = [0.1:0.1:10];
% r_list = [1:0.1:10];
% best_value = 0;
% best_sen_te = 0; best_PPV_te = 0; 
% best_Accu_te = 0; best_F1_te = 0;
% best_sen_va = 0; best_PPV_va = 0; 
% best_Accu_va = 0; best_F1_va = 0;
% 
% for c = c_list
%     for r = r_list
%         opts= struct;
%         opts.C= c;
%         opts.polyOrder= 2;
%         opts.rbfScale= r;
%         opts.type = 3;
%         
%         tic;
% 
%         modelSVM_MM = svmTrain(train_Matrix(train_cohort,:), true_train_labelmm(train_cohort)+1, opts); % train
%         timetrain= toc
% 
%         yhatMM_svm_te = svmTest(modelSVM_MM, train_Matrix(test_cohort,:)) - 1;
%         [sen_mm_svm_te,spe_mm_svm_te,PPV_mm_svm_te,NPV_mm_svm_te,Accu_mm_svm_te,F1_score_mm_svm_te] =...
%             CalcPerformance(yhatMM_svm_te,true_train_labelmm(test_cohort))
%         yhatMM_svm_va = svmTest(modelSVM_MM, train_Matrix(valid_cohort,:)) - 1;
%         [sen_mm_svm_va,spe_mm_svm_va,PPV_mm_svm_va,NPV_mm_svm_va,Accu_mm_svm_va,F1_score_mm_svm_va] =...
%             CalcPerformance(yhatMM_svm_va,true_train_labelmm(valid_cohort))
%                     this_value_te = sen_mm_svm_te + PPV_mm_svm_te + F1_score_mm_svm_te;
%                     this_value_va = sen_mm_svm_va + PPV_mm_svm_va + F1_score_mm_svm_va;
%         if best_value < this_value_te + this_value_va
%             best_c = c;
%             best_r = r;
%             best_value = this_value_te + this_value_va;
%             best_sen_te = sen_mm_svm_te; best_PPV_te = PPV_mm_svm_te; 
%             best_Accu_te = Accu_mm_svm_te; best_F1_te = F1_score_mm_svm_te;
%             best_sen_va = sen_mm_svm_va; best_PPV_va = PPV_mm_svm_va; 
%             best_Accu_va = Accu_mm_svm_va; best_F1_va = F1_score_mm_svm_va;
%               save('best_param_svm_mm.mat','best_c','best_r') 
%         end
%     end
% end
% 
%%
clc
        opts= struct;
        opts.C=0.5;
        opts.polyOrder= 2;
        opts.rbfScale= 1;
        opts.type = 3;
        tic;
        modelSVM_MM = svmTrain(train_Matrix(train_cohort,alpha_mm_ind), true_train_labelmm(train_cohort)+1, opts); % train
        timetrain= toc
        yhatMM_svm_tr = svmTest(modelSVM_MM, train_Matrix(train_cohort,alpha_mm_ind)) - 1;
        yhatMM_svm_te = svmTest(modelSVM_MM, train_Matrix(test_cohort,alpha_mm_ind)) - 1;
        [sen_mm_svm_te,spe_mm_svm_te,PPV_mm_svm_te,NPV_mm_svm_te,Accu_mm_svm_te,F1_score_mm_svm_te] =...
            CalcPerformance(yhatMM_svm_te,true_train_labelmm(test_cohort))
        yhatMM_svm_va = svmTest(modelSVM_MM, train_Matrix(valid_cohort,alpha_mm_ind)) - 1;
        [sen_mm_svm_va,spe_mm_svm_va,PPV_mm_svm_va,NPV_mm_svm_va,Accu_mm_svm_va,F1_score_mm_svm_va] =...
            CalcPerformance(yhatMM_svm_va,true_train_labelmm(valid_cohort))
[XX,YY,TT,AUC] = perfcurve(true_train_labelmm(test_cohort),yhatMM_svm_te,1)
%% ROC
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))
clear TPR_tr FPR_tr  TPR_te TPR_va FPR_va FPR_te
% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(train_cohort),yhatMM_svm_tr])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(train_cohort);
yy = yhatMM_svm_tr;
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
p = auc_bootstrap([true_train_labelmm(test_cohort),yhatMM_svm_te])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(test_cohort);
yy = yhatMM_svm_te;
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
p = auc_bootstrap([true_train_labelmm(valid_cohort),yhatMM_svm_va])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(valid_cohort);
yy = yhatMM_svm_va;
tic;
for i = 1:nsamp
   ind = randi([1,length(yy)],[nsamp,1]);
   y = yy(ind);
   t = tt(ind);
   [aa,bb] = auc([t,y],alpha,'hanley');
    A_va(i) = aa
    Aci_va(i,:) = bb
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
CI_AUC_tr_svm = CI_fun(A_tr);
CI_PPV_tr_svm = CI_fun(PPV_tr_CI);
CI_SEN_tr_svm = CI_fun(SEN_tr_CI);
CI_ACC_tr_svm= CI_fun(Accu_tr_CI);
CI_F1_tr_svm = CI_fun(F1_score_tr_CI);
CI_tr_SVM = [CI_AUC_tr_svm;CI_PPV_tr_svm;CI_SEN_tr_svm;CI_ACC_tr_svm;CI_F1_tr_svm];

CI_AUC_te_svm = CI_fun(A_te);
CI_PPV_te_svm = CI_fun(PPV_te_CI);
CI_SEN_te_svm = CI_fun(SEN_te_CI);
CI_ACC_te_svm = CI_fun(Accu_te_CI);
CI_F1_te_svm = CI_fun(F1_score_te_CI);
CI_te_SVM = [CI_AUC_te_svm;CI_PPV_te_svm;CI_SEN_te_svm;CI_ACC_te_svm;CI_F1_te_svm];

CI_AUC_va_svm = CI_fun(A_va);
CI_PPV_va_svm = CI_fun(PPV_va_CI);
CI_SEN_va_svm = CI_fun(SEN_va_CI);
CI_ACC_va_svm = CI_fun(Accu_va_CI);
CI_F1_va_svm = CI_fun(F1_score_va_CI);
CI_va_SVM = [CI_AUC_va_svm;CI_PPV_va_svm;CI_SEN_va_svm;CI_ACC_va_svm;CI_F1_va_svm];

save('SVMAUCMM_plot.mat','A','Aci',...
    "CI_va_SVM","CI_te_SVM","CI_tr_SVM",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va"); 
%% Add date +-45 days.... SVM
test_pid = pid(test_cohort);
[earliest_timeMGUS_te,earliest_timeMM_te,...
    yhatMGUS_svm_te,yhatMM_svm_te] = NLP_TP(...
    test_cohort,pid,clinicaldata,Treat_records,...
    ind_MGUS,ind_protein,ind_mspike_mgus,ind_mspike_mm,...
    mspike_date,ind1mspike,ind2mspike,...
    ind_NotMM,ind_MM, ind_SMM, ...
    ind_klratio,ind2klratio,KL_date,ind_PC,PC,PC_date,...
    KL_PatientSSN, MK_PatientSSN,PCPatientSSN,...
    yhatMGUS_svm_te, yhatMM_svm_te, ReportTime);

% % update performance after date
% [sen_mgus_svm_te_date,spe_mgus_svm_te_date,PPV_mgus_svm_te_date,...
% NPV_mgus_svm_te_date,Accu_mgus_svm_te_date,F1_score_mgus_svm_te_date] =...
%     CalcPerformance(yhatMGUS_svm_te,true_train_label(test_cohort));
% [sen_mm_svm_te_date,spe_mm_svm_te_date,PPV_mm_svm_te_date,...
%     NPV_mm_svm_te_date,Accu_mm_svm_te_date,F1_score_mm_svm_te_date] =...
%     CalcPerformance(yhatMM_svm_te,true_train_labelmm(test_cohort));

Days_diff = 180;
Days_limit = Inf;

TP_test_MGUS = zeros(sum(test_cohort),1);
FP_test_MGUS = zeros(sum(test_cohort),1);
TN_test_MGUS = zeros(sum(test_cohort),1);
FN_test_MGUS = zeros(sum(test_cohort),1);

TP_test_MM = zeros(sum(test_cohort),1);
FP_test_MM = zeros(sum(test_cohort),1);
TN_test_MM = zeros(sum(test_cohort),1);
FN_test_MM = zeros(sum(test_cohort),1);

date_diff_mgus = zeros(sum(test_cohort),1);
date_diff_mm = zeros(sum(test_cohort),1);
True_MGUS_date = table2array(TESTtable(:,2));
True_MM_date = table2array(TESTtable(:,5));


for i = 1:sum(test_cohort)
    this_test = find(test_pid(i) == pid);
    
    TP_test_MGUS(i) = ~isempty(True_MGUS_date{this_test}) & ~isempty(earliest_timeMGUS_te{i});
    FP_test_MGUS(i) = isempty(True_MGUS_date{this_test}) & ~isempty(earliest_timeMGUS_te{i});
    TN_test_MGUS(i) = isempty(True_MGUS_date{this_test}) & isempty(earliest_timeMGUS_te{i});
    FN_test_MGUS(i) = ~isempty(True_MGUS_date{this_test}) & isempty(earliest_timeMGUS_te{i});

    TP_test_MM(i) = ~isempty(True_MM_date{this_test}) & ~isempty(earliest_timeMM_te{i});
    FP_test_MM(i) = isempty(True_MM_date{this_test}) & ~isempty(earliest_timeMM_te{i});
    TN_test_MM(i) = isempty(True_MM_date{this_test}) & isempty(earliest_timeMM_te{i});
    FN_test_MM(i) = ~isempty(True_MM_date{this_test}) & isempty(earliest_timeMM_te{i});

    % check for date match
    if TP_test_MGUS(i) ~= 0
    if any(abs(datenum(True_MGUS_date{this_test}) -  datenum(earliest_timeMGUS_te{i}))<=Days_diff)
        % check value for the matched date
        if any(abs(datenum(True_MGUS_date{this_test}) -  datenum(earliest_timeMGUS_te{i}))>Days_limit)
            TP_test_MGUS(i) = 0;
            FP_test_MGUS(i) = 1;
        end
        date_diff_mgus(i) = abs(datenum(True_MGUS_date{this_test}) -  datenum(earliest_timeMGUS_te{i}));
    end 
    end
    
    if TP_test_MM(i) ~= 0
    if any(abs(datenum(True_MM_date{this_test}) -  datenum(earliest_timeMM_te{i}))<=Days_diff)
        % check value for the matched date
        if any(abs(datenum(True_MM_date{this_test}) -  datenum(earliest_timeMM_te{i}))>Days_limit)
            TP_test_MM(i) = 0;
            FP_test_MM(i) = 1;
        end
        date_diff_mm(i) = abs(datenum(True_MM_date{this_test}) -  datenum(earliest_timeMM_te{i}));
    end 
    end
end

% date agree
date_diff_test_mgus = date_diff_mgus(logical(TP_test_MGUS));
date_diff_test_mm = date_diff_mm(logical(TP_test_MM));

dates_agree = [sum(date_diff_test_mgus == 0),100*sum(date_diff_test_mgus == 0)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates7 =[sum(date_diff_test_mgus <= 7), 100*sum(date_diff_test_mgus <= 7)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates30 = [sum(date_diff_test_mgus <= 30),100*sum(date_diff_test_mgus <= 30)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates45 = [sum(date_diff_test_mgus <= 45),100*sum(date_diff_test_mgus <= 45)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates90 = [sum(date_diff_test_mgus <= 90),100*sum(date_diff_test_mgus <= 90)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates180 = [sum(date_diff_test_mgus <= 180),100*sum(date_diff_test_mgus <= 180)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
day_matrix_test_mgus = [dates_agree;dates7;dates30;dates45;dates90;dates180];

dates_agree = [sum(date_diff_test_mm == 0),100*sum(date_diff_test_mm == 0)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates7 =[sum(date_diff_test_mm <= 7), 100*sum(date_diff_test_mm <= 7)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates30 = [sum(date_diff_test_mm <= 30),100*sum(date_diff_test_mm <= 30)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates45 = [sum(date_diff_test_mm <= 45),100*sum(date_diff_test_mm <= 45)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates90 = [sum(date_diff_test_mm <= 90),100*sum(date_diff_test_mm <= 90)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates180 = [sum(date_diff_test_mm <= 180),100*sum(date_diff_test_mm <= 180)/(sum(TP_test_MM) + sum(FP_test_MM))];

day_matrix_test_mm = [dates_agree;dates7;dates30;dates45;dates90;dates180];

% % update performance after date
% sen_mgus_svm_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FN_test_MGUS));
% spe_mgus_svm_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FP_test_MGUS));
% PPV_mgus_svm_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FP_test_MGUS));
% NPV_mgus_svm_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FN_test_MGUS));
% Accu_mgus_svm_te_date = (sum(TP_test_MGUS)+sum(TN_test_MGUS))/(sum(TP_test_MGUS)+sum(TN_test_MGUS)+sum(TP_test_MGUS)+sum(FN_test_MGUS));
% F1_score_mgus_svm_te_date = 2*sum(TP_test_MGUS)/(2*sum(TP_test_MGUS)+sum(FP_test_MGUS)+sum(FN_test_MGUS));
% 
% sen_mm_svm_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FN_test_MM));
% spe_mm_svm_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FP_test_MM));
% PPV_mm_svm_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FP_test_MM));
% NPV_mm_svm_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FN_test_MM));
% Accu_mm_svm_te_date = (sum(TP_test_MM)+sum(TN_test_MM))/(sum(TP_test_MM)+sum(TN_test_MM)+sum(TP_test_MM)+sum(FN_test_MM));
% F1_score_mm_svm_te_date = 2*sum(TP_test_MM)/(2*sum(TP_test_MM)+sum(FP_test_MM)+sum(FN_test_MM));

%% +-45 days svm valid
valid_pid = pid(valid_cohort);
[earliest_timeMGUS_va,earliest_timeMM_va,...
    yhatMGUS_svm_va,yhatMM_svm_va] = NLP_TP(...
    valid_cohort,pid,clinicaldata,Treat_records,...
    ind_MGUS,ind_protein,ind_mspike_mgus,ind_mspike_mm,...
    mspike_date,ind1mspike,ind2mspike,...
    ind_NotMM,ind_MM, ind_SMM, ...
    ind_klratio,ind2klratio,KL_date,ind_PC,PC,PC_date,...
    KL_PatientSSN, MK_PatientSSN,PCPatientSSN,...
    yhatMGUS_svm_va, yhatMM_svm_va, ReportTime);

% update performance after date
[sen_mgus_svm_va_date,spe_mgus_svm_va_date,PPV_mgus_svm_va_date,...
NPV_mgus_svm_va_date,Accu_mgus_svm_va_date,F1_score_mgus_svm_va_date] =...
    CalcPerformance(yhatMGUS_svm_va,true_train_label(valid_cohort));
[sen_mm_svm_va_date,spe_mm_svm_va_date,PPV_mm_svm_va_date,...
    NPV_mm_svm_va_date,Accu_mm_svm_va_date,F1_score_mm_svm_va_date] =...
    CalcPerformance(yhatMM_svm_va,true_train_labelmm(valid_cohort));

Days_diff = 180;
Days_limit = Inf;

TP_valid_MGUS = zeros(sum(valid_cohort),1);
FP_valid_MGUS = zeros(sum(valid_cohort),1);
TN_valid_MGUS = zeros(sum(valid_cohort),1);
FN_valid_MGUS = zeros(sum(valid_cohort),1);

TP_valid_MM = zeros(sum(valid_cohort),1);
FP_valid_MM = zeros(sum(valid_cohort),1);
TN_valid_MM = zeros(sum(valid_cohort),1);
FN_valid_MM = zeros(sum(valid_cohort),1);

date_diff = zeros(sum(valid_cohort),1);
True_MGUS_date = table2array(TESTtable(:,2));
True_MM_date = table2array(TESTtable(:,5));

for i = 1:sum(valid_cohort)
    this_valid = find(valid_pid(i) == pid);
    
    TP_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & ~isempty(earliest_timeMGUS_va{i});
    FP_valid_MGUS(i) = isempty(True_MGUS_date{this_valid}) & ~isempty(earliest_timeMGUS_va{i});
    TN_valid_MGUS(i) = isempty(True_MGUS_date{this_valid}) & isempty(earliest_timeMGUS_va{i});
    FN_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & isempty(earliest_timeMGUS_va{i});

    TP_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & ~isempty(earliest_timeMM_va{i});
    FP_valid_MM(i) = isempty(True_MM_date{this_valid}) & ~isempty(earliest_timeMM_va{i});
    TN_valid_MM(i) = isempty(True_MM_date{this_valid}) & isempty(earliest_timeMM_va{i});
    FN_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & isempty(earliest_timeMM_va{i});
    
    % check for date match
    if TP_valid_MGUS(i) ~= 0
    if any(abs(datenum(True_MGUS_date{this_valid}) -  datenum(earliest_timeMGUS_va{i}))<=Days_diff)
        % check value for the matched date
        if any(abs(datenum(True_MGUS_date{this_valid}) -  datenum(earliest_timeMGUS_va{i}))>Days_limit)
            TP_valid_MGUS(i) = 0;
            FP_valid_MGUS(i) = 1;
        end
        date_diff(i) = abs(datenum(True_MGUS_date{this_valid}) -  datenum(earliest_timeMGUS_va{i}));
    end 
    end
    
    if TP_valid_MM(i) ~= 0
    if any(abs(datenum(True_MM_date(this_valid)) -  datenum(earliest_timeMM_va{i}))<=Days_diff)
        if any(abs(datenum(True_MM_date(this_valid)) -  datenum(earliest_timeMM_va{i}))>Days_limit)
            TP_valid_MM(i) = 0;
            FP_valid_MM(i) = 1;
        end
        date_diff_mm(i) = abs(datenum(True_MM_date(this_valid)) -  datenum(earliest_timeMM_va{i}));
    end 
    end
    
end

date_diff_valid = date_diff(logical(TP_valid_MGUS));

dates_agree = [sum(date_diff_valid == 0),100*sum(date_diff_valid == 0)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates7 =[sum(date_diff_valid <= 7), 100*sum(date_diff_valid <= 7)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates30 = [sum(date_diff_valid <= 30),100*sum(date_diff_valid <= 30)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates45 = [sum(date_diff_valid <= 45),100*sum(date_diff_valid <= 45)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates90 = [sum(date_diff_valid <= 90),100*sum(date_diff_valid <= 90)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates180 = [sum(date_diff_valid <= 180),100*sum(date_diff_valid <= 180)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
day_matrix_valid_mgus = [dates_agree;dates7;dates30;dates45;dates90;dates180];

date_diff_valid = date_diff_mm(logical(TP_valid_MM));
dates_agree = [sum(date_diff_valid == 0),100*sum(date_diff_valid == 0)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates7 =[sum(date_diff_valid <= 7), 100*sum(date_diff_valid <= 7)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates30 = [sum(date_diff_valid <= 30),100*sum(date_diff_valid <= 30)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates45 = [sum(date_diff_valid <= 45),100*sum(date_diff_valid <= 45)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates90 = [sum(date_diff_valid <= 90),100*sum(date_diff_valid <= 90)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates180 = [sum(date_diff_valid <= 180),100*sum(date_diff_valid <= 180)/(sum(TP_valid_MM) + sum(FP_valid_MM))];

day_matrix_valid_mm = [dates_agree;dates7;dates30;dates45;dates90;dates180];

% % update performance after date
sen_mgus_svm_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FN_test_MGUS));
spe_mgus_svm_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FP_test_MGUS));
PPV_mgus_svm_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FP_test_MGUS));
NPV_mgus_svm_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FN_test_MGUS));
Accu_mgus_svm_te_date = (sum(TP_test_MGUS)+sum(TN_test_MGUS))/(sum(TP_test_MGUS)+sum(TN_test_MGUS)+sum(FP_test_MGUS)+sum(FN_test_MGUS));
F1_score_mgus_svm_te_date = 2*sum(TP_test_MGUS)/(2*sum(TP_test_MGUS)+sum(FP_test_MGUS)+sum(FN_test_MGUS));

sen_mm_svm_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FN_test_MM));
spe_mm_svm_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FP_test_MM));
PPV_mm_svm_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FP_test_MM));
NPV_mm_svm_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FN_test_MM));
Accu_mm_svm_te_date = (sum(TP_test_MM)+sum(TN_test_MM))/(sum(TP_test_MM)+sum(TN_test_MM)+sum(FP_test_MM)+sum(FN_test_MM));
F1_score_mm_svm_te_date = 2*sum(TP_test_MM)/(2*sum(TP_test_MM)+sum(FP_test_MM)+sum(FN_test_MM));

% update performance after date
sen_mgus_svm_va_date = sum(TP_valid_MGUS)/(sum(TP_valid_MGUS)+sum(FN_valid_MGUS));
spe_mgus_svm_va_date = sum(TN_valid_MGUS)/(sum(TN_valid_MGUS)+sum(FP_valid_MGUS));
PPV_mgus_svm_va_date = sum(TP_valid_MGUS)/(sum(TP_valid_MGUS)+sum(FP_valid_MGUS));
NPV_mgus_svm_va_date = sum(TN_valid_MGUS)/(sum(TN_valid_MGUS)+sum(FN_valid_MGUS));
Accu_mgus_svm_va_date = (sum(TP_valid_MGUS)+sum(TN_valid_MGUS))/(sum(TP_valid_MGUS)+sum(TN_valid_MGUS)+sum(FP_valid_MGUS)+sum(FN_valid_MGUS));
F1_score_mgus_svm_va_date = 2*sum(TP_valid_MGUS)/(2*sum(TP_valid_MGUS)+sum(FP_valid_MGUS)+sum(FN_valid_MGUS));

sen_mm_svm_va_date = sum(TP_valid_MM)/(sum(TP_valid_MM)+sum(FN_valid_MM));
spe_mm_svm_va_date = sum(TN_valid_MM)/(sum(TN_valid_MM)+sum(FP_valid_MM));
PPV_mm_svm_va_date = sum(TP_valid_MM)/(sum(TP_valid_MM)+sum(FP_valid_MM));
NPV_mm_svm_va_date = sum(TN_valid_MM)/(sum(TN_valid_MM)+sum(FN_valid_MM));
Accu_mm_svm_va_date = (sum(TP_valid_MM)+sum(TN_valid_MM))/(sum(TP_valid_MM)+sum(TN_valid_MM)+sum(FP_valid_MM)+sum(FN_valid_MM));
F1_score_mm_svm_va_date = 2*sum(TP_valid_MM)/(2*sum(TP_valid_MM)+sum(FP_valid_MM)+sum(FN_valid_MM));
%% SVM-MGUS column
SVM_MGUS_va = [PPV_mgus_svm_va_date;sen_mgus_svm_va_date;F1_score_mgus_svm_va_date;Accu_mgus_svm_va_date;...
    day_matrix_valid_mgus(:,2)];
SVM_MGUS_te = [PPV_mgus_svm_te_date;sen_mgus_svm_te_date;F1_score_mgus_svm_te_date;Accu_mgus_svm_te_date;...
    day_matrix_test_mgus(:,2)];
MGUS_SVM = [SVM_MGUS_va;SVM_MGUS_te];

SVM_MM_va = [PPV_mm_svm_va_date;sen_mm_svm_va_date;F1_score_mm_svm_va_date;Accu_mm_svm_va_date;...
    day_matrix_valid_mm(:,2)];
SVM_MM_te = [PPV_mm_svm_te_date;sen_mm_svm_te_date;F1_score_mm_svm_te_date;Accu_mm_svm_te_date;...
    day_matrix_test_mm(:,2)];
MM_SVM = [SVM_MM_va;SVM_MM_te];
% writetable(array2table(MGUS_SVM),'Ressults20230111.xlsx','Sheet','MGUS_SVM')
% writetable(array2table(MM_SVM),'Ressults20230111.xlsx','Sheet','MM_SVM')
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Random forest - MGUS
optsRF= struct;
optsRF.depth= 10;
optsRF.numTrees = 500;
optsRF.numSplits= 50;
optsRF.verbose= true;
optsRF.classifierID = [1,2]; % weak learners to use. Can be an array for mix of weak learners too

tic
modelRF_MGUS= forestTrain(train_MatrixN(:,alpha_mgus_ind), true_train_label(train_cohort)+1, optsRF); % train
timetrain= toc;
yhatMGUS_rf_tr = forestTest(modelRF_MGUS, train_MatrixN(:,alpha_mgus_ind)) - 1;
yhatMGUS_rf_te = forestTest(modelRF_MGUS, test_MatrixN(:,alpha_mgus_ind)) - 1;
% yhatMGUS_svm = svmTest(modelSVM_MGUS, train_Matrix) - 1;
[sen_mgus_rf_te,spe_mgus_rf_te,PPV_mgus_rf_te,NPV_mgus_rf_te,Accu_mgus_rf_te,F1_score_mgus_rf_te] =...
    CalcPerformance(yhatMGUS_rf_te,true_train_label(test_cohort))
[XX,YY,TT,AUC] = perfcurve(true_train_label(test_cohort),yhatMGUS_rf_te,1)
yhatMGUS_rf_va = forestTest(modelRF_MGUS, valid_MatrixN(:,alpha_mgus_ind)) - 1;
[sen_mgus_rf_va,spe_mgus_rf_va,PPV_mgus_rf_va,NPV_mgus_rf_va,Accu_mgus_rf_va,F1_score_mgus_rf_va] =...
    CalcPerformance(yhatMGUS_rf_va,true_train_label(valid_cohort))
%% ROC
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))
clear TPR_tr FPR_tr TPR_te FPR_te TPR_va FPR_va prec_te tpr_te prec_tr tpr_tr prec_va tpr_va

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(train_cohort),yhatMGUS_rf_tr])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(train_cohort);
yy = yhatMGUS_rf_tr;
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
p = auc_bootstrap([true_train_label(test_cohort),yhatMGUS_rf_te])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(test_cohort);
yy = yhatMGUS_rf_te;
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
p = auc_bootstrap([true_train_label(valid_cohort),yhatMGUS_rf_va])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(valid_cohort);
yy = yhatMGUS_rf_va;
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
CI_AUC_tr_rf = CI_fun(A_tr);
CI_PPV_tr_rf = CI_fun(PPV_tr_CI);
CI_SEN_tr_rf = CI_fun(SEN_tr_CI);
CI_ACC_tr_rf= CI_fun(Accu_tr_CI);
CI_F1_tr_rf = CI_fun(F1_score_tr_CI);
CI_tr_RF = [CI_AUC_tr_rf;CI_PPV_tr_rf;CI_SEN_tr_rf;CI_ACC_tr_rf;CI_F1_tr_rf];

CI_AUC_te_rf = CI_fun(A_te);
CI_PPV_te_rf = CI_fun(PPV_te_CI);
CI_SEN_te_rf = CI_fun(SEN_te_CI);
CI_ACC_te_rf = CI_fun(Accu_te_CI);
CI_F1_te_rf = CI_fun(F1_score_te_CI);
CI_te_RF = [CI_AUC_te_rf;CI_PPV_te_rf;CI_SEN_te_rf;CI_ACC_te_rf;CI_F1_te_rf];

CI_AUC_va_rf = CI_fun(A_va);
CI_PPV_va_rf = CI_fun(PPV_va_CI);
CI_SEN_va_rf = CI_fun(SEN_va_CI);
CI_ACC_va_rf = CI_fun(Accu_va_CI);
CI_F1_va_rf = CI_fun(F1_score_va_CI);
CI_va_RF = [CI_AUC_va_rf;CI_PPV_va_rf;CI_SEN_va_rf;CI_ACC_va_rf;CI_F1_va_rf];

save('RFAUCMGUS_plot.mat','A','Aci',...
    "CI_va_RF","CI_te_RF","CI_tr_RF",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va"); 

%% RF- MM
% clc
% d_list = [1:2:15];
% nT_list = 10:10:200;
% nS_list = 5:5:50;
% % 15, 180, 25
% best_val = 0;
% for d = d_list
%     for nT = nT_list
%         for nS = nS_list
% optsRF= struct;
% optsRF.depth= d;
% optsRF.numTrees = nT;
% optsRF.numSplits= nS;
% optsRF.verbose= true;
% optsRF.classifierID = [1,2]; % weak learners to use. Can be an array for mix of weak learners too
%  
% tic;
% modelRF_MM = forestTrain(train_MatrixN, true_train_labelmm(train_cohort)+1, optsRF); % train
% timetrain= toc;
% 
% yhatMM_rf_te = forestTest(modelRF_MM, test_MatrixN) - 1;
% [sen_mm_rf_te,spe_mm_rf_te,PPV_mm_rf_te,NPV_mm_rf_te,Accu_mm_rf_te,F1_score_mm_rf_te] =...
%     CalcPerformance(yhatMM_rf_te,true_train_labelmm(test_cohort))
% yhatMM_rf_va = forestTest(modelRF_MM, valid_MatrixN) - 1;
% [sen_mm_rf_va,spe_mm_rf_va,PPV_mm_rf_va,NPV_mm_rf_va,Accu_mm_rf_va,F1_score_mm_rf_va] =...
%     CalcPerformance(yhatMM_rf_va,true_train_labelmm(valid_cohort))
% 
%             if best_val < sen_mm_rf_te + PPV_mm_rf_te + F1_score_mm_rf_te + ...
%                     sen_mm_rf_va + PPV_mm_rf_va + F1_score_mm_rf_va;
%                 best_val = sen_mm_rf_te + PPV_mm_rf_te + F1_score_mm_rf_te + ...
%                     sen_mm_rf_va + PPV_mm_rf_va + F1_score_mm_rf_va;
%                 best_d = d;
%                 best_nT = nT;
%                 best_nS = nS;
%                 save('best_rf_mm_param.mat','best_d','best_nT','best_nS')
%             end
%         end
%     end
% end

%%
% best 13 35, 20
optsRF= struct;
optsRF.depth= 10;
optsRF.numTrees = 200;
optsRF.numSplits= 20; % 2000
optsRF.verbose= true;
optsRF.classifierID = [1 2]; % weak learners to use. Can be an array for mix of weak learners too
 
tic;
modelRF_MM = forestTrain(train_Matrix(train_cohort,alpha_mm_ind), true_train_labelmm(train_cohort)+1, optsRF); % train
timetrain= toc;
yhatMM_rf_tr = forestTest(modelRF_MM, train_Matrix(train_cohort,alpha_mm_ind)) - 1;
yhatMM_rf_te = forestTest(modelRF_MM, train_Matrix(test_cohort,alpha_mm_ind)) - 1;
[sen_mm_rf_te,spe_mm_rf_te,PPV_mm_rf_te,NPV_mm_rf_te,Accu_mm_rf_te,F1_score_mm_rf_te] =...
    CalcPerformance(yhatMM_rf_te,true_train_labelmm(test_cohort))
yhatMM_rf_va = forestTest(modelRF_MM, train_Matrix(valid_cohort,alpha_mm_ind)) - 1;
[sen_mm_rf_va,spe_mm_rf_va,PPV_mm_rf_va,NPV_mm_rf_va,Accu_mm_rf_va,F1_score_mm_rf_va] =...
    CalcPerformance(yhatMM_rf_va,true_train_labelmm(valid_cohort))

%% ROC
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))
clear TPR_tr FPR_tr TPR_te FPR_te TPR_va FPR_va
% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(train_cohort),yhatMM_rf_tr])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(train_cohort);
yy = yhatMM_rf_tr;
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
p = auc_bootstrap([true_train_labelmm(test_cohort),yhatMM_rf_te])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(test_cohort);
yy = yhatMM_rf_te;
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
p = auc_bootstrap([true_train_labelmm(valid_cohort),yhatMM_rf_va])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(valid_cohort);
yy = yhatMM_rf_va;
tic;
for i = 1:nsamp
   ind = randi([1,length(yy)],[nsamp,1]);
   y = yy(ind);
   t = tt(ind);
   [A_va(i),Aci_va(i,:)] = auc([t,y],alpha,'hanley');
   [aa,bb] = roc([y,t]);
   TPR_va(:,i) = aa;
   FPR_va(:,i) = bb;
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
CI_AUC_tr_rf = CI_fun(A_tr);
CI_PPV_tr_rf = CI_fun(PPV_tr_CI);
CI_SEN_tr_rf = CI_fun(SEN_tr_CI);
CI_ACC_tr_rf= CI_fun(Accu_tr_CI);
CI_F1_tr_rf = CI_fun(F1_score_tr_CI);
CI_tr_RF = [CI_AUC_tr_rf;CI_PPV_tr_rf;CI_SEN_tr_rf;CI_ACC_tr_rf;CI_F1_tr_rf];

CI_AUC_te_rf = CI_fun(A_te);
CI_PPV_te_rf = CI_fun(PPV_te_CI);
CI_SEN_te_rf = CI_fun(SEN_te_CI);
CI_ACC_te_rf = CI_fun(Accu_te_CI);
CI_F1_te_rf = CI_fun(F1_score_te_CI);
CI_te_RF = [CI_AUC_te_rf;CI_PPV_te_rf;CI_SEN_te_rf;CI_ACC_te_rf;CI_F1_te_rf];

CI_AUC_va_rf = CI_fun(A_va);
CI_PPV_va_rf = CI_fun(PPV_va_CI);
CI_SEN_va_rf = CI_fun(SEN_va_CI);
CI_ACC_va_rf = CI_fun(Accu_va_CI);
CI_F1_va_rf = CI_fun(F1_score_va_CI);
CI_va_RF = [CI_AUC_va_rf;CI_PPV_va_rf;CI_SEN_va_rf;CI_ACC_va_rf;CI_F1_va_rf];

save('RFAUCMM_plot.mat','A','Aci',...
    "CI_va_RF","CI_te_RF","CI_tr_RF",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va"); 


%% Add date +-45 days.... RF
test_pid = pid(test_cohort);
[earliest_timeMGUS_te,earliest_timeMM_te,...
    yhatMGUS_rf_te,yhatMM_rf_te] = NLP_TP(...
    test_cohort,pid,clinicaldata,Treat_records,...
    ind_MGUS,ind_protein,ind_mspike_mgus,ind_mspike_mm,...
    mspike_date,ind1mspike,ind2mspike,...
    ind_NotMM,ind_MM, ind_SMM, ...
    ind_klratio,ind2klratio,KL_date,ind_PC,PC,PC_date,...
    KL_PatientSSN, MK_PatientSSN,PCPatientSSN,...
    yhatMGUS_rf_te, yhatMM_rf_te, ReportTime);

% update performance after date
[sen_mgus_rf_te_date,spe_mgus_rf_te_date,PPV_mgus_rf_te_date,...
NPV_mgus_rf_te_date,Accu_mgus_rf_te_date,F1_score_mgus_rf_te_date] =...
    CalcPerformance(yhatMGUS_rf_te,true_train_label(test_cohort))
[sen_mm_rf_te_date,spe_mm_rf_te_date,PPV_mm_rf_te_date,...
    NPV_mm_rf_te_date,Accu_mm_rf_te_date,F1_score_mm_rf_te_date] =...
    CalcPerformance(yhatMM_rf_te,true_train_labelmm(test_cohort))


Days_diff = 180;

TP_test_MGUS = zeros(sum(test_cohort),1);
FP_test_MGUS = zeros(sum(test_cohort),1);
TN_test_MGUS = zeros(sum(test_cohort),1);
FN_test_MGUS = zeros(sum(test_cohort),1);

TP_test_MM = zeros(sum(test_cohort),1);
FP_test_MM = zeros(sum(test_cohort),1);
TN_test_MM = zeros(sum(test_cohort),1);
FN_test_MM = zeros(sum(test_cohort),1);

date_diff = zeros(sum(test_cohort),1);
date_diff_mm = zeros(sum(test_cohort),1);
True_MGUS_date = table2array(TESTtable(:,2));
True_MM_date = table2array(TESTtable(:,5));

for i = 1:sum(test_cohort)
    this_test = find(test_pid(i) == pid);
    
    TP_test_MGUS(i) = ~isempty(True_MGUS_date{this_test}) & ~isempty(earliest_timeMGUS_te{i});
    FP_test_MGUS(i) = isempty(True_MGUS_date{this_test}) & ~isempty(earliest_timeMGUS_te(i));
    TN_test_MGUS(i) = isempty(True_MGUS_date{this_test}) & isempty(earliest_timeMGUS_te(i));
    FN_test_MGUS(i) = ~isempty(True_MGUS_date{this_test}) & isempty(earliest_timeMGUS_te(i));

    TP_test_MM(i) = ~isempty(True_MM_date{this_test}) & ~isempty(earliest_timeMM_te{i});
    FP_test_MM(i) = isempty(True_MM_date{this_test}) & ~isempty(earliest_timeMM_te{i});
    TN_test_MM(i) = isempty(True_MM_date{this_test}) & isempty(earliest_timeMM_te{i});
    FN_test_MM(i) = ~isempty(True_MM_date{this_test}) & isempty(earliest_timeMM_te{i});


    % check for date match
    if TP_test_MGUS(i) ~= 0
    if any(abs(datenum(True_MGUS_date(this_test)) -  datenum(earliest_timeMGUS_te(i)))<=Days_diff)
        if any(abs(datenum(True_MGUS_date{this_test}) -  datenum(earliest_timeMGUS_te{i}))>Days_limit)
            TP_test_MGUS(i) = 0;
            FP_test_MGUS(i) = 1;
        end
        % check value for the matched date
        date_diff(i) = abs(datenum(True_MGUS_date(this_test)) -  datenum(earliest_timeMGUS_te(i)));
    end 
    end
    
    if TP_test_MM(i) ~= 0
    if any(abs(datenum(True_MM_date(this_test)) -  datenum(earliest_timeMM_te{i}))<=Days_diff)
        if any(abs(datenum(True_MM_date{this_test}) -  datenum(earliest_timeMM_te{i}))>Days_limit)
            TP_test_MM(i) = 0;
            FP_test_MM(i) = 1;
        end
        date_diff_mm(i) = abs(datenum(True_MM_date(this_test)) -  datenum(earliest_timeMM_te{i}));
    end 
    end
end

% date agree
date_diff_test = date_diff(logical(TP_test_MGUS));

dates_agree = [sum(date_diff_test == 0),100*sum(date_diff_test == 0)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates7 =[sum(date_diff_test <= 7), 100*sum(date_diff_test <= 7)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates30 = [sum(date_diff_test <= 30),100*sum(date_diff_test <= 30)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates45 = [sum(date_diff_test <= 45),100*sum(date_diff_test <= 45)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates90 = [sum(date_diff_test <= 90),100*sum(date_diff_test <= 90)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates180 = [sum(date_diff_test <= 180),100*sum(date_diff_test <= 180)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
day_matrix_test_mgus = [dates_agree;dates7;dates30;dates45;dates90;dates180];

% date agree
date_diff_test = date_diff_mm(logical(TP_test_MM));

dates_agree = [sum(date_diff_test == 0),100*sum(date_diff_test == 0)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates7 =[sum(date_diff_test <= 7), 100*sum(date_diff_test <= 7)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates30 = [sum(date_diff_test <= 30),100*sum(date_diff_test <= 30)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates45 = [sum(date_diff_test <= 45),100*sum(date_diff_test <= 45)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates90 = [sum(date_diff_test <= 90),100*sum(date_diff_test <= 90)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates180 = [sum(date_diff_test <= 180),100*sum(date_diff_test <= 180)/(sum(TP_test_MM) + sum(FP_test_MM))];
day_matrix_test_mm = [dates_agree;dates7;dates30;dates45;dates90;dates180];


%% +-45 days RF valid
valid_pid = pid(valid_cohort);
[earliest_timeMGUS_va,earliest_timeMM_va,...
    yhatMGUS_rf_va,yhatMM_rf_va] = NLP_TP(...
    valid_cohort,pid,clinicaldata,Treat_records,...
    ind_MGUS,ind_protein,ind_mspike_mgus,ind_mspike_mm,...
    mspike_date,ind1mspike,ind2mspike,...
    ind_NotMM,ind_MM, ind_SMM, ...
    ind_klratio,ind2klratio,KL_date,ind_PC,PC,PC_date,...
    KL_PatientSSN, MK_PatientSSN,PCPatientSSN,...
    yhatMGUS_rf_va, yhatMM_rf_va, ReportTime);

% update performance after date
[sen_mgus_rf_va_date,spe_mgus_rf_va_date,PPV_mgus_rf_va_date,...
NPV_mgus_rf_va_date,Accu_mgus_rf_va_date,F1_score_mgus_rf_a_date] =...
    CalcPerformance(yhatMGUS_rf_va,true_train_label(valid_cohort))
[sen_mm_rf_va_date,spe_mm_rf_va_date,PPV_mm_rf_va_date,...
    NPV_mm_rf_va_date,Accu_mm_rf_va_date,F1_score_mm_rf_va_date] =...
    CalcPerformance(yhatMM_rf_va,true_train_labelmm(valid_cohort))


Days_diff = 180;

TP_valid_MGUS = zeros(sum(valid_cohort),1);
FP_valid_MGUS = zeros(sum(valid_cohort),1);
TN_valid_MGUS = zeros(sum(valid_cohort),1);
FN_valid_MGUS = zeros(sum(valid_cohort),1);

TP_valid_MM = zeros(sum(valid_cohort),1);
FP_valid_MM = zeros(sum(valid_cohort),1);
TN_valid_MM = zeros(sum(valid_cohort),1);
FN_valid_MM = zeros(sum(valid_cohort),1);

date_diff_mgus = zeros(sum(valid_cohort),1);
date_diff_mm = zeros(sum(valid_cohort),1);
True_MGUS_date = table2array(TESTtable(:,2));
True_MM_date = table2array(TESTtable(:,5));


for i = 1:sum(valid_cohort)
    this_valid = find(valid_pid(i) == pid);
    
    TP_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & ~isempty(earliest_timeMGUS_va{i});
    FP_valid_MGUS(i) = isempty(True_MGUS_date{this_valid}) & ~isempty(earliest_timeMGUS_va{i});
    TN_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & ~isempty(earliest_timeMGUS_va{i});
    FN_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & isempty(earliest_timeMGUS_va{i});

    TP_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & ~isempty(earliest_timeMM_va{i});
    FP_valid_MM(i) = isempty(True_MM_date{this_valid}) & ~isempty(earliest_timeMM_va{i});
    TN_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & ~isempty(earliest_timeMM_va{i});
    FN_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & isempty(earliest_timeMM_va{i});

    % check for date match
    if TP_valid_MGUS(i) ~= 0
    if any(abs(datenum(True_MGUS_date(this_valid)) -  datenum(earliest_timeMGUS_va{i}))<=Days_diff)
        % check value for the matched date
        if any(abs(datenum(True_MGUS_date{this_valid}) -  datenum(earliest_timeMGUS_va{i}))>Days_limit)
            TP_valid_MGUS(i) = 0;
            FP_valid_MGUS(i) = 1;
        end
        date_diff_mgus(i) = abs(datenum(True_MGUS_date(this_valid)) -  datenum(earliest_timeMGUS_va{i}));
    end 
    end
    
    if TP_valid_MM(i) ~= 0
    if any(abs(datenum(True_MM_date(this_valid)) -  datenum(earliest_timeMM_va{i}))<=Days_diff)
        if any(abs(datenum(True_MM_date{this_valid}) -  datenum(earliest_timeMM_va{i}))>Days_limit)
            TP_valid_MM(i) = 0;
            FP_valid_MM(i) = 1;
        end
        date_diff_mm(i) = abs(datenum(True_MM_date(this_valid)) -  datenum(earliest_timeMM_va{i}));
    end 
    end
end

date_diff_valid = date_diff_mgus(logical(TP_valid_MGUS));

dates_agree = [sum(date_diff_valid == 0),100*sum(date_diff_valid == 0)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates7 =[sum(date_diff_valid <= 7), 100*sum(date_diff_valid <= 7)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates30 = [sum(date_diff_valid <= 30),100*sum(date_diff_valid <= 30)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates45 = [sum(date_diff_valid <= 45),100*sum(date_diff_valid <= 45)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates90 = [sum(date_diff_valid <= 90),100*sum(date_diff_valid <= 90)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates180 = [sum(date_diff_valid <= 180),100*sum(date_diff_valid <= 180)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];

day_matrix_valid_mgus = [dates_agree;dates7;dates30;dates45;dates90;dates180];

date_diff_valid = date_diff_mm(logical(TP_valid_MM));

dates_agree = [sum(date_diff_valid == 0),100*sum(date_diff_valid == 0)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates7 =[sum(date_diff_valid <= 7), 100*sum(date_diff_valid <= 7)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates30 = [sum(date_diff_valid <= 30),100*sum(date_diff_valid <= 30)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates45 = [sum(date_diff_valid <= 45),100*sum(date_diff_valid <= 45)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates90 = [sum(date_diff_valid <= 90),100*sum(date_diff_valid <= 90)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates180 = [sum(date_diff_valid <= 180),100*sum(date_diff_valid <= 180)/(sum(TP_valid_MM) + sum(FP_valid_MM))];

day_matrix_valid_mm = [dates_agree;dates7;dates30;dates45;dates90;dates180];

% % update performance after date
sen_mgus_rf_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FN_test_MGUS));
spe_mgus_rf_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FP_test_MGUS));
PPV_mgus_rf_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FP_test_MGUS));
NPV_mgus_rf_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FN_test_MGUS));
Accu_mgus_rf_te_date = (sum(TP_test_MGUS)+sum(TN_test_MGUS))/(sum(TP_test_MGUS)+sum(TN_test_MGUS)+sum(FP_test_MGUS)+sum(FN_test_MGUS));
F1_score_mgus_rf_te_date = 2*sum(TP_test_MGUS)/(2*sum(TP_test_MGUS)+sum(FP_test_MGUS)+sum(FN_test_MGUS));

sen_mm_rf_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FN_test_MM));
spe_mm_rf_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FP_test_MM));
PPV_mm_rf_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FP_test_MM));
NPV_mm_rf_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FN_test_MM));
Accu_mm_rf_te_date = (sum(TP_test_MM)+sum(TN_test_MM))/(sum(TP_test_MM)+sum(TN_test_MM)+sum(FP_test_MM)+sum(FN_test_MM));
F1_score_mm_rf_te_date = 2*sum(TP_test_MM)/(2*sum(TP_test_MM)+sum(FP_test_MM)+sum(FN_test_MM));

% update performance after date
sen_mgus_rf_va_date = sum(TP_valid_MGUS)/(sum(TP_valid_MGUS)+sum(FN_valid_MGUS));
spe_mgus_rf_va_date = sum(TN_valid_MGUS)/(sum(TN_valid_MGUS)+sum(FP_valid_MGUS));
PPV_mgus_rf_va_date = sum(TP_valid_MGUS)/(sum(TP_valid_MGUS)+sum(FP_valid_MGUS));
NPV_mgus_rf_va_date = sum(TN_valid_MGUS)/(sum(TN_valid_MGUS)+sum(FN_valid_MGUS));
Accu_mgus_rf_va_date = (sum(TP_valid_MGUS)+sum(TN_valid_MGUS))/(sum(TP_valid_MGUS)+sum(TN_valid_MGUS)+sum(FP_valid_MGUS)+sum(FN_valid_MGUS));
F1_score_mgus_rf_va_date = 2*sum(TP_valid_MGUS)/(2*sum(TP_valid_MGUS)+sum(FP_valid_MGUS)+sum(FN_valid_MGUS));

sen_mm_rf_va_date = sum(TP_valid_MM)/(sum(TP_valid_MM)+sum(FN_valid_MM));
spe_mm_rf_va_date = sum(TN_valid_MM)/(sum(TN_valid_MM)+sum(FP_valid_MM));
PPV_mm_rf_va_date = sum(TP_valid_MM)/(sum(TP_valid_MM)+sum(FP_valid_MM));
NPV_mm_rf_va_date = sum(TN_valid_MM)/(sum(TN_valid_MM)+sum(FN_valid_MM));
Accu_mm_rf_va_date = (sum(TP_valid_MM)+sum(TN_valid_MM))/(sum(TP_valid_MM)+sum(TN_valid_MM)+sum(FP_valid_MM)+sum(FN_valid_MM));
F1_score_mm_rf_va_date = 2*sum(TP_valid_MM)/(2*sum(TP_valid_MM)+sum(FP_valid_MM)+sum(FN_valid_MM));

%% RF-MGUS column
RF_MGUS_va = [PPV_mgus_rf_va_date;sen_mgus_rf_va_date;F1_score_mgus_rf_va_date;Accu_mgus_rf_va_date;...
    day_matrix_valid_mgus(:,2)];
RF_MGUS_te = [PPV_mgus_rf_te_date;sen_mgus_rf_te_date;F1_score_mgus_rf_te_date;Accu_mgus_rf_te_date;...
    day_matrix_test_mgus(:,2)];
RF_MM_va = [PPV_mm_rf_va_date;sen_mm_rf_va_date;F1_score_mm_rf_va_date;Accu_mm_rf_va_date;...
    day_matrix_valid_mm(:,2)];
RF_MM_te = [PPV_mm_rf_te_date;sen_mm_rf_te_date;F1_score_mm_rf_te_date;Accu_mm_rf_te_date;...
    day_matrix_test_mm(:,2)];

MGUS_RF = [RF_MGUS_va;RF_MGUS_te];
MM_RF = [RF_MM_va;RF_MM_te];
% writetable(array2table(MGUS_RF),'Ressults20230111.xlsx','Sheet','MGUS_RF')
% writetable(array2table(MM_RF),'Ressults20230111.xlsx','Sheet','MM_RF')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Logstic Regression
rng(1,'twister')
% Logistic regression
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\logistic'))
%compute cost and gradient
iter = 10000; % No. of iterations for weight updation
theta=rand(size(train_Matrix(:,alpha_mgus_ind),2),1); % Initial weights
al=0.1
tic
[J grad h thMGUS]=cost(theta,train_Matrix(train_cohort,alpha_mgus_ind),true_train_label(train_cohort),al,iter); % Cost funtion
toc

iter = 10000; % No. of iterations for weight updation
theta=rand(size(train_Matrix(:,alpha_mm_ind),2),1); % Initial weights
tic
[J grad h thMM]=cost(theta,train_Matrix(train_cohort,alpha_mm_ind),true_train_labelmm(train_cohort),al,iter); % Cost funtion
toc
yhatMGUS_lg_tr=train_Matrix(train_cohort,alpha_mgus_ind)*thMGUS;
yhatMGUS_lg_te=train_Matrix(test_cohort,alpha_mgus_ind)*thMGUS; %target prediction
% probability calculation
[hp]=sigmoid(yhatMGUS_lg_te); % Hypothesis Function
yhatMGUS_lg_te(hp>=0.5)=1;
yhatMGUS_lg_te(hp<0.5)=0;

[sens_mgus_lr_te,spe_mgus_lr_te,PPV_mgus_lr_te,NPV_mgus_lr_te,...
    Accuracy_mgus_lr_te,F1_score_mgus_lr_te] =...
    CalcPerformance(yhatMGUS_lg_te,true_train_label(test_cohort))
yhatMM_lg_tr=train_Matrix(train_cohort,alpha_mm_ind)*thMM; %target prediction
yhatMM_lg_te=train_Matrix(test_cohort,alpha_mm_ind)*thMM; %target prediction
% probability calculation
[hp]=sigmoid(yhatMM_lg_te); % Hypothesis Function
yhatMM_lg_te(hp>=0.5)=1;
yhatMM_lg_te(hp<0.5)=0;

[sens_mm_lr_te,spe_mm_lr_te,PPV_mm_lr_te,NPV_mm_lr_te,...
    Accuracy_mm_lr_te,F1_score_mm_lr_te] =...
    CalcPerformance(yhatMM_lg_te,true_train_labelmm(test_cohort))


yhatMGUS_lg_va=train_Matrix(valid_cohort,alpha_mgus_ind)*thMGUS; %target prediction
% probability calculation
[hp]=sigmoid(yhatMGUS_lg_va); % Hypothesis Function
yhatMGUS_lg_va(hp>=0.5)=1;
yhatMGUS_lg_va(hp<0.5)=0;

[sens_mgus_lr_va,spe_mgus_lr_va,PPV_mgus_lr_va,NPV_mgus_lr_va,...
    Accuracy_mgus_lr_va,F1_score_mgus_lr_va] =...
    CalcPerformance(yhatMGUS_lg_va,true_train_label(valid_cohort))

yhatMM_lg_va=train_Matrix(valid_cohort,alpha_mm_ind)*thMM; %target prediction
% probability calculation
[hp]=sigmoid(yhatMM_lg_va); % Hypothesis Function
yhatMM_lg_va(hp>=0.5)=1;
yhatMM_lg_va(hp<0.5)=0;

[sens_mm_lr_va,spe_mm_lr_va,PPV_mm_lr_va,NPV_mm_lr_va,...
    Accuracy_mm_lr_va,F1_score_mm_lr_va] =...
    CalcPerformance(yhatMM_lg_va,true_train_labelmm(valid_cohort))

%% ROC
% ROC

for rng_num = 50:50
    rng(rng_num,'twister')
    try
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))
clear TPR_tr FPR_tr TPR_te FPR_te TPR_va FPR_va prec_tr tpr_tr prec_te tpr_te prec_va tpr_va
% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(train_cohort),yhatMGUS_lg_tr])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(train_cohort);
yy = yhatMGUS_lg_tr;
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
   [prec_tr{i}, tpr_tr{i}, ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end    

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(test_cohort),yhatMGUS_lg_te])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(test_cohort);
yy = yhatMGUS_lg_te;
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
   [prec_te{i}, tpr_te{i}, ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end 

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_label(valid_cohort),yhatMGUS_lg_va])
alpha = 0.05;
nsamp = 50;
tt = true_train_label(valid_cohort);
yy = yhatMGUS_lg_va;
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
CI_AUC_tr_lg = CI_fun(A_tr);
CI_PPV_tr_lg = CI_fun(PPV_tr_CI);
CI_SEN_tr_lg = CI_fun(SEN_tr_CI);
CI_ACC_tr_lg= CI_fun(Accu_tr_CI);
CI_F1_tr_lg = CI_fun(F1_score_tr_CI);
CI_tr_LG = [CI_AUC_tr_lg;CI_PPV_tr_lg;CI_SEN_tr_lg;CI_ACC_tr_lg;CI_F1_tr_lg];

CI_AUC_te_lg = CI_fun(A_te);
CI_PPV_te_lg = CI_fun(PPV_te_CI);
CI_SEN_te_lg = CI_fun(SEN_te_CI);
CI_ACC_te_lg = CI_fun(Accu_te_CI);
CI_F1_te_lg = CI_fun(F1_score_te_CI);
CI_te_LG = [CI_AUC_te_lg;CI_PPV_te_lg;CI_SEN_te_lg;CI_ACC_te_lg;CI_F1_te_lg];

CI_AUC_va_lg = CI_fun(A_va);
CI_PPV_va_lg = CI_fun(PPV_va_CI);
CI_SEN_va_lg = CI_fun(SEN_va_CI);
CI_ACC_va_lg = CI_fun(Accu_va_CI);
CI_F1_va_lg = CI_fun(F1_score_va_CI);
CI_va_LG = [CI_AUC_va_lg;CI_PPV_va_lg;CI_SEN_va_lg;CI_ACC_va_lg;CI_F1_va_lg];

save('LGAUCMGUS_plot.mat','A','Aci',...
    "CI_va_LG","CI_te_LG","CI_tr_LG",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va");  
    catch
        disp(i)
    end

end

%% ROC
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NN_deep\neural network'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\MGUSandProgression\auc_boot'))
clear TPR_tr FPR_tr TPR_te FPR_te TPR_va FPR_va prec_tr tpr_tr prec_te tpr_te prec_va tpr_va
% Bootstrap test of difference from 0.5
working_rnd = [];

for rng_num = 1:1
    rng(rng_num,'twister')
    try
p = auc_bootstrap([true_train_labelmm(train_cohort),yhatMM_lg_tr])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(train_cohort);
yy = yhatMM_lg_tr;
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
   [prec_tr{i}, tpr_tr{i}, ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end    

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(test_cohort),yhatMM_lg_te])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(test_cohort);
yy = yhatMM_lg_te;
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
   [prec_te{i}, tpr_te{i}, ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end 

% Bootstrap test of difference from 0.5
p = auc_bootstrap([true_train_labelmm(valid_cohort),yhatMM_lg_va])
alpha = 0.05;
nsamp = 50;
tt = true_train_labelmm(valid_cohort);
yy = yhatMM_lg_va;
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
   [prec_va{i}, tpr_va{i}, ~, ~] = prec_rec(y, t,'instanceCount',ones(1,nsamp));
end 

A = [A_tr; A_te; A_va];
Aci = [Aci_tr; Aci_te; Aci_va];

CI_fun = @(x) [mean(x),mean(x) - 1.96.*(std(x)./sqrt(numel(x))), ...
               mean(x) + 1.96.*(std(x)./sqrt(numel(x)))];
CI_AUC_tr_lg = CI_fun(A_tr);
CI_PPV_tr_lg = CI_fun(PPV_tr_CI);
CI_SEN_tr_lg = CI_fun(SEN_tr_CI);
CI_ACC_tr_lg= CI_fun(Accu_tr_CI);
CI_F1_tr_lg = CI_fun(F1_score_tr_CI);
CI_tr_LG = [CI_AUC_tr_lg;CI_PPV_tr_lg;CI_SEN_tr_lg;CI_ACC_tr_lg;CI_F1_tr_lg];

CI_AUC_te_lg = CI_fun(A_te);
CI_PPV_te_lg = CI_fun(PPV_te_CI);
CI_SEN_te_lg = CI_fun(SEN_te_CI);
CI_ACC_te_lg = CI_fun(Accu_te_CI);
CI_F1_te_lg = CI_fun(F1_score_te_CI);
CI_te_LG = [CI_AUC_te_lg;CI_PPV_te_lg;CI_SEN_te_lg;CI_ACC_te_lg;CI_F1_te_lg];

CI_AUC_va_lg = CI_fun(A_va);
CI_PPV_va_lg = CI_fun(PPV_va_CI);
CI_SEN_va_lg = CI_fun(SEN_va_CI);
CI_ACC_va_lg = CI_fun(Accu_va_CI);
CI_F1_va_lg = CI_fun(F1_score_va_CI);
CI_va_LG = [CI_AUC_va_lg;CI_PPV_va_lg;CI_SEN_va_lg;CI_ACC_va_lg;CI_F1_va_lg];

save('LGAUCMM_plot.mat','A','Aci',...
    "CI_va_LG","CI_te_LG","CI_tr_LG",...
    "TPR_tr","FPR_tr","TPR_te","TPR_va","FPR_va","FPR_te",...
    "PPV_tr_CI","SEN_tr_CI","Accu_tr_CI","F1_score_tr_CI",...
    "PPV_te_CI","SEN_te_CI","Accu_te_CI","F1_score_te_CI",...
    "PPV_va_CI","SEN_va_CI","Accu_va_CI","F1_score_va_CI",...
    "prec_tr", "tpr_tr", "prec_te", "tpr_te", "prec_va", "tpr_va"); 
working_rnd  = [working_rnd, rng_num];
    catch
        
    end

end

%%
save('ML_results_02232023.mat')
%%
test_pid = pid(test_cohort);
[earliest_timeMGUS_te,earliest_timeMM_te,...
    yhatMGUS_lg_te,yhatMM_lg_te] = NLP_TP(...
    test_cohort,pid,clinicaldata,Treat_records,...
    ind_MGUS,ind_protein,ind_mspike_mgus,ind_mspike_mm,...
    mspike_date,ind1mspike,ind2mspike,...
    ind_NotMM,ind_MM, ind_SMM, ...
    ind_klratio,ind2klratio,KL_date,ind_PC,PC,PC_date,...
    KL_PatientSSN, MK_PatientSSN,PCPatientSSN,...
    yhatMGUS_lg_te, yhatMM_lg_te, ReportTime);

% update performance after date
[sen_mgus_lg_te_date,spe_mgus_lg_te_date,PPV_mgus_lg_te_date,...
NPV_mgus_lg_te_date,Accu_mgus_lg_te_date,F1_score_mgus_lg_te_date] =...
    CalcPerformance(yhatMGUS_lg_te,true_train_label(test_cohort));
[sen_mm_lg_te_date,spe_mm_lg_te_date,PPV_mm_lg_te_date,...
    NPV_mm_lg_te_date,Accu_mm_lg_te_date,F1_score_mm_lg_te_date] =...
    CalcPerformance(yhatMM_lg_te,true_train_labelmm(test_cohort));


Days_diff = 180;

TP_test_MGUS = zeros(sum(test_cohort),1);
FP_test_MGUS = zeros(sum(test_cohort),1);
TN_test_MGUS = zeros(sum(test_cohort),1);
FN_test_MGUS = zeros(sum(test_cohort),1);

TP_test_MM = zeros(sum(test_cohort),1);
FP_test_MM = zeros(sum(test_cohort),1);
TN_test_MM = zeros(sum(test_cohort),1);
FN_test_MM = zeros(sum(test_cohort),1);

date_diff_mgus = zeros(sum(test_cohort),1);
date_diff_mm = zeros(sum(test_cohort),1);
True_MGUS_date = table2array(TESTtable(:,2));
True_MM_date = table2array(TESTtable(:,5));

for i = 1:sum(test_cohort)
    this_test = find(test_pid(i) == pid);
    
    TP_test_MGUS(i) = ~isempty(True_MGUS_date{this_test}) & ~isempty(earliest_timeMGUS_te{i});
    FP_test_MGUS(i) = isempty(True_MGUS_date{this_test}) & ~isempty(earliest_timeMGUS_te(i));
    TN_test_MGUS(i) = isempty(True_MGUS_date{this_test}) & isempty(earliest_timeMGUS_te(i));
    FN_test_MGUS(i) = ~isempty(True_MGUS_date{this_test}) & isempty(earliest_timeMGUS_te(i));

    TP_test_MM(i) = ~isempty(True_MM_date{this_test}) & ~isempty(earliest_timeMM_te{i});
    FP_test_MM(i) = isempty(True_MM_date{this_test}) & ~isempty(earliest_timeMM_te{i});
    TN_test_MM(i) = isempty(True_MM_date{this_test}) & isempty(earliest_timeMM_te{i});
    FN_test_MM(i) = ~isempty(True_MM_date{this_test}) & isempty(earliest_timeMM_te{i});

    % check for date match
    if TP_test_MGUS(i) ~= 0
    if any(abs(datenum(True_MGUS_date(this_test)) -  datenum(earliest_timeMGUS_te(i)))<=Days_diff)
        % check value for the matched date
        if any(abs(datenum(True_MGUS_date{this_test}) -  datenum(earliest_timeMGUS_te{i}))>Days_limit)
            TP_test_MGUS(i) = 0;
            FP_test_MGUS(i) = 1;
        end
        date_diff_mgus(i) = abs(datenum(True_MGUS_date(this_test)) -  datenum(earliest_timeMGUS_te(i)));
    end 
    end
    
    if TP_test_MM(i) ~= 0
    if any(abs(datenum(True_MM_date(this_test)) -  datenum(earliest_timeMM_te{i}))<=Days_diff)
        if any(abs(datenum(True_MM_date{this_test}) -  datenum(earliest_timeMM_te{i}))>Days_limit)
            TP_test_MM(i) = 0;
            FP_test_MM(i) = 1;
        end
        date_diff_mm(i) = abs(datenum(True_MM_date(this_test)) -  datenum(earliest_timeMM_te{i}));
    end 
    end
end

% date agree
date_diff_test_mgus = date_diff_mgus(logical(TP_test_MGUS));
date_diff_test_mm = date_diff_mm(logical(TP_test_MM));

dates_agree = [sum(date_diff_test_mgus == 0),100*sum(date_diff_test_mgus == 0)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates7 =[sum(date_diff_test_mgus <= 7), 100*sum(date_diff_test_mgus <= 7)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates30 = [sum(date_diff_test_mgus <= 30),100*sum(date_diff_test_mgus <= 30)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates45 = [sum(date_diff_test_mgus <= 45),100*sum(date_diff_test_mgus <= 45)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates90 = [sum(date_diff_test_mgus <= 90),100*sum(date_diff_test_mgus <= 90)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
dates180 = [sum(date_diff_test_mgus <= 180),100*sum(date_diff_test_mgus <= 180)/(sum(TP_test_MGUS) + sum(FP_test_MGUS))];
day_matrix_test_mgus = [dates_agree;dates7;dates30;dates45;dates90;dates180];

dates_agree = [sum(date_diff_test_mm == 0),100*sum(date_diff_test_mm == 0)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates7 =[sum(date_diff_test_mm <= 7), 100*sum(date_diff_test_mm <= 7)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates30 = [sum(date_diff_test_mm <= 30),100*sum(date_diff_test_mm <= 30)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates45 = [sum(date_diff_test_mm <= 45),100*sum(date_diff_test_mm <= 45)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates90 = [sum(date_diff_test_mm <= 90),100*sum(date_diff_test_mm <= 90)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates180 = [sum(date_diff_test_mm <= 180),100*sum(date_diff_test_mm <= 180)/(sum(TP_test_MM) + sum(FP_test_MM))];

day_matrix_test_mm = [dates_agree;dates7;dates30;dates45;dates90;dates180];


%% +-45 days LG valid
valid_pid = pid(valid_cohort);
[earliest_timeMGUS_va,earliest_timeMM_va,...
    yhatMGUS_lg_va,yhatMM_lg_va] = NLP_TP(...
    valid_cohort,pid,clinicaldata,Treat_records,...
    ind_MGUS,ind_protein,ind_mspike_mgus,ind_mspike_mm,...
    mspike_date,ind1mspike,ind2mspike,...
    ind_NotMM,ind_MM, ind_SMM, ...
    ind_klratio,ind2klratio,KL_date,ind_PC,PC,PC_date,...
    KL_PatientSSN, MK_PatientSSN,PCPatientSSN,...
    yhatMGUS_lg_va, yhatMM_lg_va, ReportTime);

% update performance after date
[sen_mgus_lg_va_date,spe_mgus_lg_va_date,PPV_mgus_lg_va_date,...
NPV_mgus_lg_va_date,Accu_mgus_lg_va_date,F1_score_mgus_lg_va_date] =...
    CalcPerformance(yhatMGUS_lg_va,true_train_label(valid_cohort));
[sen_mm_lg_va_date,spe_mm_lg_va_date,PPV_mm_lg_va_date,...
    NPV_mm_lg_va_date,Accu_mm_lg_va_date,F1_score_mm_lg_va_date] =...
    CalcPerformance(yhatMM_lg_va,true_train_labelmm(valid_cohort));

Days_diff = 180;

TP_valid_MGUS = zeros(sum(valid_cohort),1);
FP_valid_MGUS = zeros(sum(valid_cohort),1);
TN_valid_MGUS = zeros(sum(valid_cohort),1);
FN_valid_MGUS = zeros(sum(valid_cohort),1);

TP_valid_MM = zeros(sum(valid_cohort),1);
FP_valid_MM = zeros(sum(valid_cohort),1);
TN_valid_MM = zeros(sum(valid_cohort),1);
FN_valid_MM = zeros(sum(valid_cohort),1);

date_diff = zeros(sum(valid_cohort),1);
date_diff_mm = zeros(sum(valid_cohort),1);
True_MGUS_date = table2array(TESTtable(:,2));
True_MM_date = table2array(TESTtable(:,5));

for i = 1:sum(valid_cohort)
    this_valid = find(valid_pid(i) == pid);
    
    TP_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & ~isempty(earliest_timeMGUS_va{i});
    FP_valid_MGUS(i) = isempty(True_MGUS_date{this_valid}) & ~isempty(earliest_timeMGUS_va(i));
    TN_valid_MGUS(i) = isempty(True_MGUS_date{this_valid}) & isempty(earliest_timeMGUS_va(i));
    FN_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & isempty(earliest_timeMGUS_va(i));

    TP_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & ~isempty(earliest_timeMM_va{i});
    FP_valid_MM(i) = isempty(True_MM_date{this_valid}) & ~isempty(earliest_timeMM_va{i});
    TN_valid_MM(i) = isempty(True_MM_date{this_valid}) & isempty(earliest_timeMM_va{i});
    FN_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & isempty(earliest_timeMM_va{i});
    
    % check for date match
    if TP_valid_MGUS(i) ~= 0
    if any(abs(datenum(True_MGUS_date(this_valid)) -  datenum(earliest_timeMGUS_va(i)))<=Days_diff)
        % check value for the matched date
        if any(abs(datenum(True_MGUS_date{this_valid}) -  datenum(earliest_timeMGUS_va{i}))>Days_limit)
            TP_valid_MGUS(i) = 0;
            FP_valid_MGUS(i) = 1;
        end
        date_diff(i) = abs(datenum(True_MGUS_date(this_valid)) -  datenum(earliest_timeMGUS_va(i)));
    end 
    end
    
    if TP_valid_MM(i) ~= 0
    if any(abs(datenum(True_MM_date(this_valid)) -  datenum(earliest_timeMM_va{i}))<=Days_diff)
        if any(abs(datenum(True_MM_date{this_valid}) -  datenum(earliest_timeMM_va{i}))>Days_limit)
            TP_valid_MM(i) = 0;
            FP_valid_MM(i) = 1;
        end
        date_diff_mm(i) = abs(datenum(True_MM_date(this_valid)) -  datenum(earliest_timeMM_va{i}));
    end 
    end
    
end

date_diff_valid = date_diff(logical(TP_valid_MGUS));

dates_agree = [sum(date_diff_valid == 0),100*sum(date_diff_valid == 0)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates7 =[sum(date_diff_valid <= 7), 100*sum(date_diff_valid <= 7)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates30 = [sum(date_diff_valid <= 30),100*sum(date_diff_valid <= 30)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates45 = [sum(date_diff_valid <= 45),100*sum(date_diff_valid <= 45)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates90 = [sum(date_diff_valid <= 90),100*sum(date_diff_valid <= 90)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates180 = [sum(date_diff_valid <= 180),100*sum(date_diff_valid <= 180)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
day_matrix_valid_mgus = [dates_agree;dates7;dates30;dates45;dates90;dates180];

date_diff_valid = date_diff_mm(logical(TP_valid_MM));
dates_agree = [sum(date_diff_valid == 0),100*sum(date_diff_valid == 0)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates7 =[sum(date_diff_valid <= 7), 100*sum(date_diff_valid <= 7)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates30 = [sum(date_diff_valid <= 30),100*sum(date_diff_valid <= 30)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates45 = [sum(date_diff_valid <= 45),100*sum(date_diff_valid <= 45)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates90 = [sum(date_diff_valid <= 90),100*sum(date_diff_valid <= 90)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates180 = [sum(date_diff_valid <= 180),100*sum(date_diff_valid <= 180)/(sum(TP_valid_MM) + sum(FP_valid_MM))];

day_matrix_valid_mm = [dates_agree;dates7;dates30;dates45;dates90;dates180];

% % update performance after date
sen_mgus_lg_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FN_test_MGUS));
spe_mgus_lg_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FP_test_MGUS));
PPV_mgus_lg_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FP_test_MGUS));
NPV_mgus_lg_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FN_test_MGUS));
Accu_mgus_lg_te_date = (sum(TP_test_MGUS)+sum(TN_test_MGUS))/(sum(TP_test_MGUS)+sum(TN_test_MGUS)+sum(FP_test_MGUS)+sum(FN_test_MGUS));
F1_score_mgus_lg_te_date = 2*sum(TP_test_MGUS)/(2*sum(TP_test_MGUS)+sum(FP_test_MGUS)+sum(FN_test_MGUS));

sen_mm_lg_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FN_test_MM));
spe_mm_lg_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FP_test_MM));
PPV_mm_lg_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FP_test_MM));
NPV_mm_lg_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FN_test_MM));
Accu_mm_lg_te_date = (sum(TP_test_MM)+sum(TN_test_MM))/(sum(TP_test_MM)+sum(TN_test_MM)+sum(FP_test_MM)+sum(FN_test_MM));
F1_score_mm_lg_te_date = 2*sum(TP_test_MM)/(2*sum(TP_test_MM)+sum(FP_test_MM)+sum(FN_test_MM));

% update performance after date
sen_mgus_lg_va_date = sum(TP_valid_MGUS)/(sum(TP_valid_MGUS)+sum(FN_valid_MGUS));
spe_mgus_lg_va_date = sum(TN_valid_MGUS)/(sum(TN_valid_MGUS)+sum(FP_valid_MGUS));
PPV_mgus_lg_va_date = sum(TP_valid_MGUS)/(sum(TP_valid_MGUS)+sum(FP_valid_MGUS));
NPV_mgus_lg_va_date = sum(TN_valid_MGUS)/(sum(TN_valid_MGUS)+sum(FN_valid_MGUS));
Accu_mgus_lg_va_date = (sum(TP_valid_MGUS)+sum(TN_valid_MGUS))/(sum(TP_valid_MGUS)+sum(TN_valid_MGUS)+sum(FP_valid_MGUS)+sum(FN_valid_MGUS));
F1_score_mgus_lg_va_date = 2*sum(TP_valid_MGUS)/(2*sum(TP_valid_MGUS)+sum(FP_valid_MGUS)+sum(FN_valid_MGUS));

sen_mm_lg_va_date = sum(TP_valid_MM)/(sum(TP_valid_MM)+sum(FN_valid_MM));
spe_mm_lg_va_date = sum(TN_valid_MM)/(sum(TN_valid_MM)+sum(FP_valid_MM));
PPV_mm_lg_va_date = sum(TP_valid_MM)/(sum(TP_valid_MM)+sum(FP_valid_MM));
NPV_mm_lg_va_date = sum(TN_valid_MM)/(sum(TN_valid_MM)+sum(FN_valid_MM));
Accu_mm_lg_va_date = (sum(TP_valid_MM)+sum(TN_valid_MM))/(sum(TP_valid_MM)+sum(TN_valid_MM)+sum(FP_valid_MM)+sum(FN_valid_MM));
F1_score_mm_lg_va_date = 2*sum(TP_valid_MM)/(2*sum(TP_valid_MM)+sum(FP_valid_MM)+sum(FN_valid_MM));

%% RF-MGUS column
LG_MGUS_va = [PPV_mgus_lg_va_date;sen_mgus_lg_va_date;F1_score_mgus_lg_va_date;Accu_mgus_lg_va_date;...
    ...
    day_matrix_valid_mgus(:,2)];
LG_MGUS_te = [PPV_mgus_lg_te_date;sen_mgus_lg_te_date;F1_score_mgus_lg_te_date;Accu_mgus_lg_te_date;...
    ...
    day_matrix_test_mgus(:,2)];
LG_MM_va = [PPV_mm_lg_va_date;sen_mm_lg_va_date;F1_score_mm_lg_va_date;Accu_mm_lg_va_date;...
    ...
    day_matrix_valid_mm(:,2)];
LG_MM_te = [PPV_mm_lg_te_date;sen_mm_lg_te_date;F1_score_mm_lg_te_date;Accu_mm_lg_te_date;...
    ...
    day_matrix_test_mm(:,2)];
MGUS_LG = [LG_MGUS_va;LG_MGUS_te];
MM_LG = [LG_MM_va;LG_MM_te];
% writetable(array2table(MGUS_LG),'Ressults20230111.xlsx','Sheet','MGUS_LG')
% writetable(array2table(MM_LG),'Ressults20230111.xlsx','Sheet','MM_LG')


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ICD code 
T_ICD = readtable('ICD_identified_results.xlsx');
T_ICD_mugs_labels = table2array(T_ICD(:,3));
T_ICD_mugs_dates = table2array(T_ICD(:,4));

T_ICD_mm_labels = table2array(T_ICD(:,5));
T_ICD_mm_dates = table2array(T_ICD(:,6));

[sen_mgus_icd_te,spe_mgus_icd_te,PPV_mgus_icd_te,...
NPV_mgus_icd_te,Accu_mgus_icd_te,F1_score_mgus_icd_te] =...
    CalcPerformance(T_ICD_mugs_labels(test_cohort),true_train_label(test_cohort));

[sen_mgus_icd_va,spe_mgus_icd_va,PPV_mgus_icd_va,...
NPV_mgus_icd_va,Accu_mgus_icd_va,F1_score_mgus_icd_va] =...
    CalcPerformance(T_ICD_mugs_labels(valid_cohort),true_train_label(valid_cohort));

[sen_mm_icd_te,spe_mm_icd_te,PPV_mm_icd_te,...
NPV_mm_icd_te,Accu_mm_icd_te,F1_score_mm_icd_te] =...
    CalcPerformance(T_ICD_mm_labels(test_cohort),true_train_labelmm(test_cohort));

[sen_mm_icd_va,spe_mm_icd_va,PPV_mm_icd_va,...
NPV_mm_icd_va,Accu_mm_icd_va,F1_score_mm_icd_va] =...
    CalcPerformance(T_ICD_mm_labels(valid_cohort),true_train_labelmm(valid_cohort));

%% Add date +-45 days. ICD code
test_pid = pid(test_cohort);
Days_diff = 180;

TP_test_MGUS = zeros(sum(test_cohort),1);
FP_test_MGUS = zeros(sum(test_cohort),1);
TN_test_MGUS = zeros(sum(test_cohort),1);
FN_test_MGUS = zeros(sum(test_cohort),1);
date_diff = zeros(sum(test_cohort),1);
True_MGUS_date = table2array(TESTtable(:,2));

T_ICD_mugs_dates_test = T_ICD_mugs_dates(test_cohort);
for i = 1:sum(test_cohort)
    this_test = find(test_pid(i) == pid);
    TP_test_MGUS(i) = ~isempty(True_MGUS_date{this_test}) & ~isempty(T_ICD_mugs_dates_test(i));
    FP_test_MGUS(i) = isempty(True_MGUS_date{this_test}) & ~isempty(T_ICD_mugs_dates_test(i));
    TN_test_MGUS(i) = isempty(True_MGUS_date{this_test}) & isempty(T_ICD_mugs_dates_test(i));
    FN_test_MGUS(i) = ~isempty(True_MGUS_date{this_test}) & isempty(T_ICD_mugs_dates_test(i));
    % check for date match
    if TP_test_MGUS(i) ~= 0
        if any(abs(datenum(True_MGUS_date(this_test)) -  datenum(T_ICD_mugs_dates_test(i)))<=Days_diff)
            if any(abs(datenum(True_MGUS_date(this_test)) -  datenum(T_ICD_mugs_dates_test(i)))>45)
                TP_test_MGUS(i) = 0;
                FP_test_MGUS(i) = 1;
            end
            % check value for the matched date
            date_diff(i) = abs(datenum(True_MGUS_date(this_test)) -  datenum(T_ICD_mugs_dates_test(i)));
        end 
    end
end

% date agree
date_diff_test = date_diff(logical(TP_test_MGUS));

dates_agree = [sum(date_diff_test == 0),100*sum(date_diff_test == 0)/(sum(TP_test_MGUS)+sum(FP_test_MGUS))];
dates7 =[sum(date_diff_test <= 7), 100*sum(date_diff_test <= 7)/(sum(TP_test_MGUS)+sum(FP_test_MGUS))];
dates30 = [sum(date_diff_test <= 30),100*sum(date_diff_test <= 30)/(sum(TP_test_MGUS)+sum(FP_test_MGUS))];
dates45 = [sum(date_diff_test <= 45),100*sum(date_diff_test <= 45)/(sum(TP_test_MGUS)+sum(FP_test_MGUS))];
dates90 = [sum(date_diff_test <= 90),100*sum(date_diff_test <= 90)/(sum(TP_test_MGUS)+sum(FP_test_MGUS))];
dates180 = [sum(date_diff_test <= 180),100*sum(date_diff_test <= 180)/(sum(TP_test_MGUS)+sum(FP_test_MGUS))];
day_matrix_test = [dates_agree;dates7;dates30;dates45;dates90;dates180];


sen_mgus_icd_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FN_test_MGUS));
spe_mgus_icd_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FP_test_MGUS));
PPV_mgus_icd_te_date = sum(TP_test_MGUS)/(sum(TP_test_MGUS)+sum(FP_test_MGUS));
NPV_mgus_icd_te_date = sum(TN_test_MGUS)/(sum(TN_test_MGUS)+sum(FN_test_MGUS));
Accu_mgus_icd_te_date = (sum(TP_test_MGUS)+sum(TN_test_MGUS))/(sum(TP_test_MGUS)+sum(TN_test_MGUS)+sum(TP_test_MGUS)+sum(FN_test_MGUS));
F1_score_mgus_icd_te_date = 2*sum(TP_test_MGUS)/(2*sum(TP_test_MGUS)+sum(FP_test_MGUS)+sum(FN_test_MGUS));

%% +-45 days icd valid
valid_pid = pid(valid_cohort);
Days_diff = 180;

TP_valid_MGUS = zeros(sum(valid_cohort),1);
FP_valid_MGUS = zeros(sum(valid_cohort),1);
TN_valid_MGUS = zeros(sum(valid_cohort),1);
FN_valid_MGUS = zeros(sum(valid_cohort),1);

date_diff = zeros(sum(valid_cohort),1);
True_MGUS_date = table2array(TESTtable(:,2));
True_MM_date = table2array(TESTtable(:,5));

T_ICD_mugs_dates_valid = T_ICD_mugs_dates(valid_cohort);
for i = 1:sum(valid_cohort)
    this_valid = find(valid_pid(i) == pid);
    TP_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & ~isempty(T_ICD_mugs_dates_valid(i));
    FP_valid_MGUS(i) = isempty(True_MGUS_date{this_valid}) & ~isempty(T_ICD_mugs_dates_valid(i));
    TN_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & ~isempty(T_ICD_mugs_dates_valid(i));
    FN_valid_MGUS(i) = ~isempty(True_MGUS_date{this_valid}) & isempty(T_ICD_mugs_dates_valid(i));

    % check for date match
    if TP_valid_MGUS(i) ~= 0
        if any(abs(datenum(True_MGUS_date(this_valid)) -  datenum(T_ICD_mugs_dates_valid(i)))<=Days_diff)
            if any(abs(datenum(True_MGUS_date(this_valid)) -  datenum(T_ICD_mugs_dates_valid(i)))>45)
                TP_valid_MGUS(i) = 0;
                FP_valid_MGUS(i) = 1;
            end
            % check value for the matched date
            date_diff(i) = abs(datenum(True_MGUS_date(this_valid)) -  datenum(T_ICD_mugs_dates_valid(i)));
        end 
    end
end

date_diff_valid = date_diff(logical(TP_valid_MGUS));

dates_agree = [sum(date_diff_valid == 0),100*sum(date_diff_valid == 0)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates7 =[sum(date_diff_valid <= 7), 100*sum(date_diff_valid <= 7)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates30 = [sum(date_diff_valid <= 30),100*sum(date_diff_valid <= 30)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates45 = [sum(date_diff_valid <= 45),100*sum(date_diff_valid <= 45)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates90 = [sum(date_diff_valid <= 90),100*sum(date_diff_valid <= 90)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];
dates180 = [sum(date_diff_valid <= 180),100*sum(date_diff_valid <= 180)/(sum(TP_valid_MGUS) + sum(FP_valid_MGUS))];

day_matrix_valid = [dates_agree;dates7;dates30;dates45;dates90;dates180];

sen_mgus_icd_va_date = sum(TP_valid_MGUS)/(sum(TP_valid_MGUS)+sum(FN_valid_MGUS));
spe_mgus_icd_va_date = sum(TN_valid_MGUS)/(sum(TN_valid_MGUS)+sum(FP_valid_MGUS));
PPV_mgus_icd_va_date = sum(TP_valid_MGUS)/(sum(TP_valid_MGUS)+sum(FP_valid_MGUS));
NPV_mgus_icd_va_date = sum(TN_valid_MGUS)/(sum(TN_valid_MGUS)+sum(FN_valid_MGUS));
Accu_mgus_icd_va_date = (sum(TP_valid_MGUS)+sum(TN_valid_MGUS))/(sum(TP_valid_MGUS)+sum(TN_valid_MGUS)+sum(TP_valid_MGUS)+sum(FN_valid_MGUS));
F1_score_mgus_icd_va_date = 2*sum(TP_valid_MGUS)/(2*sum(TP_valid_MGUS)+sum(FP_valid_MGUS)+sum(FN_valid_MGUS));

%% RF-MGUS column
ICD_MGUS_va = [PPV_mgus_icd_va_date;sen_mgus_icd_va_date;...
    F1_score_mgus_icd_va_date;Accu_mgus_icd_va_date;...
    day_matrix_valid(:,2)];
ICD_MGUS_te = [PPV_mgus_icd_te_date;sen_mgus_icd_te_date;...
    F1_score_mgus_icd_te_date;Accu_mgus_icd_te_date;...
    day_matrix_test(:,2)];
MGUS_ICD = [ICD_MGUS_va;ICD_MGUS_te];

writetable(array2table(MGUS_ICD),'Ressults20230111.xlsx','Sheet','MGUS_ICD')

%% ICD - MM
%% Add date +-45 days. ICD code
test_pid = pid(test_cohort);
Days_diff = 180;

TP_test_MM = zeros(sum(test_cohort),1);
FP_test_MM = zeros(sum(test_cohort),1);
TN_test_MM = zeros(sum(test_cohort),1);
FN_test_MM = zeros(sum(test_cohort),1);
date_diff = zeros(sum(test_cohort),1);
True_MM_date = table2array(TESTtable(:,5));

T_ICD_mm_dates_test = T_ICD_mm_dates(test_cohort);
for i = 1:sum(test_cohort)
    this_test = find(test_pid(i) == pid);
    TP_test_MM(i) = ~isempty(True_MM_date{this_test}) & ~isempty( T_ICD_mm_dates_test{i});
    TN_test_MM(i) = isempty(True_MM_date{this_test}) & isempty( T_ICD_mm_dates_test{i});
    FP_test_MM(i) = isempty(True_MM_date{this_test}) & ~isempty( T_ICD_mm_dates_test{i});
    FN_test_MM(i) = ~isempty(True_MM_date{this_test}) & isempty( T_ICD_mm_dates_test{i});
    % check for date match
    if TP_test_MM(i) ~= 0
        if any(abs(datenum(True_MM_date(this_test)) -  datenum(T_ICD_mm_dates_test(i)))<=Days_diff)
            if any(abs(datenum(True_MM_date(this_test)) -  datenum(T_ICD_mm_dates_test(i)))>45)
                TP_test_MM(i) = 0;
                FP_test_MM(i) = 1;
            end
            % check value for the matched date
            date_diff(i) = abs(datenum(True_MM_date(this_test)) -  datenum(T_ICD_mm_dates_test(i)));
        end 
    end
end

% date agree
date_diff_test = date_diff(logical(TP_test_MM));

dates_agree = [sum(date_diff_test == 0),100*sum(date_diff_test == 0)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates7 =[sum(date_diff_test <= 7), 100*sum(date_diff_test <= 7)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates30 = [sum(date_diff_test <= 30),100*sum(date_diff_test <= 30)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates45 = [sum(date_diff_test <= 45),100*sum(date_diff_test <= 45)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates90 = [sum(date_diff_test <= 90),100*sum(date_diff_test <= 90)/(sum(TP_test_MM) + sum(FP_test_MM))];
dates180 = [sum(date_diff_test <= 180),100*sum(date_diff_test <= 180)/(sum(TP_test_MM) + sum(FP_test_MM))];
day_matrix_test = [dates_agree;dates7;dates30;dates45;dates90;dates180];

sen_mm_icd_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FN_test_MM));
spe_mm_icd_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FP_test_MM));
PPV_mm_icd_te_date = sum(TP_test_MM)/(sum(TP_test_MM)+sum(FP_test_MM));
NPV_mm_icd_te_date = sum(TN_test_MM)/(sum(TN_test_MM)+sum(FN_test_MM));
Accu_mm_icd_te_date = (sum(TP_test_MM)+sum(TN_test_MM))/(sum(TP_test_MM)+sum(TN_test_MM)+sum(TP_test_MM)+sum(FN_test_MM));
F1_score_mm_icd_te_date = 2*sum(TP_test_MM)/(2*sum(TP_test_MM)+sum(FP_test_MM)+sum(FN_test_MM));

%% +-45 days icd valid
valid_pid = pid(valid_cohort);
Days_diff = 180;

TP_valid_MM = zeros(sum(valid_cohort),1);
FP_valid_MM = zeros(sum(valid_cohort),1);
TN_valid_MM = zeros(sum(valid_cohort),1);
FN_valid_MM = zeros(sum(valid_cohort),1);
date_diff = zeros(sum(valid_cohort),1);
True_MM_date = table2array(TESTtable(:,5));

T_ICD_mm_dates_valid = T_ICD_mm_dates(valid_cohort);
for i = 1:sum(valid_cohort)
    this_valid = find(valid_pid(i) == pid);
    TP_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & ~isempty(T_ICD_mm_dates_valid{i});
    TN_valid_MM(i) = isempty(True_MM_date{this_valid}) & isempty(T_ICD_mm_dates_valid{i});
    FP_valid_MM(i) = ~isempty(True_MM_date{this_valid}) & isempty(T_ICD_mm_dates_valid{i});
    FN_valid_MM(i) = isempty(True_MM_date{this_valid}) & ~isempty(T_ICD_mm_dates_valid{i});
    % check for date match
    if TP_valid_MM(i) ~= 0
        if any(abs(datenum(True_MM_date(this_valid)) -  datenum(T_ICD_mm_dates_valid(i)))<=Days_diff)
            if any(abs(datenum(True_MM_date(this_valid)) -  datenum(T_ICD_mm_dates_valid(i)))>45)
                TP_valid_MM(i) = 0;
                FP_valid_MM(i) = 1;
            end
            % check value for the matched date
            date_diff(i) = abs(datenum(True_MM_date(this_valid)) -  datenum(T_ICD_mm_dates_valid(i)));
        end 
    end
end

date_diff_valid = date_diff(logical(TP_valid_MM));

dates_agree = [sum(date_diff_valid == 0),100*sum(date_diff_valid == 0)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates7 =[sum(date_diff_valid <= 7), 100*sum(date_diff_valid <= 7)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates30 = [sum(date_diff_valid <= 30),100*sum(date_diff_valid <= 30)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates45 = [sum(date_diff_valid <= 45),100*sum(date_diff_valid <= 45)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates90 = [sum(date_diff_valid <= 90),100*sum(date_diff_valid <= 90)/(sum(TP_valid_MM) + sum(FP_valid_MM))];
dates180 = [sum(date_diff_valid <= 180),100*sum(date_diff_valid <= 180)/(sum(TP_valid_MM) + sum(FP_valid_MM))];

day_matrix_valid = [dates_agree;dates7;dates30;dates45;dates90;dates180];

sen_mm_icd_va_date = sum(TP_valid_MM)/(sum(TP_valid_MM)+sum(FN_valid_MM));
spe_mm_icd_va_date = sum(TN_valid_MM)/(sum(TN_valid_MM)+sum(FP_valid_MM));
PPV_mm_icd_va_date = sum(TP_valid_MM)/(sum(TP_valid_MM)+sum(FP_valid_MM));
NPV_mm_icd_va_date = sum(TN_valid_MM)/(sum(TN_valid_MM)+sum(FN_valid_MM));
Accu_mm_icd_va_date = (sum(TP_valid_MM)+sum(TN_valid_MM))/(sum(TP_valid_MM)+sum(TN_valid_MM)+sum(TP_valid_MM)+sum(FN_valid_MM));
F1_score_mm_icd_va_date = 2*sum(TP_valid_MM)/(2*sum(TP_valid_MM)+sum(FP_valid_MM)+sum(FN_valid_MM));

%% ICD-MM column
ICD_MM_va = [PPV_mm_icd_va_date;sen_mm_icd_va_date;...
    F1_score_mm_icd_va_date;Accu_mm_icd_va_date;...
    day_matrix_valid(:,2)];
ICD_MM_te = [PPV_mm_icd_te_date;sen_mm_icd_te_date;...
    F1_score_mm_icd_te_date;Accu_mm_icd_te_date;...
    day_matrix_test(:,2)];
MM_ICD = [ICD_MM_va;ICD_MM_te];

writetable(array2table(MM_ICD),'Ressults20230111.xlsx','Sheet','MM_ICD')
