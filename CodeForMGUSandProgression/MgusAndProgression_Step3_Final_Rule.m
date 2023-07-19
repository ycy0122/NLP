clc;clear;close all
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\cohort1\Finished'))
read_reflists = true;
save_name = 'Final_060523_Rule';
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
plasmacell_filename = "True_Plasma_cell.xlsx";
plasmacell = readtable(plasmacell_filename);
PCPatientSSN = str2double(plasmacell.PatientSSN);
PC = cellfun(@str2num,(plasmacell.x_OfPlasma));
PC_date = plasmacell.Date;
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
sensitivitymm = sum(classify_labelmm == 1 & true_train_labelmm == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm == 1) +  sum(classify_labelmm == 0 & true_train_labelmm == 1))
specificitymm = sum(classify_labelmm == 0 & true_train_labelmm == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm == 0) +  sum(classify_labelmm == 1 & true_train_labelmm == 0))
PPVmm =  sum(classify_labelmm == 1 & true_train_labelmm == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm == 1) +  sum(classify_labelmm == 1 & true_train_labelmm == 0))
NPVmm =  sum(classify_labelmm == 0 & true_train_labelmm == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm == 0) +  sum(classify_labelmm == 0 & true_train_labelmm == 1))
Accuracymm = sum(classify_labelmm==true_train_labelmm)/numel(true_train_labelmm)
F1_scoremm = 2/(1/sensitivitymm + 1/PPVmm)

%% MM FP FN
FP_MM = classify_labelmm == 1 & true_train_labelmm == 0;
FP_MM_pid = pid(FP_MM);
writetable(array2table(FP_MM_pid),['FP_MM',save_name,'.txt'])
FN_MM = classify_labelmm == 0 & true_train_labelmm == 1;
FN_MM_pid = pid(FN_MM);
writetable(array2table(FN_MM_pid),['FN_MM',save_name,'.txt'])

T = readtable('True_MGUSPROG.xlsx','Sheet','Sheet1','Range','F1:F701');
train_cohort = strcmp(table2array(T),'TR');
test_cohort = strcmp(table2array(T),'TE');
valid_cohort = strcmp(table2array(T),'VA');

% train
MGUS_set1_tr = [];
MGUS_set1_tr = any(ConditionMGUS_matrix(train_cohort,logical(combsMGUS(7,:))),2);
true_train_label_tr = true_train_label(train_cohort);
classify_label = MGUS_set1_tr;
sensitivity = sum(classify_label == 1 & true_train_label_tr == 1)./ ( sum(classify_label == 1 & true_train_label_tr == 1) +  sum(classify_label == 0 & true_train_label_tr == 1))
specificity = sum(classify_label == 0 & true_train_label_tr == 0)./ ( sum(classify_label == 0 & true_train_label_tr == 0) +  sum(classify_label == 1 & true_train_label_tr == 0))
PPV =  sum(classify_label == 1 & true_train_label_tr == 1)./ ( sum(classify_label == 1 & true_train_label_tr == 1) +  sum(classify_label == 1 & true_train_label_tr == 0))
NPV =  sum(classify_label == 0 & true_train_label_tr == 0)./ ( sum(classify_label == 0 & true_train_label_tr == 0) +  sum(classify_label == 0 & true_train_label_tr == 1))
Accuracy = sum(classify_label==true_train_label_tr)/numel(true_train_label_tr)
F1_score = 2/(1/sensitivity + 1/PPV)

% test
MGUS_set1_te = [];
MGUS_set1_te = any(ConditionMGUS_matrix(test_cohort,logical(combsMGUS(7,:))),2);
true_train_label_te = true_train_label(test_cohort);
classify_label = MGUS_set1_te;
sensitivity = sum(classify_label == 1 & true_train_label_te == 1)./ ( sum(classify_label == 1 & true_train_label_te == 1) +  sum(classify_label == 0 & true_train_label_te == 1))
specificity = sum(classify_label == 0 & true_train_label_te == 0)./ ( sum(classify_label == 0 & true_train_label_te == 0) +  sum(classify_label == 1 & true_train_label_te == 0))
PPV =  sum(classify_label == 1 & true_train_label_te == 1)./ ( sum(classify_label == 1 & true_train_label_te == 1) +  sum(classify_label == 1 & true_train_label_te == 0))
NPV =  sum(classify_label == 0 & true_train_label_te == 0)./ ( sum(classify_label == 0 & true_train_label_te == 0) +  sum(classify_label == 0 & true_train_label_te == 1))
Accuracy = sum(classify_label==true_train_label_te)/numel(true_train_label_te)
F1_score = 2/(1/sensitivity + 1/PPV)

% valid
MGUS_set1_va = [];
MGUS_set1_va = any(ConditionMGUS_matrix(valid_cohort,logical(combsMGUS(7,:))),2);
true_train_label_va = true_train_label(valid_cohort);
classify_label = MGUS_set1_va;
sensitivity = sum(classify_label == 1 & true_train_label_va == 1)./ ( sum(classify_label == 1 & true_train_label_va == 1) +  sum(classify_label == 0 & true_train_label_va == 1))
specificity = sum(classify_label == 0 & true_train_label_va == 0)./ ( sum(classify_label == 0 & true_train_label_va == 0) +  sum(classify_label == 1 & true_train_label_va == 0))
PPV =  sum(classify_label == 1 & true_train_label_va == 1)./ ( sum(classify_label == 1 & true_train_label_va == 1) +  sum(classify_label == 1 & true_train_label_va == 0))
NPV =  sum(classify_label == 0 & true_train_label_va == 0)./ ( sum(classify_label == 0 & true_train_label_va == 0) +  sum(classify_label == 0 & true_train_label_va == 1))
Accuracy = sum(classify_label==true_train_label_va)/numel(true_train_label_va)
F1_score = 2/(1/sensitivity + 1/PPV)

% MM
% train
true_train_labelmm_tr = true_train_labelmm(train_cohort);
classify_labelmm = ismember(pid(train_cohort),(pid(logical(MM_set1.'))));
sensitivitymm = sum(classify_labelmm == 1 & true_train_labelmm_tr == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm_tr == 1) +  sum(classify_labelmm == 0 & true_train_labelmm_tr == 1))
specificitymm = sum(classify_labelmm == 0 & true_train_labelmm_tr == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm_tr == 0) +  sum(classify_labelmm == 1 & true_train_labelmm_tr == 0))
PPVmm =  sum(classify_labelmm == 1 & true_train_labelmm_tr == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm_tr == 1) +  sum(classify_labelmm == 1 & true_train_labelmm_tr == 0))
NPVmm =  sum(classify_labelmm == 0 & true_train_labelmm_tr == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm_tr == 0) +  sum(classify_labelmm == 0 & true_train_labelmm_tr == 1))
Accuracymm = sum(classify_labelmm==true_train_labelmm_tr)/numel(true_train_labelmm_tr)
F1_scoremm = 2/(1/sensitivitymm + 1/PPVmm)

% test
true_train_labelmm_te = true_train_labelmm(test_cohort);
classify_labelmm = ismember(pid(test_cohort),(pid(logical(MM_set1.'))));
sensitivitymm = sum(classify_labelmm == 1 & true_train_labelmm_te == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm_te == 1) +  sum(classify_labelmm == 0 & true_train_labelmm_te == 1))
specificitymm = sum(classify_labelmm == 0 & true_train_labelmm_te == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm_te == 0) +  sum(classify_labelmm == 1 & true_train_labelmm_te == 0))
PPVmm =  sum(classify_labelmm == 1 & true_train_labelmm_te == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm_te == 1) +  sum(classify_labelmm == 1 & true_train_labelmm_te == 0))
NPVmm =  sum(classify_labelmm == 0 & true_train_labelmm_te == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm_te == 0) +  sum(classify_labelmm == 0 & true_train_labelmm_te == 1))
Accuracymm = sum(classify_labelmm==true_train_labelmm_te)/numel(true_train_labelmm_te)
F1_scoremm = 2/(1/sensitivitymm + 1/PPVmm)


% valid
true_train_labelmm_va = true_train_labelmm(valid_cohort);
classify_labelmm = ismember(pid(valid_cohort),(pid(logical(MM_set1.'))));
sensitivitymm = sum(classify_labelmm == 1 & true_train_labelmm_va == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm_va == 1) +  sum(classify_labelmm == 0 & true_train_labelmm_va == 1))
specificitymm = sum(classify_labelmm == 0 & true_train_labelmm_va == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm_va == 0) +  sum(classify_labelmm == 1 & true_train_labelmm_va == 0))
PPVmm =  sum(classify_labelmm == 1 & true_train_labelmm_va == 1)./ ( sum(classify_labelmm == 1 & true_train_labelmm_va == 1) +  sum(classify_labelmm == 1 & true_train_labelmm_va == 0))
NPVmm =  sum(classify_labelmm == 0 & true_train_labelmm_va == 0)./ ( sum(classify_labelmm == 0 & true_train_labelmm_va == 0) +  sum(classify_labelmm == 0 & true_train_labelmm_va == 1))
Accuracymm = sum(classify_labelmm==true_train_labelmm_va)/numel(true_train_labelmm_va)
F1_scoremm = 2/(1/sensitivitymm + 1/PPVmm)


%% earliest MGUS time
earliest_timeMGUS = {};
for i = 1:length(pid)
    this_pid = find(table2array(clinicaldata(:,1))== pid(i));

    this_MGUS = any(ind_MGUS(this_pid) == 1);           %  C1
    this_PROP = any(ind_protein(this_pid) == 1);        %  C2
    this_MSPK = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mgus)); 
    this_NoMM = any(ind_NotMM(this_pid) == 1);          %  C4

    C_MGUS_small = [[ind_MGUS(this_pid) == 1].' & [ind_protein(this_pid) == 1].',...
                    [ind_protein(this_pid) == 1].'];

    if MGUS_set1(i) == 1
        time_id1 = find(C_MGUS_small(:,1),1,'first');
        time_id2 = find(C_MGUS_small(:,2),1,'first');
        if ~isempty(time_id1) && isempty(time_id2)
            earliest_timeMGUS{i} = datestr(ReportTime{this_pid(time_id1)});
        elseif isempty(time_id1) && ~isempty(time_id2)
            earliest_timeMGUS{i} = datestr(ReportTime{this_pid(time_id2)});
        elseif ~isempty(time_id1) && ~isempty(time_id2)

            earliest_timeMGUS{i} = datestr(ReportTime{this_pid(min([time_id1,time_id2]))});                    
        elseif isempty(time_id1) && isempty(time_id2)

            this_MSPK_T = (intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mgus));
            if ~isempty(this_MSPK_T)
                MSPK_time_id = find(this_MSPK_T == MK_PatientSSN);
                MSPK_time_id2 = ind1mspike(MSPK_time_id);
                MSPK_time = mspike_date(MSPK_time_id(MSPK_time_id2));
            end 
            earliest_timeMGUS{i} = datestr(min(MSPK_time));
        end

    else
        earliest_timeMGUS{i} = [];
    end
end

        for i = 1:length(pid)
            if ~isempty(earliest_timeMM{i})
                time_cell{i,1} = datetime(earliest_timeMM{i});
            end
        end

        %% write as excel sheet
        T_pid = array2table(pid);
        TMGUS = cell2table(earliest_timeMGUS.');
        %%
        writetable(T_pid,['Results_clinical_700_',save_name,'.xlsx'],'Sheet','pid')
        writetable(TMGUS,['Results_clinical_700_',save_name,'.xlsx'],'Sheet','diagMGUS_time')

        if ~isempty(table2array(cell2table(MGUS_report)))
            writetable(cell2table(MGUS_report),...
                ['Results_clinical_700_',save_name,'.xlsx'],...
                'Sheet','MGUS','WriteVariableNames',0)
        end
        if ~isempty(table2array(cell2table(NotMM_report)))
            writetable(cell2table(NotMM_report),...
                ['Results_clinical_700_',save_name,'.xlsx'],...
                'Sheet','NotMM','WriteVariableNames',0)
        end
        if ~isempty(table2array(cell2table(Protein_report)))
            writetable(cell2table(Protein_report),...
                ['Results_clinical_700_',save_name,'.xlsx'],...
                'Sheet','Protein','WriteVariableNames',0)
        end
        if ~isempty(table2array(cell2table(mspk_report)))
            writetable(cell2table(mspk_report),...
                ['Results_clinical_700_',save_name,'.xlsx'],...
                'Sheet','mspike','WriteVariableNames',0)
        end
        writetable(array2table(MGUS_label),...
            ['Results_clinical_700_',save_name,'.xlsx'],...
            'Sheet','MGUS_label','WriteVariableNames',0)

        
    %% use best_com_ss again
    % MM_set1 = [];
    % MM_set1 = any(Condition_matrix(:,logical(combs(end,:))),2);
    MM_label = ismember(pid,(pid(logical(MM_set1))));

    %% save results...
    save(['Results_clinical_700_',save_name,'.mat'],'earliest_timeMM','MM_label',...
        'ind_PC','ind_klratio','ind_MM','ind_mspike_mm','ind_mspike_mgus','ind_Treat','ind_SMM',...
        'earliest_timeMGUS','MGUS_label','ind_MGUS','ind_protein','ind_NotMM')

    %% write as excel sheet
    T_pid = array2table(pid);
    T1 = cell2table(earliest_timeMM);

    %%
%     writetable(T_pid,['Results_clinical_s',num2str(s),'.xlsx'],'Sheet','pid')
    writetable(T1,['Results_clinical_700_',save_name,'.xlsx'],'Sheet','diagMM_time')

    if ~isempty(table2array(cell2table(MM_report)))
        writetable(cell2table(MM_report),...
            ['Results_clinical_700_',save_name,'.xlsx'],...
            'Sheet','MM','WriteVariableNames',0)
    end
    if ~isempty(table2array(cell2table(SMM_report)))
        writetable(cell2table(SMM_report),...
            ['Results_clinical_700_',save_name,'.xlsx'],...
            'Sheet','SMM','WriteVariableNames',0)
    end
    if ~isempty(table2array(cell2table(Treat_report)))
        writetable(cell2table(Treat_report),...
            ['Results_clinical_700_',save_name,'.xlsx'],...
            'Sheet','Treatment','WriteVariableNames',0)
    end

    writetable(array2table(MM_label),...
        ['Results_clinical_700_',save_name,'.xlsx'],...
        'Sheet','Prog_label','WriteVariableNames',0)        
    

%% 
disp('============ NLP ALGORITHM FINISHED ==============');
disp(sprintf('Total Run Time: %.2f seconds. Average %.2f seconds per patient',...
    TotalRunTime, TotalRunTime/length(true_train_label)));



