clc;clear;close all
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei'))
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_code\MATLAB\NLP_MGUS'))
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Yao-Chi\cohort1'))
read_reflists = 0;

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
        refwordsMGUSA{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMGUSA);    % remove empty cell
    refwordsMGUSA = refwordsMGUSA(~emptye_r_cell);
    fclose(fid);
    
    fid = fopen('DiseaseReferenceSetB.txt');
    refwordsMGUSB = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMGUSB{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMGUSB);    % remove empty cell
    refwordsMGUSB = refwordsMGUSB(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetC.txt');
    refwordsMGUSC = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMGUSC{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMGUSC);    % remove empty cell
    refwordsMGUSC = refwordsMGUSC(~emptye_r_cell);
    fclose(fid);

    % MM word lists
    fid = fopen('DiseaseReferenceSetMMA.txt');
    refwordsMMA = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMMA{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMMA);    % remove empty cell
    refwordsMMA = refwordsMMA(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetMMB.txt');
    refwordsMMB = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMMB{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMMB);    % remove empty cell
    refwordsMMB = refwordsMMB(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetMMC.txt');
    refwordsMMC = {};
    while ~feof(fid)
        buffer = fgetl(fid);
        refwordsMMC{end+1} = strcat(" ",buffer," ");
    end
    emptye_r_cell = cellfun(@isempty,refwordsMMC);    % remove empty cell
    refwordsMMC = refwordsMMC(~emptye_r_cell);
    fclose(fid);

    fid = fopen('DiseaseReferenceSetMMD.txt');
    refwordsMMD = {};
    while ~feof(fid)
        buffer = fgetl(fid);
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
    end
    emptye_r_cell = cellfun(@isempty,refwordsMME);    % remove empty cell
    refwordsMME = refwordsMME(~emptye_r_cell);
    fclose(fid);
    

    save('ReflistsMGUS.mat','refwordsMGUSA','refwordsMGUSB','refwordsMGUSC','negwords')
    save('ReflistsMM.mat','refwordsMMA','refwordsMMB','refwordsMMC','refwordsMMD','refwordsMME','negwords')
else
    load('ReflistsMGUS.mat','refwordsMGUSA','refwordsMGUSB','refwordsMGUSC','negwords')
    load('ReflistsMM.mat','refwordsMMA','refwordsMMB','refwordsMMC','refwordsMMD','refwordsMME','negwords')
    load('Reflists.mat','refwordsC')
end

%% get the previous 400 cohort
load(['SAMPLE1_400_PTS_TRAINING_CLINICAL.mat'],'clinicaldata')
[pid, ic, ia] = unique(table2array(clinicaldata(:,1)));
N_report = size(clinicaldata,1);

% read 700 new patients and get the repetitive ones
load(['NLP_700_clinical_processed.mat'],'ReportTime','clinicaldata','reportsencellarray','N_report','reportcell')
ReportTime = strtrim(ReportTime);
[pid_new, ic, ia] = intersect(pid,unique(table2array(clinicaldata(:,1))));

% get the clinicaldata and time
[keep_id1,keep_id2] = ismember(table2array(clinicaldata(:,1)),pid_new);
% update 
clinicaldata = clinicaldata(keep_id1,:);
ReportTime = ReportTime(1,keep_id1);  
reportcell = reportcell(keep_id1,1);
reportsencellarray = reportsencellarray(1,keep_id1);
N_report = size(clinicaldata,1);

%% universal data & variables
addpath(genpath('P:\ORD_Chang_202011003D\Yao-Chi\NLP_data\cohort1_500'))
mspike_filename = "SAMPLE1_400_PTS_TRAINING_MSPIKE.xlsx";
mspike = readtable(mspike_filename);
MK_PatientSSN = str2double(mspike.PatientSSN);
M_Spike = mspike.mspike;
ind1mspike = M_Spike<3;
mspike_date = mspike.LabChemCompleteDate;
ind_mspike_mgus = unique(MK_PatientSSN(M_Spike<3));
ind_mspike_mm = unique(MK_PatientSSN(M_Spike>=3));
ind2mspike = M_Spike>=3;
mspike_date = datetime(mspike_date);


%%    
for i = 1:N_report
    disp(sprintf('progress: %d out of %d reports', i, N_report));
    if ~isempty(reportsencellarray{1,i})
        [ind_MGUS(i),ind_protein(i),ind_MM(i),ind_SMM(i),ind_Treat(i)] = ...
            featurizeALL(reportsencellarray{1,i}, refwordsMGUSA, refwordsMGUSB, refwordsMMC, refwordsMMA, refwordsMME,  refwordsMMD, negwords);
        [ind_NotMM(i),~,~] = featurizeC(reportsencellarray{1,i}, refwordsC, negwords);


    else
        ind_MGUS(i) = 0;
        ind_protein(i) = 0;
        ind_NotMM(i) = 0;
        ind_SMM(i) = 0;
        ind_MM(i) = 0;
        ind_Treat(i) = 0;
    end
end

%% condiions:
% C1, C2, C3, C1+C2, C1+C3, C1+C2+C3,  C2+C3, C2+C4, C2+C3+C4, C3+C4
ConditionMGUS_matrix = zeros(length(pid),3);

MGUS_set1 = [];
MGUS_report = cell(length(pid),1);
Protein_report = cell(length(pid),1);
NotMM_report = cell(length(pid),1);

for i = 1:length(pid)
    MSPK_time = [];  
    MM_time = []; SMM_time = [];

    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    this_MGUS = any(ind_MGUS(this_pid) == 1);           %  C1
    this_PROP = any(ind_protein(this_pid) == 1);        %  C2
    this_MSPK = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mgus)); 
    this_NoMM = any(ind_NotMM(this_pid) == 1);          %  C4

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
end


combsMGUS = dec2base(0:power(2,3)-1,2) - '0';
combsMGUS = combsMGUS(2:end,:);

%% specificity + sensitivity
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei\True'))

train_true_filename = "TRUE_MGUSPROG.xlsx";
true_train_label = readtable(train_true_filename);
true_train_label = table2array(true_train_label(ia,4));

MGUS_set1 = [];
MGUS_set1 = any(ConditionMGUS_matrix(:,logical(combsMGUS(end,:))),2);
MGUS_label = ismember(pid,(pid(MGUS_set1)));

%% use best_com_ss again

classify_label = ismember(pid,(pid(MGUS_set1)));
sensitivity = sum(classify_label == 1 & true_train_label == 1)./ ( sum(classify_label == 1 & true_train_label == 1) +  sum(classify_label == 0 & true_train_label == 1))
specificity = sum(classify_label == 0 & true_train_label == 0)./ ( sum(classify_label == 0 & true_train_label == 0) +  sum(classify_label == 1 & true_train_label == 0))
PPV =  sum(classify_label == 1 & true_train_label == 1)./ ( sum(classify_label == 1 & true_train_label == 1) +  sum(classify_label == 1 & true_train_label == 0))
NPV =  sum(classify_label == 0 & true_train_label == 0)./ ( sum(classify_label == 0 & true_train_label == 0) +  sum(classify_label == 0 & true_train_label == 1))
Accuracy = sum(classify_label==true_train_label)/numel(true_train_label)
F1_score = 2/(1/sensitivity + 1/PPV)

%% old label
addpath(genpath('P:\ORD_Chang_202011003D\Mei\NLP\NLP_manuscript_700\Mei\True'))

train_true_filename = "TRUE_400_PTS_TRAINING.xlsx";
true_train_label2 = readtable(train_true_filename);
true_train_label2 = table2array(true_train_label2(:,2));

classify_label = ismember(pid,(pid(MGUS_set1)));
sensitivity2 = sum(classify_label == 1 & true_train_label2 == 1)./ ( sum(classify_label == 1 & true_train_label2 == 1) +  sum(classify_label == 0 & true_train_label2 == 1));
specificity2 = sum(classify_label == 0 & true_train_label2 == 0)./ ( sum(classify_label == 0 & true_train_label2 == 0) +  sum(classify_label == 1 & true_train_label2 == 0));
PPV2 =  sum(classify_label == 1 & true_train_label2 == 1)./ ( sum(classify_label == 1 & true_train_label2 == 1) +  sum(classify_label == 1 & true_train_label2 == 0));
NPV2 =  sum(classify_label == 0 & true_train_label2 == 0)./ ( sum(classify_label == 0 & true_train_label2 == 0) +  sum(classify_label == 0 & true_train_label2 == 1));
Accuracy2 = sum(classify_label==true_train_label2)/numel(true_train_label2);
F1_score2 = 2/(1/sensitivity2 + 1/PPV2);


%% find the earliest time of confiremed MGUS
Report_matrix_MGUS =  zeros(length(table2array(clinicaldata)),3);
for i = 1:length(pid)
    this_pid = find(table2array(clinicaldata(:,1))== pid(i));
    
    this_MGUS = any(ind_MGUS(this_pid) == 1);           %  C1
    this_PROP = any(ind_protein(this_pid) == 1);        %  C2
    this_MSPK = any(intersect(table2array(clinicaldata(this_pid(1),1)),ind_mspike_mgus)); 
    this_NoMM = any(ind_NotMM(this_pid) == 1);          %  C4

        % C1
        Report_matrix_MGUS(i,1) = (this_PROP ~= 0 && this_NoMM~=0);
        % C2
        Report_matrix_MGUS(i,2) =  (this_MSPK~=0 && this_NoMM~=0);
        % C3 
        Report_matrix_MGUS(i,3) = (this_MGUS ~= 0 && this_PROP~=0);
end

MGUS_id_vec = any(Report_matrix_MGUS(:,logical(combs(end,:))),2);

%%
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

%% write as excel sheet
T_pid = array2table(pid);
TMGUS = cell2table(earliest_timeMGUS.');

%%
writetable(T_pid,['Results_SAMPLE1_400_PTS_TRAINING_new.xlsx'],'Sheet','pid')
writetable(TMGUS,['Results_SAMPLE1_400_PTS_TRAINING_new.xlsx'],'Sheet','diagMGUS_time')

if ~isempty(table2array(cell2table(MGUS_report)))
    writetable(cell2table(MGUS_report),...
        ['Results_SAMPLE1_400_PTS_TRAINING_new.xlsx'],...
        'Sheet','MGUS','WriteVariableNames',0)
end
if ~isempty(table2array(cell2table(NotMM_report)))
    writetable(cell2table(NotMM_report),...
        ['Results_SAMPLE1_400_PTS_TRAINING_new.xlsx'],...
        'Sheet','NotMM','WriteVariableNames',0)
end
if ~isempty(table2array(cell2table(Protein_report)))
    writetable(cell2table(Protein_report),...
        ['Results_SAMPLE1_400_PTS_TRAINING_new.xlsx'],...
        'Sheet','Protein','WriteVariableNames',0)
end
if ~isempty(table2array(cell2table(mspk_report)))
    writetable(cell2table(mspk_report),...
        ['Results_SAMPLE1_400_PTS_TRAINING_new.xlsx'],...
        'Sheet','mspike','WriteVariableNames',0)
end
writetable(array2table(MGUS_label),...
    ['Results_SAMPLE1_400_PTS_TRAINING_new.xlsx'],...
    'Sheet','MGUS_label','WriteVariableNames',0)

        
    %% save results...
    save(['SAMPLE1_400_PTS_TRAINING_results_new.mat'],...
        'ind_mspike_mgus',...
        'earliest_timeMGUS','MGUS_label','ind_MGUS','ind_protein','ind_NotMM')






