clc;clear;close all
%%

xlsxfile = ['NLP_700_clinical.xlsx'];
clinicaldata = readtable(xlsxfile, 'TextType','string','Sheet','Sheet1');

% process data:
reportcellarray ={};
reportsencellarray ={};
N_report = size(clinicaldata,1);
ReportTime = {};
tic
for i = 1:N_report            
    line = table2char(clinicaldata(i,2));
    time = table2char(clinicaldata(i,3));
    % comment: remove punctunation and numbers
    % comment2: remove numbers only
    text = strtrim(regexprep(line(end,:), ' +',' '));
    [comment, comment2] = SanitizeNote(text);
    ReportTime{end+1} = time(end,:);
    comment = lower(comment);
    r=regexp(comment,' ','split');
    % find comment with Mr., Ms., 
    comment2 = strrep(comment2,'Mr.','Mr');
    comment2 = strrep(comment2,'Ms.','Ms');
    comment2 = strrep(comment2,'Dr.','Dr');
    comment2 = strrep(comment2,'et al.','et al');
    comment2 = strrep(comment2,'vs.','vs');

    commentsentence = split(comment2,{'.',';'});

    for j = 1:size(commentsentence,1)
        idx = strfind(commentsentence{j}, '?');
        if ~isempty(idx)
            commentsentence{j} = commentsentence{j}(idx(end)+1:end);
        end
    end
    emptye_r_cell = cellfun(@isempty,r);    % remove empty cell
    r = r(~emptye_r_cell);

    disp(sprintf('%d out of %d', i, N_report));
    reportcellarray{end+1} = r;

    % remove the punctuation for sentences now
    comment2 = lower(commentsentence);

    for  j  = 1:size(comment2,1)
         [comment2{j},~] = SanitizeNote(comment2{j});
         r=regexp(comment2{j},' ','split');
         emptye_r_cell = cellfun(@isempty,r);    % remove empty cell
         r = r(~emptye_r_cell);
         comment2{j} = r;
    end

    emptye_c_cell = cellfun(@isempty,comment2);    % remove empty cell
    comment2 = comment2(~emptye_c_cell);  
    comment2 = cellfun(@strjoin,comment2,'UniformOutput',false);
    reportsencellarray{end+1} = comment2;

end
toc
reportcellarray = reportcellarray'; % 1 gram
reportcell = cellfun(@strjoin,reportcellarray,'UniformOutput',false);

save(['NLP_700_clinical_processed.mat'],'ReportTime','clinicaldata','reportsencellarray','N_report','reportcell')
%     decide where to split...
