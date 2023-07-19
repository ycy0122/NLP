function [ind_MGUS, ind_Prot, ind_MM, ind_SMM, ind_Treat] = ...
    featurizeALL(inputcellarray, refwordsMGUS, refwordsProtein, refwordsMM, refwordsSMM, refwordsTreat, refwordsDiseaseNeg, negwords)
% featureVector = featurize(inputcellarray)
%
% Takes an input cell array in which each cell is a review or text
% outputs:
%       ind_MGUS    :label for MGUS
%       ind_Pro     :label for igg protein
%       ind_MM      :label for MM (multiple myeloma)
%       ind_SMM     :label for Smoldering MM
%       ind_Treat   :label for MM Treatment
%
% Inputs:
%       inputcellarray      : a cell array with texts as the content of each cell
%       refwordsMGUS        : Reference words to identify MGUS
%       refwordsProtein     : Reference words to identify igg protein
%       refwordsMM          : Reference words to identify MM
%       refwordsSMM         : Reference words to identify SMM
%       refwordsTreat       : Reference words to identify MM Treatment
%       refwordsDiseaseNeg  : Reference words to identify diseases other
%                               than MM
%       negwords            : Reference words of negation
%

MGUS_label = zeros(size(inputcellarray,1),1);
Prot_label = zeros(size(inputcellarray,1),1);
MM_label = zeros(size(inputcellarray,1),1);
SMM_label = zeros(size(inputcellarray,1),1);
Treat_label = zeros(size(inputcellarray,1),1);

% add blank at start and end of the string
for i = 1:size(inputcellarray,1)
    inputcellarray{i} = strcat(" ", inputcellarray{i}," ");
end


for i = 1:size(inputcellarray,1)
    this_cell = inputcellarray{i};
    % examine if there is any word identifying MGUS
    indMGUS_tmp = [];
    indProt_tmp = [];
    indMM_tmp = [];
    indSMM_tmp = [];
    indTreat_tmp = [];
    indDiseaseNeg_tmp = [];
    indneg_tmp = [];
    
    for  j =  1:length(refwordsMGUS)
        this_ref = refwordsMGUS{j};
        indMGUS_tmp = [indMGUS_tmp, strfind( this_cell, this_ref)];
    end

    for  j =  1:length(refwordsProtein)
        this_ref = refwordsProtein{j};
        indProt_tmp = [indProt_tmp, strfind( this_cell, this_ref)];
    end
    
    for  j =  1:length(refwordsMM)
        this_ref = refwordsMM{j};
        indMM_tmp = [indMM_tmp, strfind( this_cell, this_ref)];
    end

    for  j =  1:length(refwordsSMM)
        this_ref = refwordsSMM{j};
        indSMM_tmp = [indSMM_tmp, strfind( this_cell, this_ref)];
    end
    
    for  j =  1:length(refwordsTreat)
        this_ref = refwordsTreat{j};
        indTreat_tmp = [indTreat_tmp, strfind( this_cell, this_ref)];
    end
    
    for k = 1:length(negwords)
        this_neg = negwords{k};
        indneg_tmp = [indneg_tmp, strfind( this_cell, this_neg)];        
    end

    for  l =  1:length(refwordsDiseaseNeg)
        this_ref = refwordsDiseaseNeg{l};
        indDiseaseNeg_tmp = [indDiseaseNeg_tmp, strfind( this_cell, this_ref)];
    end
    
    indexMGUS{i} = indMGUS_tmp;
    indexProt{i} = indProt_tmp;
    indexMM{i} = indMM_tmp;
    indexSMM{i} = indSMM_tmp;
    indexTreat{i} = indTreat_tmp;
    index_neg{i} = indneg_tmp;
    indexD{i} = indDiseaseNeg_tmp;
    
    [MGUS_label(i)] = subfeaturize(inputcellarray{i},indexMGUS{i},index_neg{i},indexD{i},negwords);
    [Prot_label(i)] = subfeaturize(inputcellarray{i},indexProt{i},index_neg{i},indexD{i},negwords);
    [MM_label(i)] = subfeaturize(inputcellarray{i},indexMM{i},index_neg{i},indexD{i},negwords);
    [SMM_label(i)] = subfeaturize(inputcellarray{i},indexSMM{i},index_neg{i},indexD{i},negwords);
    [Treat_label(i)] = subfeaturize(inputcellarray{i},indexTreat{i},index_neg{i},indexD{i},negwords);
end
ind_MGUS = any(MGUS_label);
ind_Prot = any(Prot_label);
ind_MM = any(MM_label);
ind_SMM = any(SMM_label);
ind_Treat = any(Treat_label);
end

