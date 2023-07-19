function [ind_MGUS,indexA,index_neg] = featurizeMGUSA(inputcellarray, refwords, negwords)
% featureVector = featurize(inputcellarray)
%
% Takes an input cell array in which each cell is a review or text
% outputs
%
% Inputs:
%       inputcellarray: a cell array with texts as the content of each cell
%       nFeatures: the number of features that we like to see in the vetor
%       removeStopWords: if ==1 it will remove all the stop words
%       doStem: a flag if true porter stemmer will be used
% Outputs:
%       match_count:
%       match_index:
%

MGUS_label = zeros(size(inputcellarray,1),1);

% add blank at start and end of the string
for i = 1:size(inputcellarray,1)
    inputcellarray{i} = strcat(" ", inputcellarray{i}," ");
end


for i = 1:size(inputcellarray,1)
    this_cell = inputcellarray{i};
    % examine if there is any word identifying MGUS
    indA_tmp = [];
    indneg_tmp = [];
%     indother_tmp = [];
    
%     [Year{i}, Month{i}, Hh{i}, YourTable{i}] = readdates(inputcellarray{i,1});
    
    for  j =  1:length(refwords)
        this_ref = refwords{j};
        indA_tmp = [indA_tmp, strfind( this_cell, this_ref)];
    end
    
    for k = 1:length(negwords)
        this_neg = negwords{k};
        this_neg_ind =  strfind( this_cell, this_neg);
        indneg_tmp = [indneg_tmp, this_neg_ind];        
    end
    

    indexA{i} = indA_tmp;
    index_neg{i} = indneg_tmp;
%     index_other{i} = indother_tmp;
    
    if ~isempty(indexA{i}) && ~isempty(index_neg{i})
        MGUS_label(i) = 0;

        if all(index_neg{i} < indexA{i}.')
            MGUS_label(i) = 0;
        else
            MGUS_label(i) = 1;
        end
    elseif ~isempty(indexA{i}) && isempty(index_neg{i})
        MGUS_label(i) = 1;
    else
        MGUS_label(i) = 0;
    end
    
%     if  ~isempty(index_other{i}) && isempty(index_neg{i})
%         MGUS_label(i) = 0;
%     end
end
ind_MGUS = any(MGUS_label);
end

