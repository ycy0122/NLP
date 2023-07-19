function [ind_NotMM,indexC,index_neg] = featurizeMGUSC(inputcellarray, refwords, negwords)
% featureVector = featurize(inputcellarray)
%
% Takes an input cell array in which each cell is a review or text
% outputs
%
% Inputs:
%       inputcellarray: a cell array with texts as the content of each cell
%       refwords: a cell array of MM-related wordlist
%       negwords: negation modifiers to test if there is negation in front
%       of MM
% Outputs:
%       ind_NotMM:
%       indexC: location where no_MM appears in each sentence
%

NotMM_label = zeros(size(inputcellarray,1),1);

% add blank at start and end of the string
for i = 1:size(inputcellarray,1)
    inputcellarray{i} = strcat(" ", inputcellarray{i}," ");
end


for i = 1:size(inputcellarray,1)
    this_cell = inputcellarray{i};
    % examine if there is any word identifying MM
    indC_tmp = [];
    indneg_tmp = [];

    for  j =  1:length(refwords)
        this_ref = refwords{j};
        indC_tmp = [indC_tmp, strfind( this_cell, this_ref)];
    end
    
    for k = 1:length(negwords)
        this_neg = negwords{k};
        this_neg_ind =  strfind( this_cell, this_neg);
        indneg_tmp = [indneg_tmp, this_neg_ind];        
    end
    
    indexC{i} = indC_tmp;
    index_neg{i} = indneg_tmp;
%     index_other{i} = indother_tmp;
    
    if ~isempty(indexC{i}) && ~isempty(index_neg{i})
        NotMM_label(i) = 0;
        if all(index_neg{i} <  indexC{i}.')
            NotMM_label(i) = 1;
        else
            NotMM_label(i) = 0;
        end
    elseif ~isempty(indexC{i}) && isempty(index_neg{i})
        try str2num(char(extractBefore(this_cell,indexC{i})))
            % check if this is the start of sentence. if so, not MM
            if isempty(str2num(char(extractBefore(this_cell,indexC{i}))))
                NotMM_label(i) = 1;
            else
                NotMM_label(i) = 0;
            end
        catch
            NotMM_label(i) = 1;
        end
    else
        NotMM_label(i) = 1;
    end
    
%     if  ~isempty(index_other{i}) && isempty(index_neg{i})
%         MGUS_label(i) = 0;
%     end
end
ind_NotMM = all(NotMM_label);
% ind_NotMM = any(NotMM_label);
end

