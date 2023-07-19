function [ind_Pro, indexB, index_neg] = featurizeMGUSB(inputcellarray, refwords, negwords)
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

Prot_label = zeros(size(inputcellarray,1),1);

% add blank at start and end of the string
for i = 1:size(inputcellarray,1)
    inputcellarray{i} = strcat(" ", inputcellarray{i}," ");
end

for i = 1:size(inputcellarray,1)
    this_cell = inputcellarray{i};
    % examine if there is any word identifying MGUS
    indexB_tmp = [];
    indneg_tmp = [];
%     indexB{i} = find(ismember(inputcellarray,refwords));
    match_str= {};
    for  j =  1:length(refwords)
        match_str_id =  strfind(  this_cell, refwords{j});
%         match_str_id2 = ismember(inputcellarray{i,1}, refwords{j});
            % any(strcmp(inputcellarray{i,1}, refwords{j}))
%         if any(strcmp(inputcellarray{i,1}, refwords{j}))
%             match_str = [ match_str,refwords{j} ];
%         end
        if ~isempty(match_str_id)
            match_str{end+1} = [refwords{j} ];
        end
        indexB_tmp = [indexB_tmp,  match_str_id];
    end
    
    for k = 1:length(negwords)
        this_neg = negwords{k};
        this_neg_ind =  strfind( this_cell, this_neg);
        indneg_tmp = [indneg_tmp, this_neg_ind];        
    end
    
    indexB{i} = unique(indexB_tmp);
    index_neg{i} = indneg_tmp;

    if ~isempty(indexB{i}) && ~isempty(index_neg{i})
        Prot_label(i) = 0;
    elseif ~isempty(indexB{i}) && isempty(index_neg{i})
        Prot_label(i) = 1;
    else
        Prot_label(i) = 0;
    end
    
    
end
   
ind_Pro = any(Prot_label);
end


