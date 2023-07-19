function [ind_Treat,indexE,index_neg] = featurizeTreatE(inputcellarray, refwords, refwordsD, negwords)
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

Treat_label = zeros(size(inputcellarray,1),1);

% add blank at start and end of the string
for i = 1:size(inputcellarray,1)
    inputcellarray{i} = strcat(" ", inputcellarray{i}," ");
end

for i = 1:size(inputcellarray,1)
    this_cell = inputcellarray{i};
    % examine if there is any word identifying MGUS
    indE_tmp = [];
    indneg_tmp = [];
    indD_tmp = [];
    
%     [Year{i}, Month{i}, Hh{i}, YourTable{i}] = readdates(inputcellarray{i,1});
    
    for  j =  1:length(refwords)
        this_ref = refwords{j};
        indE_tmp = [indE_tmp, strfind( this_cell, this_ref)];
    end
    
    for k = 1:length(negwords)
        this_neg = negwords{k};
        this_neg_ind =  strfind( this_cell, this_neg);
        indneg_tmp = [indneg_tmp, this_neg_ind];        
    end
    
    for  l =  1:length(refwordsD)
        this_ref = refwordsD{l};
        indD_tmp = [indD_tmp, strfind( this_cell, this_ref)];
    end

    indexE{i} = indE_tmp;
    index_neg{i} = indneg_tmp;
    indexD{i} = indD_tmp;
    
    if ~isempty(indexE{i}) && ~isempty(index_neg{i})
        Treat_label(i) = 0;
        %  make sure negation  words is in front of MGUS
        if (~isempty(strfind(inputcellarray{i},negwords{68}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{69}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{70}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{71}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{72}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{73}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{74}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{75}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{76}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{77}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{78}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{79}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{80}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{81}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{82}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{83}))) || ...
                    (~isempty(strfind(inputcellarray{i},negwords{84})))
                Treat_label(i) = 0;
            
        elseif any(any(index_neg{i} < indexE{i}.'))
            Treat_label(i) = 0;
        elseif ~isempty(indexD{i})
            Treat_label(i) = 0;
        else
            Treat_label(i) = 1;
        end
    elseif ~isempty(indexE{i}) && isempty(index_neg{i}) && isempty(indexD{i})
        Treat_label(i) = 1;
    else 
        Treat_label(i) = 0;
    end
    
%     if  ~isempty(index_other{i}) && isempty(index_neg{i})
%         MGUS_label(i) = 0;
%     end
end
ind_Treat = any(Treat_label);
% ind_NotMM = any(NotMM_label);
end

