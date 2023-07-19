function [ind_MM,indexA,index_neg] = featurizeMMA(inputcellarray, refwords, refwordsD, negwords)
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

MM_label = zeros(size(inputcellarray,1),1);

% fid = fopen('DiseaseReferenceSetC.txt');
% otherwords = {};
% while ~feof(fid)
%     buffer = fgetl(fid);
%     otherwords{end+1} = strcat(" ",buffer," ");
% end
% fclose(fid);

% add blank at start and end of the string
for i = 1:size(inputcellarray,1)
    inputcellarray{i} = strcat(" ", inputcellarray{i}," ");
end


for i = 1:size(inputcellarray,1)
    this_cell = inputcellarray{i};
    % examine if there is any word identifying MGUS
    indA_tmp = [];
    indneg_tmp = [];
    indD_tmp = [];
    
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
    
    for  l =  1:length(refwordsD)
        this_ref = refwordsD{l};
        indD_tmp = [indD_tmp, strfind( this_cell, this_ref)];
    end
    
    indexA{i} = indA_tmp;
    index_neg{i} = indneg_tmp;
    indexD{i} = indD_tmp;
    
    if ~isempty(indexA{i}) && ~isempty(index_neg{i})
        MM_label(i) = 0;
        %  make sure negation  words is in front of MGUS
%         if size(index_neg{i} < indexA{i}.',1) > 2
%             indexA{i}
%         end
        
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
                MM_label(i) = 0;
        elseif any(any(index_neg{i} <= indexA{i}.'))
            MM_label(i) = 0;
        elseif ~isempty(indexD{i})
            MM_label(i) = 0;
        else
            MM_label(i) = 1;
        end
    elseif ~isempty(indexA{i}) && isempty(index_neg{i}) && isempty(indexD{i})
        MM_label(i) = 1;
    else 
        MM_label(i) = 0;
    end
    
%     if  ~isempty(index_other{i}) && isempty(index_neg{i})
%         MGUS_label(i) = 0;
%     end
end
ind_MM = any(MM_label);
end

