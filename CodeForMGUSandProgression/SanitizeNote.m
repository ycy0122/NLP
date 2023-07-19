function [out1, out2] = SanitizeNote(in1)
    % clears a string from puctuations
	
% 	start_expression = ':';
%    [startCell, startIndex] = regexp(char(in1), start_expression, 'match');
   
    
    myoutstring =regexprep(char(in1), '[^\w\s]', ' ');
% 	myoutstring =regexprep(myoutstring, '[\d]', ' '); %removes numbers 
% 	myoutstring =regexprep(myoutstring, '_', ' '); %removes numbers 
	
	% the following part takes care of \n in MATLAB
	% MATLAB's regex has a bug and could not handle these two cases
	
% 	myoutstring(myoutstring==13)=[];
% 	myoutstring(myoutstring==10)=[];

%     idx = strfind(char(in1), '?');
%     if  ~isempty(idx)
%         idx
%     end

%     myoutstring2 =regexprep(char(in1), '[\d]', ' '); %removes numbers 
%     myoutstring2 =regexprep(myoutstring2, '_', ' '); %removes numbers
    
    
    out1 =  myoutstring;
    out2 =  char(in1);
    