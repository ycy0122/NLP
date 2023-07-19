function [Outputtable] = fread_html_reports( fin_name )

fid = fopen(fin_name,'rt');

fout_name = [];
stop_signal = 0;
Outputtable = cell(1,3);

table_count = 1;
report_count = 1;

while stop_signal == 0
    
    stuff_inbetween = [];
    
    line = fgetl(fid);
    
    if line == -1
        stop_signal = 1;
    end
    
    starts = strfind(line,'<td class="l data"');
    stops = strfind(line,'</td>');

    tstarts = strfind(line,'<td class="r data"');
    tstops = strfind(line,'</td>');
    
    
    if ~isempty(starts) && ~isempty(stops) % start and stop in the same line
        stuff_inbetween = line(starts+19:stops-1);
        if mod(table_count,3) == 1 
            Outputtable{report_count,1} = str2double(stuff_inbetween);
        end
        table_count = table_count + 1;
    elseif ~isempty(starts) && isempty(stops) % start and stop in different lines
        
        stops = strfind(line,'</td>');
        while isempty(stops)        
            if isempty(stuff_inbetween) && ~isempty(strfind(line,'<td class="l data"')) && ~isempty(line(starts+19:end)) % first: read starting line
                stuff_inbetween = line(starts+19:end);
                
            elseif isempty(stuff_inbetween) && isempty(strfind(line,'<td class="l data"')) && ~isempty(line(starts+19:end))
                stuff_inbetween = line(starts:end);
            elseif ~isempty(stuff_inbetween) % second: read following text

                line = fgetl(fid);
                line = regexprep(line,'[&][#][0-9][0-9][;]|[&][#][0-9][0-9][0-9][;]','');
                stops = strfind(line,'</td>');
                if isempty(stops)
                    stuff_inbetween_tmp = line(starts:end);
                else
                    stuff_inbetween_tmp = line(starts:stops-1);
                end
                stuff_inbetween = [stuff_inbetween,stuff_inbetween_tmp];

            elseif isempty(stuff_inbetween) && isempty(line(starts+19:end))  % empty after starting point...
                line = fgetl(fid);
                line = regexprep(line,'[&][#][0-9][0-9][;]|[&][#][0-9][0-9][0-9][;]','');
                while isempty(line(starts+19:end))
                    line = fgetl(fid);
                    stuff_inbetween = line(starts:end);
                end
            end
        end 
        
        if mod(table_count,3) == 2
            the_string = stuff_inbetween;
            Outputtable{report_count,2} = the_string;
        end
        table_count = table_count + 1;
        
    elseif ~isempty(tstarts) && ~isempty(tstops)
        stuff_inbetween = line(tstarts+19:tstops-1);
        if mod(table_count,3) == 0
            try
            Outputtable{report_count,3} = ...
                datestr(datetime(datenum(stuff_inbetween,'ddmmmyy'),...
                'ConvertFrom','datenum','Format','dd-MMM-yy'));
            catch
                Outputtable{report_count,3} = [];
            end
            report_count = report_count + 1
            table_count = table_count + 1;
        end
    end

%     starts = strfind(line,'<');
%     stops = strfind(line,'>');
%     if length(starts) == 1 & length(stops) == 1
%         line = [];
%     end
%     
%     if ~isempty(fout_name)
%         fid_out = fopen(fout_name,'a');
%         if ~isempty(line) & line ~= -1
%             fprintf(fid_out,[line,'\n']);
%         end
%         fclose(fid_out);
%     end

end
Outputtable = cell2table(Outputtable);
Outputtable.Properties.VariableNames = {'PattientSSN','ReportText','EpisodeBeginDate'};
end