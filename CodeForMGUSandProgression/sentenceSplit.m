addpath(genpath('C:\Users\Yao-Chi\OneDrive - Washington University in St. Louis\Su-Hsin\NLP_Mei\Minighao&Yuchen\MATLAB\sentenceSplit0630\examples_to_debug'))

example = fopen('example8509.txt');
text = fgetl(example);
fclose(example);

global END offset overlap_checkers rule_cell rule_map;
END = '<END>';
offset = 0;
overlap_checkers = containers.Map;
[rule_map, rule_cell] = GetRuleMap('rush_rules.tsv');

out = SplitSentence(text);
for i = 1:length(out)
    begin_pos = out{i}.begin;
    end_pos = out{i}.end;
    text_pos = text(begin_pos:end_pos);
    show = [begin_pos, end_pos];
    disp(show)
    disp(text_pos)
end

function output = SplitSentence(text)
   % Split sentences in text.
   global rule_map
   
   auto_fix_gap = true;
   min_sent_chars = 5;

   Begin = 'stbegin';
   End = 'stend';
   output = {};
   matches = containers.Map;
   matches(Begin) = {};
   matches(End) = {};
   matches = process(text, rule_map, matches);
   begins = matches(Begin);
   ends = matches(End);
   st_begin = 1;
   st_started = false;
   st_end = 1;
   j = 1;
   for i = 1:length(begins)
       token = begins{i};
       if ~st_started
           st_begin = token.begin;
           if st_begin < st_end
               continue
           end
           st_started = true;
       elseif token.begin < st_end
           continue
       end
       %if auto_fix_gap & length(output) > 0 & st_begin > output{length(output)}.end
       %    output = fix_gap(output, text, output{length(output)}.end, st_begin, min_sent_chars); 
       %end
       for k = j:length(ends)
           if i < length(begins) & k < length(ends) & begins{i+1}.begin < (ends{k}.begin)+1
               break;
           end
           st_end = ends{k}.begin+1;
           j = k;
           while st_end > 1 & (text(st_end-1) == ' ' | double(text(st_end-1))==160)
               st_end = st_end-1;
           end
           if st_end < st_begin
               continue;
           elseif st_started
               output = [output, Span(st_begin, st_end, '', -1, -1)];
               st_started = false;
               
               if i == length(begins) | (k < length(ends)-1 & begins{i+1}.begin > ends{k+1}.end)
                   continue
               end
               break;
           else
               output{length(output)} = Span(st_begin, st_end, '', -1, -1);
               st_started = false;
           end
       end
   end

   if auto_fix_gap    
       if length(output)>0
           begin_trimed_gap = trim_gap(text, 1, output{1}.begin);
           if ~isempty(begin_trimed_gap)
               if output{1}.begin <= ends{1}.begin
                   output{1}.begin = begin_trimed_gap.begin;
               else
                   output = [{begin_trimed_gap}, output];      
               end
           end
           end_trimed_gap = trim_gap(text, output{length(output)}.end, length(text));
           if ~isempty(end_trimed_gap)
               if end_trimed_gap.width > min_sent_chars
                   output = [output, end_trimed_gap];       
               else
                   output{length(output)}.end = end_trimed_gap.end;
               end
           end
       else
           trimed_gap = trim_gap(text, 1, length(text));
           if ~isempty(trimed_gap) & trimed_gap.width > min_sent_chars
               output = [output, trimed_gap];
           end
       end
   end
end


function matches = process(text, rule_map, matches)
    %Returns a cell of matches Span grouped by types (keys).
    for i = 1:length(text)
        if i > 1
            pre_char = text(i - 1);
        else
            pre_char = " ";
        end
        processRules(text, rule_map, i, 1, i, matches, pre_char, false, " ");
    end
        matches = removePseudo(matches);
end

function matches = processRules(text, rule_map, match_begin, match_end, current_position, matches, pre_char, wildcard_support, pre_key)
    global END support_replication;
    len = length(text);
    if current_position <= len
        this_char = text(current_position);
        if isKey(rule_map, '\')
            matches = processWildCards(text, rule_map('\'), match_begin, match_end, current_position, matches, pre_char, '\');
        end
        if isKey(rule_map, '(') & pre_key ~= '\'
            matches = processRules(text, rule_map('('), current_position, match_end, current_position, matches, pre_char, false, '(');
        end
        if isKey(rule_map, ')') & pre_key ~= '\'
            matches = processRules(text, rule_map(')'), match_begin, current_position, current_position, matches, pre_char, false, '(');
        end
        if isKey(rule_map, END)
            matches = addDeterminants(text, rule_map, matches, match_begin, match_end, current_position);
        end
        if isKey(rule_map, this_char) & this_char ~= '(' & this_char ~= ')'
            matches = processRules(text, rule_map(this_char), match_begin, match_end, current_position + 1, matches, this_char, false, this_char);
        end
        if support_replication & isKey(rule_map, '+')
            processRules(text, rule_map('+'), match_begin, match_end, current_position, matches, this_char, false, '+');
            processReplication(text, rule_map('+'), match_begin, match_end, current_position, matches, this_char, wildcard_support, pre_key);
        end
    elseif current_position == len + 1 & isKey(rule_map, END)
        if match_end == 1
            matches = addDeterminants(text, rule_map, matches, match_begin, current_position, current_position);
        else
            matches = addDeterminants(text, rule_map, matches, match_begin, match_end, current_position);
        end
    elseif current_position == len + 1 & isKey(rule_map, '\') & isKey(rule_map('\'), 'e')
        deter_rule1 = rule_map('\');
        deter_rule2 = deter_rule1('e');
        if match_end == 1
            matches = addDeterminants(text, deter_rule2, matches, match_begin, current_position, current_position);
        else
            matches = addDeterminants(text, deter_rule2, matches, match_begin, match_end, current_position);
        end
    elseif current_position == len + 1 & isKey(rule_map, ')')
        deter_rule = rule_map(')');
        if isKey(deter_rule, END)
            matches = addDeterminants(text, deter_rule, matches, match_begin, current_position, current_position);
        elseif isKey(deter_rule, '\') & isKey(deter_rule, 'e')
            matches = processRules(text, deter_rule('\'), match_begin, current_position, current_position, matches, pre_char, false, ' ');
        end
    elseif current_position == len + 1 & isKey(rule_map, '+')
        processRules(text, rule_map('+'), match_begin, match_end, current_position, matches, pre_char, wildcard_support, pre_key);
    end
end

function matches = addDeterminants(text, deter_rule, matches, match_begin, match_end, current_position)
    global END offset overlap_checkers;
    deter_rule = deter_rule(END);
    if match_end == 1
        end_position = current_position;
    else
        end_position = match_end;
    end
    if match_begin > end_position
        t = match_begin;
        match_begin = end_position;
        end_position = t;
    end
    current_spans_list = {};
    deter_rule_keys = keys(deter_rule);
    for k = 1:length(deter_rule_keys)
        key = deter_rule_keys{k};
        rule_id = deter_rule(key);
        current_span = Span(match_begin + offset, end_position + offset, text(match_begin: end_position - 1), rule_id, NaN);
        if isKey(overlap_checkers, key)
            current_spans_list = matches(key);
            overlap_checker = overlap_checkers(key);
            pos = searchPos(overlap_checker, current_span.begin, current_span.end);
            update = 0;
            if pos > 0
                overlapped_span = current_spans_list{pos};
                if ~compareSpan(current_span, overlapped_span)
                    return
                end
                current_spans_list{pos} = current_span;
                for i = 1:length(overlap_checker)
                    interval = overlap_checker{i};
                    if interval.begin == current_span.begin & interval.end == current_span.end
                        interval.pos = pos;
                        update = 1;
                        break;
                    end
                end
                if update == 0
                    last_pos = length(overlap_checker) + 1;
                    overlap_checker{last_pos} = Interval(current_span.begin, current_span.end, pos);
                end
            else
                last_pos = length(overlap_checker) + 1;
                len = length(current_spans_list);
                current_spans_list{len + 1} = current_span;
                overlap_checker{last_pos} = Interval(current_span.begin, current_span.end, length(current_spans_list));
            end
            matches(key) = current_spans_list;
            overlap_checkers(key) = overlap_checker;
        else
            len = length(current_spans_list);
            current_spans_list{len + 1} = current_span;
            matches(key) = current_spans_list;
            overlap_checker = {};
            overlap_checker{1} = Interval(current_span.begin, current_span.end, length(current_spans_list));
            overlap_checkers(key) = overlap_checker;
        end
    end
end

function pos = searchPos(intervals, begin_pos, end_pos)
    for i = 1:length(intervals)
        interval = intervals{i};
        if begin_pos >= interval.begin & begin_pos < interval.end
            pos = interval.pos;
            return
        elseif end_pos >= interval.begin & end_pos < interval.end
            pos = interval.pos;
            return
        end
    end
    pos = -1;
end

function logic = compareSpan(a, b)
    logic = a.score < 0 | a.score > b.score | (a.score == b.score && a.width > b.width);
end

function matches = processReplication(text, rule_map, match_begin, match_end, current_position, matches, pre_char, wildcard_support, pre_key)
    this_char = text(current_position);
    if wildcard_support
        if pre_key == 's'
            matches = processReplication_s(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
        if pre_key == 'n'
            matches = processReplication_n(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
        if pre_key == 'd'
            matches = processReplication_d(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
        if pre_key == 'C'
            matches = processReplication_C(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
        if pre_key == 'c'
            matches = processReplication_c(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
        if pre_key == 'p'
            matches = processReplication_p(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
        if pre_key == 'a'
            matches = processReplication_a(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
        if pre_key == 'u'
            matches = processReplication_u(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
        if pre_key == 'w'
            matches = processReplication_w(text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
        end
    else
        logic = this_char == pre_key;
        matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, pre_char);
    end
end

function matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    max_repeat = 50;
    current_repeat = 0;
    text_length = length(text);
    while logic & current_repeat < max_repeat & current_position <= text_length
        current_repeat = current_repeat + 1;
        current_position = current_position + 1;
        if current_position > text_length
            break
        end
        this_char = text(current_position);
    end
    matches = processRules(text, rule_map, match_begin, match_end, current_position, matches, previous_char, false, '+');
end

function matches = processReplication_s(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    logic = (this_char == ' ' | double(this_char) == 9 | double(this_char) == 160);
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processReplication_n(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    logic = (double(this_char) == 10 | double(this_char) == 13);
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processReplication_d(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    logic = isnumeric(this_char);
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processReplication_C(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    logic = this_char == upper(this_char);
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processReplication_c(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    logic = this_char == lower(this_char);
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processReplication_p(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    list = ['!', '"', '#', '$', '%', '&', '', '()', '*' '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '\', "'"];
    logic = ismember(this_char, list);
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processReplication_a(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    logic = ~isspace(this_char);
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processReplication_u(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    logic = (this_char > '~' & double(this_char) ~= 160);
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processReplication_w(text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char)
    logic = (this_char > '~' | isspace(this_char));
    matches = processReplicationCommon(logic, text, rule_map, match_begin, match_end, current_position, matches, this_char, previous_char);
end

function matches = processWildCards(text, rule_map, match_begin, match_end, current_position, matches, pre_char, pre_key)
    this_char = text(current_position);
    rule_keys = keys(rule_map);
    for k = 1:length(rule_map)
        if rule_keys{k} == 's'
            matches = processWildCard_s(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'n'
            matches = processWildCard_n(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == '('
            matches = processWildCard_openParan(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == ')'
            matches = processWildCard_closeParan(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'd'
            matches = processWildCard_d(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'C'
            matches = processWildCard_C(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'c'
            matches = processWildCard_c(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'p'
            matches = processWildCard_p(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == '+'
            matches = processWildCard_plus(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == '\'
            matches = processWildCard_backSlash(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'b'
            matches = processWildCard_b(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'a'
            matches = processWildCard_a(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'u'
            matches = processWildCard_u(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'w'
            matches = processWildCard_w(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
        if rule_keys{k} == 'e'
            matches = processWildCard_e(text, rule_map, match_begin, match_end, current_position, matches, this_char);
        end
    end
end

function matches = processWildCard_s(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char == ' ' | double(this_char) == 9 | double(this_char) == 160
        matches = processRules(text, rule_map('s'), match_begin, match_end, current_position + 1, matches, this_char, true, 's');
    end
end

function matches = processWildCard_n(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if double(this_char) == 10 | double(this_char) == 13
        matches = processRules(text, rule_map('n'), match_begin, match_end, current_position + 1, matches, this_char, true, 'n');
    end
end

function matches = processWildCard_openParan(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char == '('
        matches = processRules(text, rule_map('('), match_begin, match_end, current_position + 1, matches, this_char, true, '(');
    end
end

function matches = processWildCard_closeParan(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char == ')'
        matches = processRules(text, rule_map(')'), match_begin, match_end, current_position + 1, matches, this_char, true, ')');
    end
end

function matches = processWildCard_d(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if isnumeric(this_char)
        matches = processRules(text, rule_map('d'), match_begin, match_end, current_position + 1, matches, this_char, true, 'd');
    end
end

function matches = processWildCard_C(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char == upper(this_char)
        matches = processRules(text, rule_map('C'), match_begin, match_end, current_position + 1, matches, this_char, true, 'C');
    end
end

function matches = processWildCard_c(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char == lower(this_char)
        matches = processRules(text, rule_map('c'), match_begin, match_end, current_position + 1, matches, this_char, true, 'c');
    end
end

function matches = processWildCard_p(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    list = ['!', '"', '#', '$', '%', '&', '', '()', '*' '+', ',', ' ', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', '\', "'"];
    if ismember(this_char, list)
        matches = processRules(text, rule_map('p'), match_begin, match_end, current_position + 1, matches, this_char, true, 'p');
    end
end

function matches = processWildCard_plus(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char == lower(this_char)
        matches = processRules(text, rule_map('c'), match_begin, match_end, current_position + 1, matches, this_char, true, 'c');
    end
end

function matches = processWildCard_backSlash(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char == '\'
        matches = processRules(text, rule_map('\'), match_begin, match_end, current_position + 1, matches, this_char, true, '\');
    end
end

function matches = processWildCard_a(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if isspace(this_char)
        matches = processRules(text, rule_map('a'), match_begin, match_end, current_position + 1, matches, this_char, true, 'a');
    end
end

function matches = processWildCard_u(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char > '~' & double(this_char) ~= 160
        matches = processRules(text, rule_map('u'), match_begin, match_end, current_position + 1, matches, this_char, true, 'u');
    end
end

function matches = processWildCard_w(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char > '~' | isspace(this_char)
        matches = processRules(text, rule_map('w'), match_begin, match_end, current_position + 1, matches, this_char, true, 'w');
    end
end

function matches = processWildCard_b(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if current_position == 0
        matches = processRules(text, rule_map('b'), match_begin, match_end, current_position + 1, matches, this_char, true, 'b');
    end
end

function matches = processWildCard_e(text, rule_map, match_begin, match_end, current_position, matches, this_char)
    if this_char == length(text) + 1
        matches = processRules(text, rule_map('e'), match_begin, match_end, current_position + 1, matches, this_char, true, 'e');
    end
end

function matches = removePseudo(matches)
    global rule_cell;
    matches_keys = keys(matches);
    for k = 1:length(matches_keys)
        key = matches_keys{k};
        spans_list = matches(key);
        new_spans_list = {};
        index = 1;
        for i = 1:length(spans_list)
            span = spans_list{i};
            for i = 1:length(rule_cell)
                rule_id = rule_cell{i}{1};
                if rule_id == span.rule_id
                    if rule_cell{i}{5} == 'ACTUAL'
                        new_spans_list{index} = span;
                        index = index + 1;
                    end
                    break;
                end
            end
        end
        matches(key) = new_spans_list;
    end
end

function sentence = fix_gap(sentence, text, previous_end, this_begin, min_sent_chars)
    trimed_gap = trim_gap(text, previous_end, this_begin);
    if trimed_gap == []
    elseif trimed_gap.width > min_sent_chars
        sentence = [sentence, trimed_gap];
    elseif ~isempty(sentence)
        len = length(sentence);
        sentence(len).end = trimed_gap.end;
    end
end

function gap = trim_gap(text, previous_end, this_begin)
    begin = 0;
    End = 1;
    gap_chars = text(previous_end:this_begin-1);
    CELL = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','Y','U','V','W','X','Y','Z','1','2','3','4','5','6','7','8','9','0'};
    for i = 1:this_begin-previous_end
        this_char = gap_chars(i);
        if this_char ~= ' '
            begin = i;
            break;
        end
    end
    for i = this_begin-previous_end:-1:begin+1
        this_char = gap_chars(i);
        if any(strcmp(CELL, this_char)) | this_char=='.' | this_char=='!' | this_char=='?' | this_char==')' | this_char==']' | this_char=='"'
            End = i;
            break;
        end
    end
    if End > begin & begin ~= 0
        gap = Span(previous_end+begin-1, previous_end+End, '', -1, -1);
    else
        gap = [];
    end
end


