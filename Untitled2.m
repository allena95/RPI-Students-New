%% add the soc and hs gpa

[num,txt,raw] = xlsread('data with courses.xlsx');

[hsnum,hstxt,hsraw] = xlsread('DATUM HS GPA 201309.xlsx');

[socnum,soctxt,socraw] = xlsread('SOC Code Fall 2013 Freshmen.xlsx');

[m,n] = size(num)

hsnum = [hsnum' zeros(3,9)]';
socnum = [socnum' zeros(2,28)]';
%%
new_num = [num zeros(1385,2)];

for i = 1:m;
    for j = 1:m
        if new_num(i,1) == hsnum(j,1)
            new_num(i,:) = [num(i,:) hsnum(j,2:end)];
        end
    end
end

%%
num2 = [new_num zeros(1385,1)];

for i = 1:m;
    for j = 1:m
        if num2(i,1) == socnum(j,1);
            num2(i,:) = [new_num(i,:) socnum(j,2:end)];
        end
    end
end
%%
xlswrite('_data_.xlsx', num2(:,67:69))