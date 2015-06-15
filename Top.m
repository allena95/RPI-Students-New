%% DATUM RPI Students
clear;
close all;

%% Load in the data
[num,txt,raw] = xlsread('SortedByGPA.xlsx');
[m,n] = size(num);

data = num(1:1348,2:end);

top = data(1:337,:);
bottom = data(1011:end,:);

%%
xlswrite('Bottom.xlsx', bottom);
xlswrite('Top.xlsx', top);