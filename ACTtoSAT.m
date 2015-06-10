%% DATUM RPI Students
clear;
close all;

%% Load in the data
[num,txt,raw] = xlsread('NoCourses.xlsx');

%% SAT and ACT score convertion in num part

[m,n] = size(num);
for i = 1:m;
    if num(i,46) == 22
        num(i,45) = 1030;
    elseif num(i,46) == 23
        num(i,45) = 1070;
    elseif num(i,46) == 24
        num(i,45) = 1110;
    elseif num(i,46) == 25
        num(i,45) = 1150;
    elseif num(i,46) == 26
        num(i,45) = 1220;
    elseif num(i,46) == 27
        num(i,45) = 1220; 
    elseif num(i,46) == 28
        num(i,45) = 1260;
    elseif num(i,46) == 29
        num(i,45) = 1300;
    elseif num(i,46) == 30
        num(i,45) = 1340;
    elseif num(i,46) == 31
        num(i,45) = 1380;
    elseif num(i,46) == 32
        num(i,45) = 1420; 
    elseif num(i,46) == 33
        num(i,45) = 1460;
    elseif num(i,46) == 34
        num(i,45) = 1510; 
    elseif num(i,46) == 35
        num(i,45) = 1560;
    elseif num(i,46) == 36
        num(i,45) = 1600;
    end
end

xlswrite('NoCourses.xlsx',num(:,45),'AS2:AS1386')