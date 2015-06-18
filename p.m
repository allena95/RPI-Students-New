[num,txt,raw] = xlsread('asdf.xlsx');

f = num(:,1:end-1);
lab = num(:,end);

%%
[fisherror,w,t,perror_percent,merror_percent,error_total] = ...
    classifier(f,lab,.75);