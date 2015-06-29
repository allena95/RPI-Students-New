%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%% Only Survey Data
[num2,txt2,raw2] = xlsread('NoCoursesWithGPA2.xlsx');

data2 = [num2(:,2:33) num2(:,46)];

data2 = data2(~any(isnan(data2),2),:); %remove students with missing data

features2 = [data2(:,1:end-1)];
label2 = data2(:,end);

[fisherror2,Train2,Test2,w2,t2,perror2,merror2,error_total2] = ...
classifier(features2,label2,.75);

%Fisher: 34% Train, 39% Test,  ~62% error on leave test
%KNN: 18.71% error, 84% leave error



[num,txt,raw] = xlsread('featurenamesSurvey.xlsx');
names = txt;
A = w2;
[m,n] = size(A);
%A = abs(A);
[I,B] = sort(abs(A),'descend');

for i = 1:m
    display(sprintf('Feature %d: %s   Score: %d',i, char(names(B(i))),A(B(i))))
end;



%% Only GPA
[num3,txt3,raw3] = xlsread('NoCoursesWithGPA.xlsx');

data3 = [num3(:,45:47) num3(:,50:51) num3(:,49)];

data3 = data3(~any(isnan(data3),2),:); %remove students with missing data

features3 = data3(:,1:end-1);
label3 = data3(:,end);

[fisherror3,Train3,Test3,w3,t3,perror3,merror3,error_total3] = ...
classifier(features3,label3,.75);

%Fisher: 17% error Train, 17% error Test, 50% leave test error
%KNN: 5.5% error, 60% leave error

[num,txt,raw] = xlsread('featurenamesGPA.xlsx');
names = txt;
A = w3;
[m,n] = size(A);
%A = abs(A);
[I,B] = sort(abs(A),'descend');

for i = 1:m
    display(sprintf('Feature %d: %s   Score: %d',i, char(names(B(i))),A(B(i))))
end;















