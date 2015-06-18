%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;
%% Top 15 features
[num1,txt1,raw1] = xlsread('asdf.xlsx');
[fisherror1,Train1,Test1,w1,t1,perror1,merror1,error_total1] = ...
    classifier(num1(:,1:end-1),num1(:,end),.75);

%Fisher: 38.6% Train, 39.3% Test with 69% error on leave test
%KNN: 10% error, 92% leave error
%% Only Survey Data
[num2,txt2,raw2] = xlsread('NoCoursesWithGPA.xlsx');

data2 = [num2(:,2:36) num2(:,43) num2(:,49)];

data2 = data2(~any(isnan(data2),2),:); %remove students with missing data

features2 = [data2(:,1:end-1)];
label2 = data2(:,end);

[fisherror2,Train2,Test2,w2,t2,perror2,merror2,error_total2] = ...
classifier(features2,label2,.75);

%Fisher: 35% Train, 36% Test, only ~40% error on leave test
%KNN: 20% error, 88% leave error

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

%%















