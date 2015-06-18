%% DATUM RPI Students - Predict leaving Fall 2014
clear;
close all;

%% 

[num,txt,raw] = xlsread('NoCoursesWithGPA.xlsx');

want = [num(:,2:36) num(:,43) num(:,45) num(:,47) num(:,52:53) num(:,48)];

want = want(~any(isnan(want),2),:); %remove students with missing data


features = want(:,1:end-1);
features = features(:,37:40);

labels = (want(:,end));







%%
[fisherror,Train,Test,w,t,perror,merror,error_total] = ...
    classifier(features,labels,.75);
