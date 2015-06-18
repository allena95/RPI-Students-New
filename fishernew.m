%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('NoCoursesWithGPA.xlsx');

%num = num(~any(isnan(num),2),:); %remove students with missing data

data = [num(:,2:43) num(:,45) num(:,47) num(:,51) num(:,49)]; 
data = data(~any(isnan(data),2),:); %remove students with missing data

features = data(:,1:end-1);



%features = features(:,33:37) %gpa only
%features = features(:,21);
%features = features(:,32:end);
%features = [features(:,1:32) features(:,35:end)];
%features = features(~any(isnan(features),2),:);

%survey = features(:,1:31);

s14 = data(:,33); %register labels spring 2014
f14 = data(:,37); %return labels f14
s15 = data(:,end); %return labels s15
%%

[fisherror,w,t,perror_percent,merror_percent,error_total] = ...
    classifier(features,s15,.75);
