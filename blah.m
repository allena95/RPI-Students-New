%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('DataWQ.xlsx');

data = num(:,2:end);

features = [data(:,1:33) data(:,35) data(:,37) data(:,41:end) data(:,34) data(:,38:39)];
features = features(~any(isnan(features),2),:); %remove students with missing data

s14 = features(:,end-2); %register labels spring 2014
f14 = features(:,end-1); %return labels f14
s15 = features(:,end); %return labels s15

features = features(:,1:end-3);

%%

[fisherror,Train,Test,w,t,perror,merror,error_total] = ...
classifier(features,s15,.75);