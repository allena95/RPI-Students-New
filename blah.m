%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('EditedNoCourses.xlsx');
num = num(~any(isnan(num),2),:); %remove students with missing data
data = num(:,2:end);
features = data;




%%


s=std(features);
a = diag(1./s);
[m,n] = size(features);
one_m = ones(m,m);

features = (features - (1/m)*(ones(m,m)*features))*a; 



figure
imagesc(corr(features))
title('KNN Confusion Matrix')
colorbar
