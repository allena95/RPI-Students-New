%% DATUM RPI Students
clear;
close all;

%% Load in the data
[num,txt,raw] = xlsread('NoCoursesWithGPA.xlsx');

data = num(:,2:end); % remove survey number

data = data(~any(isnan(data),2),:); %remove students with missing data


%% Correlation matrix with all the data

cor = corr(data);


figure
imagesc(cor)
title('Correlation Matrix')
colorbar


