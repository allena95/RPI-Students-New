%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
% [num,txt,raw] = xlsread('DataWoutCat.xlsx');
% 
% num = num(~any(isnan(num),2),:); %remove students with missing data
% 
% data = num(:,2:end);
% features = [data(:,34:36) data(:,39:end)];
% 
% features = [data(:,1:32) data(:,34:36) data(:,39:end)];
% 
% s14 = data(:,33); %register labels spring 2014
% f14 = data(:,37); %return labels f14
% s15 = data(:,38); %return labels s15

[num3,txt3,raw3] = xlsread('NoCoursesWithGPA.xlsx');

data3 = [num3(:,45:47) num3(:,50:51) num3(:,49)];

data3 = data3(~any(isnan(data3),2),:); %remove students with missing data

features = data3(:,1:end-1);
s15 = data3(:,end);


%%
% classm = features(s15==0,:);
% classp = features(s15==1,:);
% [m,n] = size(classm);
% 
% %%
% classm = [classm;classm]; %duplicate the negative class
% %classm = [classm;classm;classm;classm;classm;classm;classm;classm];
% 
% classp = [classp ones(size(classp,1),1)];
% classm = [classm zeros(size(classm,1),1)];
% 
% new = [classp; classm];
% new_features = new(:,1:39);
% new_labels = new(:,40);
%%
[fisherror,train,test,w,t,perror_percent,merror_percent,error_total] = ...
    classifier(features,s15,.75);
%% Double Negative Testing Class

feat_test = test(:,1:end-1);
label_test = test(:,end);

neg_feat = feat_test(label_test==0,:);

neg_feat = [neg_feat;neg_feat];
pos_feat = feat_test(label_test==1,:);

train_labels = train(:,end);

Test = [pos_feat;neg_feat];

label_test = [ones(size(pos_feat,1),1);zeros(size(neg_feat,1),1)];

Train = train(:,1:end-1);

total = [Test;Train];

labels = [label_test;train_labels];


[fisherror,train,test,w,t,perror_percent,merror_percent,error_total] = ...
    classifier(total,labels,.75);

%%














