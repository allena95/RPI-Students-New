
%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('DataWoutCat.xlsx');

num = num(~any(isnan(num),2),:); %remove students with missing data

data = num(:,2:end);
features = [data(:,34:36) data(:,39:end)];

features = [data(:,1:32) data(:,34:36) data(:,39:end) data(:,38)];
%features = [features(:,33:37) features(:,end)] %gpa only


s14 = data(:,33); %register labels spring 2014
f14 = data(:,37); %return labels f14
s15 = data(:,38); %return labels s15

%s15 = s15(~any(isnan(features),2),:);

%% Mean center and scale

% s=std(features);
% a = diag(1./s);
% [m,n] = size(features);
% one_m = ones(m,m);
% 
% features = (features - (1/m)*(ones(m,m)*features))*a; 

%% Define testing and training sets

% Training and testing matrices for DatasetA

% Classp_train  := Class 1 training data
% Classm_train  := Class -1 training data
% Classp_test   := Class 1 testing data
% Classm_test   := Class -1 testing  data


% Set random number to an initial seed
[r,c]=size(features);
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,r);
features=features(p,:);
Y=s15(p);
%Use trainpct percent of the data for training and the rest for testing.
trainpct=.75;
train_size=ceil(r*trainpct);

% Grab training and test data
Train = features(1:train_size,:);
Test = features(train_size+1:end,:);
% YTrain = Y(1:train_size,:);
% YTest = Y(train_size+1:end,:);
%%
% s=std(Train);
% a = diag(1./s);
% [m,n] = size(Train);
% one_m = ones(m,m);
% train_mean = (1/m)*(ones(1,m)*Train);
% 
% Train = (Train - ones(m,1)*train_mean)*a;
% %%
% [m_test,n_test] = size(Test);
% Test = (Test - ones(m_test,1)*train_mean)*a;


%%
Ensemble = fitensemble(Train(:,1:end-1),Train(:,end),'AdaBoostM1',30,'Discriminant')

[Y_c,score] = predict(Ensemble,Test(:,1:end-1));
posclass = 1
Y_t = Test(:,end);

[CX,CY,T,AUC] = perfcurve(Y_t,score(:,1),posclass);

EVAL = Evaluate(Y_t,Y_c);

figure
plot(CX,CY)
title('ROC Curve')


perror = 0;
merror = 0;
for i = 1:size(Y_t,1);
    if and(Y_t(i)~=Y_c(i),Y_t(i)==0);
        merror = merror +1;
    end
    if and(Y_t(i)==1,Y_t(i)~=Y_c(i));
        perror = perror +1;
    end
end
%% Trying Built in knnclassify
Class = knnclassify(Test(:,1:end-1),Train(:,1:end-1),Train(:,end)); 
perror = 0;
merror = 0;
for i = 1:size(Y_t,1);
    if and(Y_t(i)~=Class(i),Y_t(i)==0);
        merror = merror +1;
    end
    if and(Y_t(i)==1,Y_t(i)~=Class(i));
        perror = perror +1;
    end
end



% �\_(?)_/� neg error always so high






