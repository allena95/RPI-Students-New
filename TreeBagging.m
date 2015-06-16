
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
trainpct=.90;
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

%% Tree Bagging

% trees = true;                       
% while trees
%     
%     NTrees = round(rand(1)*1000); 
%     if NTrees ~= 0;                 %Generate a random integer between 0 and
%         trees = false;              %1000 that is not equal to 0
%     end
% end
% 
% vary = true;
% while vary
%     
%     vars = round(rand(1)*39); %Generate a random integer between 0 and 39
%     if vars ~= 0;            %that is not equal to 0
%         vary = false;
%     end
% end

NTrees = 35;
vars = 10


mdl =  TreeBagger(NTrees, Train(:,1:end-1),Train(:,end),'NVarToSample', vars);

[Y_c,score] = predict(mdl,Test(:,1:end-1));

Y_t = Test(:,end);

Y_c = str2double(Y_c);

%EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
EVAL = Evaluate(Y_t,Y_c);

posclass = 1;

[CX,CY,T,AUC] = perfcurve(Y_t,score(:,1),posclass);

txtv = strcat('ROC Curve ', num2str(vars), ' ', 'Variables', num2str(NTrees), 'Trees');

figure
plot(CX,CY)
title(txtv)

AUC
EVAL(1)
EVAL(end-1)

