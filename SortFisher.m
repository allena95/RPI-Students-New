%% DATUM RPI Students
clear;
close all;

%% Load in the data
[num_train,txt_train,raw_train] = xlsread('train.xlsx');
[num_test,txt_test,raw_test] = xlsread('test.xlsx');

% Get rid of NaN in each cell
num_train = num_train(~any(isnan(num_train),2),:);
num_test = num_test(~any(isnan(num_test),2),:);

YTrain = num_train(:,49);
YTest = num_test(:,49);

% Get the features out
features_train = num_train(:,[2:20,25:36,43,45:47,50:53]);
features_test = num_test(:,[2:20,25:36,43,45:47,50:53]);


%%
%Break them up into Class 1 and Class 0
Classp_train = features_train(YTrain==1,:);
Classm_train = features_train(YTrain==0,:);

Classp_test = features_test(YTest==1,:);
Classm_test = features_test(YTest==0,:);

%% Run Principle Components Analysis
sorted_train = [Classp_train;Classm_train];

[eigenvectors, score, eigenvalues]= pca(sorted_train);

train_N = size(features_train,1) 
img_mean = mean(features_train);
B = features_train- ones(train_N,1)*img_mean; % B is mean-centered training data
score_train = B*eigenvectors;

score_trainp = score_train(1:size(Classp_train,1),:);
score_trainm = score_train(size(Classp_train,1)+1:end,:);

%% Fisher Method on Training Set 

meanp=mean(score_trainp);
meanm=mean(score_trainm);
w=(meanp-meanm)';
w=w/norm(w);

% Calculate threshold t
t= (meanp+meanm)/2*w;

format long;
psize=size(score_trainp,1);
nsize=size(score_trainm,1);
Bp=score_trainp-ones(psize,1)*meanp;   %Fisher train using scores as features 
Bn=score_trainm-ones(nsize,1)*meanm;
Sw=Bp'*Bp+Bn'*Bn;
wfisher = Sw\(meanp-meanm)';
wfisher=wfisher/norm(wfisher);

tfisher=(meanp+meanm)./2*wfisher;

FisherPosErrorTrain = sum(score_trainp*wfisher <= tfisher); % Fisher train using scores as features 
FisherPosTrainError = FisherPosErrorTrain/size(score_trainp,1);

FisherNegErrorTrain = sum(score_trainm*wfisher >= tfisher);
FisherNegTrainError = FisherNegErrorTrain/size(score_trainm,1);

FisherTrainError = (FisherPosErrorTrain + FisherNegErrorTrain)/train_N

%% Sort part
nums = eigenvectors * wfisher;
[B,I] = sort(abs(nums),'descend');
nums(I(1:30));
[num_features, txt_features, raw_features] = xlsread('featurenames.xlsx','Sheet1');
% % raw(:,2:3);
Top30 = txt_features(I(1:30)+1,1);
nu30 = [1:30]';
T = table(nu30 ,nums(I(1:30)), Top30, I(1:30))
