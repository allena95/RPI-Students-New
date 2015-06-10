%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('DataWoutCat.xlsx');

num = num(~any(isnan(num),2),:); %remove students with missing data

data = num(:,2:end);
features = [data(:,1:32) data(:,34:36) data(:,39:end)];

%features = [features(:,1:32) features(:,35:end)];
%features = features(~any(isnan(features),2),:);

survey = features(:,1:31);

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

%% Define testing and trianing sets

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
YTrain = Y(1:train_size,:);
YTest = Y(train_size+1:end,:);

%Break them up into Class 1 and Class -1
Classp_train = Train(YTrain==1,:);
Classm_train = Train(YTrain==0,:);

Classp_test = Test(YTest==1,:);
Classm_test = Test(YTest==0,:);
%%
Train_total = [Classp_train; Classm_train];

[eigenvectors, scores, eigenvalues] = pca(Train_total);

%% 
explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100)
figure
bar(explainedVar)

%% 
[m,n] = size(Train_total)
[mp,np] = size(Classp_train)

Classp_train = scores(1:mp,:);
Classm_train = scores(mp+1:m,:);





%% Fisher method

meanp=mean(Classp_train);
meanm=mean(Classm_train);

psize=size(Classp_train,1)
nsize=size(Classm_train,1)
Bp=Classp_train-ones(psize,1)*meanp;
Bn=Classm_train-ones(nsize,1)*meanm;

Sw=Bp'*Bp+Bn'*Bn;
wfisher = Sw\(meanp-meanm)';
wfisher=wfisher/norm(wfisher)

tfisher=(meanp+meanm)./2*wfisher
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Analyze training data  results of the Fisher Linear Discriminant

FisherPosErrorTrain = sum(Classp_train*wfisher <= tfisher);
FisherNegErrorTrain = sum(Classm_train*wfisher >= tfisher);

FisherTrainError= ((FisherPosErrorTrain + FisherNegErrorTrain)/(size(Train,1)))  

% Histogram of Fisher Training Results
HistClass(Classp_train,Classm_train,wfisher,tfisher,...
    'Fisher Method Training Results',FisherTrainError); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
Test_total = [Classp_test; Classm_test];

[mp_test,np_test] = size(Classp_test);    % size for Classp
[mm_test,nm_test] = size(Classm_test);    % size for Classm
[m_test,n_test] = size(Test_total);      % size for total

%Test_total2 = Test_total - ones(m_test,1)*train_mean;
%Classp_test2 = Test_total2(1:mp_test,:);
%Classm_test2 = Test_total2((mp_test+1):end,:);

Classm_test_scores = Classm_test * eigenvectors;
Classp_test_scores = Classp_test * eigenvectors;

scores_test_total = [Classp_test_scores; Classm_test_scores];



Classp_test = scores_test_total(1:mp_test,:);
Classm_test = scores_test_total(mp_test+1:m_test,:);


FisherPosErrorTest = sum(Classp_test*wfisher <= tfisher);
FisherNegErrorTest = sum(Classm_test*wfisher >= tfisher);

FisherTestError= ((FisherPosErrorTest + FisherNegErrorTest)/(size(Test,1)))   

% Histogram of Fisher Testing Results
HistClass(Classp_test,Classm_test,wfisher,tfisher,...
    'Fisher Method Testing Results',FisherTestError);

%Train error is 13.6%, testing error ~5%, but negative error is quite bad