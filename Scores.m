
%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;
trim = 39
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

[m,n] = size(Train);    % size for total

train_mean = (1/m)*(ones(1,m)*Train);
s=std(Train);
a = diag(1./s);
[m,n] = size(Train);
one_m = ones(m,m);
Train = (Train - (1/m)*(ones(m,m)*Train))*a; 

%Break them up into Class 1 and Class -1
Classp_train = Train(YTrain==1,:);
Classm_train = Train(YTrain==0,:);

Classp_test = Test(YTest==1,:);
Classm_test = Test(YTest==0,:);

[mp,np] = size(Classp_train);    % size for Classp
[mm,nm] = size(Classm_train);    % size for Classm
[m,n] = size(Train);      % size for total


%%
[eigenvectors,scores,eigenvalues] = pca(Train);

explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100)
figure
bar(explainedVar)
%%
trimmed_scores = scores(:,1:trim);
classp_scores = trimmed_scores(1:mp,:);
classm_scores = trimmed_scores(mp+1:m,:);


%% Fisher method

meanp=mean(classp_scores);
meanm=mean(classm_scores);

psize=size(classp_scores,1)
nsize=size(classm_scores,1)
Bp=classp_scores-ones(psize,1)*meanp;
Bn=classm_scores-ones(nsize,1)*meanm;

Sw=Bp'*Bp+Bn'*Bn;
wfisher = Sw\(meanp-meanm)';
wfisher=wfisher/norm(wfisher)

tfisher=(meanp+meanm)./2*wfisher
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Analyze training data  results of the Fisher Linear Discriminant

FisherPosErrorTrain = sum(classp_scores*wfisher <= tfisher);
FisherNegErrorTrain = sum(classm_scores*wfisher >= tfisher);

FisherTrainError= ((FisherPosErrorTrain + FisherNegErrorTrain)/(size(Train,1)))  

% Histogram of Fisher Training Results
HistClass(classp_scores,classm_scores,wfisher,tfisher,...
    'Fisher Method Training Results',FisherTrainError); 


%% Compute test scores

Test_total = [Classp_test; Classm_test];

[mp_test,np_test] = size(Classp_test);    % size for Classp
[mm_test,nm_test] = size(Classm_test);    % size for Classm
[m_test,n_test] = size(Test_total);      % size for total

Test_total2 = (Test_total - ones(m_test,1)*train_mean)*a;
Classp_test2 = Test_total2(1:mp_test,:);
Classm_test2 = Test_total2((mp_test+1):end,:);


Classm_test_scores = Classm_test2 * eigenvectors;
Classp_test_scores = Classp_test2 * eigenvectors;

scores_test_total = [Classp_test_scores; Classm_test_scores];


trimmed_scores_test = scores_test_total(:,1:trim);
classp_test_scores = trimmed_scores_test(1:mp_test,:);
classm_test_scores = trimmed_scores_test(mp_test+1:m_test,:);

%% Fisher on Test

FisherPosErrorTest = sum(classp_test_scores*wfisher <= tfisher);
FisherNegErrorTest = sum(classm_test_scores*wfisher >= tfisher);

FisherTestError= ((FisherPosErrorTest + FisherNegErrorTest)/(size(trimmed_scores_test,1)))   

% Histogram of Fisher Testing Results
HistClass(classp_test_scores,classm_test_scores,wfisher,tfisher,...
    'Fisher Method Testing Results',FisherTestError); 

%%


%% Nearest Neighbor
% Finds the nearest element in Train for each element in Test.
% Classifier gives the index of the nearest Train for the corresponding 
%row in Test

classifier=knnsearch(trimmed_scores,trimmed_scores_test);
total_error=0;

%% KNN Error
[ptrain_m,ptrain_n]=size(classp_scores);
[mtrain_m,mtrain_n]=size(classm_scores);
[ptest_m,ptest_n]=size(classp_test_scores);
[mtest_m,mtest_n]=size(classm_test_scores);

stay_error=0;
for i=1:ptest_m,
    if(YTest(i)~=YTrain(classifier(i)))
        stay_error=stay_error+1;
    end
end
stay_error_percent = stay_error/size(classp_test_scores,1) % percent error on those who stayed


leave_error=0;
for i=ptest_m+1:size(Test,1);
    if(YTest(i)~=YTrain(classifier(i)))
        leave_error=leave_error+1;
    end
end
leave_error_percent = leave_error/size(classm_test_scores,1) % percent error on those who left

total_error = leave_error+stay_error
error_percent = total_error/size(trimmed_scores_test,1) % Total error of classifier


