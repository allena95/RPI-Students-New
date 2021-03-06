
%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('DataWoutCat.xlsx');

num = num(~any(isnan(num),2),:); %remove students with missing data

data = num(:,2:end);
features = [data(:,34:36) data(:,39:end)];

features = [data(:,1:32) data(:,34:36) data(:,39:end)];

%features = features(:,33:37) %gpa only
%features = features(:,21);
%features = features(:,32:end);
%features = [features(:,1:32) features(:,35:end)];
%features = features(~any(isnan(features),2),:);

%survey = features(:,1:31);

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
YTrain = Y(1:train_size,:);
YTest = Y(train_size+1:end,:);
%%
s=std(Train);
a = diag(1./s);
[m,n] = size(Train);
one_m = ones(m,m);
train_mean = (1/m)*(ones(1,m)*Train);

Train = (Train - ones(m,1)*train_mean)*a;
%%
[m_test,n_test] = size(Test);
Test = (Test - ones(m_test,1)*train_mean)*a;
%%
%Break them up into Class 1 and Class -1
Classp_train = Train(YTrain==1,:);
Classm_train = Train(YTrain==0,:);

Classp_test = Test(YTest==1,:);
Classm_test = Test(YTest==0,:);

Classm_train = [Classm_train; Classm_train];







%% Fisher method

meanp=mean(Classp_train);
meanm=mean(Classm_train);

psize=size(Classp_train,1);
nsize=size(Classm_train,1);
Bp=Classp_train-ones(psize,1)*meanp;
Bn=Classm_train-ones(nsize,1)*meanm;

Sw=Bp'*Bp+Bn'*Bn;
wfisher = Sw\(meanp-meanm)';
wfisher=wfisher/norm(wfisher);

tfisher=(meanp+meanm)./2*wfisher;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Analyze training data  results of the Fisher Linear Discriminant

FisherPosErrorTrain = sum(Classp_train*wfisher <= tfisher);
FisherNegErrorTrain = sum(Classm_train*wfisher >= tfisher);

FisherTrainError= ((FisherPosErrorTrain + FisherNegErrorTrain)/(size(Train,1)))  

% Histogram of Fisher Training Results
HistClass(Classp_train,Classm_train,wfisher,tfisher,...
    'Fisher Method Training Results',FisherTrainError); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


FisherPosErrorTest = sum(Classp_test*wfisher <= tfisher);
FisherNegErrorTest = sum(Classm_test*wfisher >= tfisher);

FisherTestError= ((FisherPosErrorTest + FisherNegErrorTest)/(size(Test,1)))   

% Histogram of Fisher Testing Results
HistClass(Classp_test,Classm_test,wfisher,tfisher,...
    'Fisher Method Testing Results',FisherTestError);









%%
Train = [Classp_train;Classm_train];
Test = [Classp_test;Classm_test];

%%

Test = [Classp_test;Classm_test];

[ptrain_m,ptrain_n]=size(Classp_train);
[mtrain_m,mtrain_n]=size(Classm_train);
[ptest_m,ptest_n]=size(Classp_test);
[mtest_m,mtest_n]=size(Classm_test);

YTrain = [ones(ptrain_m,1);zeros(mtrain_m,1)];
YTest = [ones(ptest_m,1);zeros(mtest_m,1)];


%% Nearest Neighbor
% Finds the nearest element in Train for each element in Test.
% Classifier gives the index of the nearest Train for the corresponding 
%row in Test

classifier=knnsearch(Train,Test);
total_error=0;

%% KNN Error

stay_error=0;
for i=1:ptest_m,
    if YTest(i)~= YTrain(classifier(i))
        stay_error=stay_error+1;
    end
end
stay_error_percent = stay_error/size(Classp_test,1) % percent error on those who stayed


leave_error=0;
for i=ptest_m+1:size(Test,1);
    if YTest(i)~= YTrain(classifier(i))
        leave_error=leave_error+1;
    end
end
leave_error_percent = leave_error/size(Classm_test,1) % percent error on those who left

total_error = leave_error+stay_error
error_percent = total_error/size(Test,1) % Total error of classifier

%% 


%% Normal Vector Weight Thing

[num,txt,raw] = xlsread('featurenames.xlsx');
names = txt;
A = wfisher;
size(A);
%A = abs(A);
[I,B] = sort(abs(A),'descend');
n = 39;
for i = 1:n
    display(sprintf('Feature %d: %s   Score: %d',i, char(names(B(i))),A(B(i))))
end;






