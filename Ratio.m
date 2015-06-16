%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('DataWoutCat.xlsx');

num = num(~any(isnan(num),2),:); %remove students with missing data

data = num(:,2:end);
features = [data(:,34:36) data(:,39:end)];

features = [data(:,1:32) data(:,34:36) data(:,39:end)];

s14 = data(:,33); %register labels spring 2014
f14 = data(:,37); %return labels f14
s15 = data(:,38); %return labels s15

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
%Break them up into Class 1 and Class 0
Classp_train = Train(YTrain==1,:);
Classm_train = Train(YTrain==0,:);

Classp_test = Test(YTest==1,:);
Classm_test = Test(YTest==0,:);

%%
classm = [Classm_test; Classm_train];
classp = [Classp_test; Classp_train];
[m,n] = size(classm);
p = randperm(s,size(classp,1));

classp = classp(p,:);
classp = classp(1:m,:);

classp = [classp ones(m,1)];
classm = [classm zeros(m,1)];

reduced = [classp; classm];
new_features = reduced(:,1:39);
reduced_labels = reduced(:,40);

%%

% Set random number to an initial seed
[r,c]=size(new_features);
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,r);
new_features=new_features(p,:);
Y=reduced_labels(p);
%Use trainpct percent of the data for training and the rest for testing.
trainpct=.75;
train_size=ceil(r*trainpct);

% Grab training and test data
Train = new_features(1:train_size,:);
Test = new_features(train_size+1:end,:);
YTrain = Y(1:train_size,:);
YTest = Y(train_size+1:end,:);

st=std(Train);
a = diag(1./st);
[m,n] = size(Train);
one_m = ones(m,m);
train_mean = (1/m)*(ones(1,m)*Train);

Train = (Train - ones(m,1)*train_mean)*a;

[m_test,n_test] = size(Test);
Test = (Test - ones(m_test,1)*train_mean)*a;

%Break them up into Class 1 and Class 0
Classp_train = Train(YTrain==1,:);
Classm_train = Train(YTrain==0,:);

Classp_test = Test(YTest==1,:);
Classm_test = Test(YTest==0,:);

Train = [Classp_train; Classp_test];
Test = [Classp_test;Classm_test];

[ptrain_m,ptrain_n]=size(Classp_train);
[mtrain_m,mtrain_n]=size(Classm_train);
[ptest_m,ptest_n]=size(Classp_test);
[mtest_m,mtest_n]=size(Classm_test);

YTrain = [ones(ptrain_m,1);-ones(mtrain_m,1)];
YTest = [ones(ptest_m,1);-ones(mtest_m,1)];
%%

classifier=knnsearch(Train,Test);
total_error=0;


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


































