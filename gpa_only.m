%% 
[num,txt,raw] = xlsread('GPA ONLY.xlsx'); %read in the student data with only gpa

[m,n] = size(num)
% 
% %change any blank or 0 labels to -1
% for i = 1:m;
%     if num(i,6) ~= 1;
%         num(i,6) = -1;
%     end
%     if num(i,7) ~= 1;
%         num(i,7) = -1;
%     end
% end



num = num(~any(isnan(num),2),:); %remove students with missing data

returns = num(:,6:8);

s14 = num(:,8);
f14 = num(:,6);
s15 = num(:,7);

data = num(:,1:5); %strip the return labels

F14 = num(:,6); %return labels for fall 2014
S15 = num(:,7); %return lables for spring 2015



%% Define testing and trianing sets

% Training and testing matrices for DatasetA

% Classp_train  := Class 1 training data
% Classm_train  := Class -1 training data
% Classp_test   := Class 1 testing data
% Classm_test   := Class -1 testing  data


% Set random number to an initial seed
[r,c]=size(data);
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,r);
data=data(p,:);
Y=S15(p);
%Use trainpct percent of the data for training and the rest for testing.
trainpct=.75;
train_size=ceil(r*trainpct);

% Grab training and test data
Train = data(1:train_size,:);
Test = data(train_size+1:end,:);
YTrain = Y(1:train_size,:);
YTest = Y(train_size+1:end,:);

%Break them up into Class 1 and Class -1
Classp_train = Train(YTrain==1,:);
Classm_train = Train(YTrain==0,:);

Classp_test = Test(YTest==1,:);
Classm_test = Test(YTest==0,:);

% %%
% Train_total = [Classp_train; Classm_train];
% 
% [mp,np] = size(Classp_train);    % size for Classp
% [mm,nm] = size(Classm_train);    % size for Classm
% [m,n] = size(Train_total);      % size for total
% 
% train_mean = (1/m)*(ones(1,m)*Train_total);
% 
% Train_total = Train_total - ones(m,1)*train_mean;
% 
% 
% [eigenvectors, scores, eigenvalues] = pca(Train_total)
% 
% Classp_train = scores(1:mp,:);
% Classm_train = scores(mp+1:m,:);
% 
% 
% %%
% Test_total = [Classp_test; Classm_test];
% 
% [mp_test,np_test] = size(Classp_test);    % size for Classp
% [mm_test,nm_test] = size(Classm_test);    % size for Classm
% [m_test,n_test] = size(Test_total);      % size for total
% 
% Test_total = Test_total - ones(m_test,1)*train_mean;
% Classp_test2 = Test_total(1:mp_test,:);
% Classm_test2 = Test_total((mp_test+1):end,:);
% 
% %%
% Classm_test_scores = Classm_test2 * eigenvectors;
% Classp_test_scores = Classp_test2 * eigenvectors;
% 
% scores_test = [Classp_test_scores; Classm_test_scores];
% 
% Classp_test = scores_test(1:mp_test,:);
% Classm_test = scores_test(mp_test+1:m_test,:);

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


FisherPosErrorTest = sum(Classp_test*wfisher <= tfisher);
FisherNegErrorTest = sum(Classm_test*wfisher >= tfisher);

FisherTestError= ((FisherPosErrorTest + FisherNegErrorTest)/(size(Test,1)))   

% Histogram of Fisher Testing Results
HistClass(Classp_test,Classm_test,wfisher,tfisher,...
    'Fisher Method Testing Results',FisherTestError);
%% 

[eigenvectors, scores, eigenvalues] = pca(data);

explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100)
figure
bar(explainedVar)