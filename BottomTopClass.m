%% DATUM RPI Students
clear;
close all;

%%

[top,txt1,raw1] = xlsread('Top.xlsx');
[bottom,txt2,raw2] = xlsread('Bottom.xlsx');

bottop = [top;bottom];

data = [bottop(:,1:20) bottop(:,25:35) bottop(:,42) bottop(:,44:46) bottop(:,49:50) bottop(:,51:53)];

data = data(~any(isnan(data),2),:); %remove students with missing data

features = (data(:,1:end-1));
%features = [features(:,1:32) features(:,38:39)];
labels = data(:,end);


%%
% Set random number to an initial seed
[r,c]=size(features);
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,r);
features=features(p,:);
Y=labels(p);
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
Test = [Classp_test;Classm_test];
Train = [Classp_train;Classm_train];
[ptrain_m,ptrain_n]=size(Classp_train);
[mtrain_m,mtrain_n]=size(Classm_train);
[ptest_m,ptest_n]=size(Classp_test);
[mtest_m,mtest_n]=size(Classm_test);

YTrain = [ones(ptrain_m,1);zeros(mtrain_m,1)];
YTest = [ones(ptest_m,1);zeros(mtest_m,1)];

classifier=knnsearch(Train,Test);
total_error=0;

%% KNN Error
% [ptrain_m,ptrain_n]=size(Classp_train);
% [mtrain_m,mtrain_n]=size(Classm_train);
% [ptest_m,ptest_n]=size(Classp_test);
% [mtest_m,mtest_n]=size(Classm_test);

top_error=0;
for i=1:ptest_m,
    if YTest(i)~= YTrain(classifier(i))
        top_error=top_error+1;
    end
end
top_error_percent = top_error/size(Classp_test,1) % percent error on those who stayed


bottom_error=0;
for i=ptest_m+1:size(Test,1);
    if YTest(i)~= YTrain(classifier(i))
        bottom_error=bottom_error+1;
    end
end
bottom_error_percent = bottom_error/size(Classm_test,1) % percent error on those who left

total_error = bottom_error+top_error
error_percent = total_error/size(Test,1) % Total error of classifier

%%
Train = [Classp_train;Classm_train];

nC = 4;

% Do k-means with 10 restarts. 
opts = statset('Display','final');
[cidx, ctrs, SUMD, D]= kmeans(Train, nC,'Replicates',10,'Options',opts);;

% K=means objective
objective = sum(SUMD);

[eigenvectors, scores, eigenvalues] = pca(Train);
explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100);
figure
bar(explainedVar)

%%
[eigenvectors,zscores,eigenvalues] = pca(Train);

figure
gscatter(zscores(:,1),zscores(:,2),cidx);

hold on
legend

[m,n] = size(eigenvectors)



for j = 1:m
    
    plot(20*[0,eigenvectors(j,1)], 20*[0,eigenvectors(j,2)])
    
end


axis square
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('Principal Component Scatter Plot');
hold off
%%

figure
imagesc(ctrs)
title('Ctrs')
colorbar
%%

NTrees = 35;
vars = 10
options.MaxIter = inf

mdl =  svmtrain(Train,YTrain,'options',options);
%%
Y_c = svmclassify(mdl,Test);

%[Y_c,score] = predict(mdl,Test);

Y_t = YTest;

%Y_c = str2double(Y_c);

%EVAL = [accuracy sensitivity specificity precision recall f_measure gmean];
EVAL = Evaluate(Y_t,Y_c);

posclass = 1;

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


%% Get the best features for top and bottom
% [num,txt,raw] = xlsread('featurenames2.xlsx');
% names = txt;
% A = eigenvectors*wfisher;
% size(A);
% A = abs(A);
% [I,B] = sort(A,'descend');
% n = size(wfisher,1);
% for i = 1:n
%     display(sprintf('Feature %d: %s   Score: %d',i, char(names(B(i))),A(B(i))))
% end;




%%












