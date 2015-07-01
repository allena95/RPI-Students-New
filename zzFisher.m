
%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('EditedNoCourses.xlsx');

%num = num(~any(isnan(num),2),:); %remove students with missing data

data = num(:,2:end);
%%
features = [data(:,1:33) data(:,35:37) data(:,40:end)];

want = [data(:,1:20) data(:,25:39) data(:,50) data(:,52:54) data(:,57:end) data(:,51) data(:,55:56)];
%want = want(:,33:end);
want = want(~any(isnan(want),2),:); %remove students with missing data

features = want(:,1:end-3);


s14 = want(:,end-2); %register labels spring 2014
f14 = want(:,end-1); %return labels f14
s15 = want(:,end); %return labels s15

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
%Train = [Train sort(YTrain,'descend')];
Test = [Classp_test;Classm_test];

nC = 4;

% Do k-means with 10 restarts. 
opts = statset('Display','final');
[cidx, ctrs, SUMD, D]= kmeans(Train, nC,'Replicates',10,'Options',opts);

% K=means objective
objective = sum(SUMD);

[eigenvectors, scores, eigenvalues] = pca(Train);
explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100);
figure
bar(explainedVar)

%% K-Means Test and Full

% 
% total = [Train; Test];
% 
% % Do k-means with 10 restarts. 
% opts = statset('Display','final');
% [cidxTest, ctrsTest, SUMD, D]= kmeans(Test, nC,'Replicates',10,'Options',opts);;
% 
% % K=means objective
% objective = sum(SUMD);
% 
% 
% % Do k-means with 10 restarts. 
% opts = statset('Display','final');
% [cidxfull, ctrsFull, SUMD, D]= kmeans(total, nC,'Replicates',10,'Options',opts);;
% 
% % K=means objective
% objective = sum(SUMD);


%%
[eigenvectors,zscores,eigenvalues] = pca(Train);

figure
gscatter(zscores(:,1),zscores(:,2),cidx);

hold on
legend

[m,n] = size(eigenvectors)
evectors = num2str((1:m)')




for j = 1:m
    
    plot(20*[0,eigenvectors(j,1)], 20*[0,eigenvectors(j,2)])
    %text(20*[0,eigenvectors(j,1)], 20*[0,eigenvectors(j,2)], evectors(j))
end



% turn off the ticks
%set(gca,'xtick',[])
%set(gca,'ytick',[])

axis square
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('Principal Component Scatter Plot');
hold off




%% Skree Plot
% 
% k_obj = ones(15,2);
% 
% for nC = 1:15   
% % Do k-means with 10 restarts. 
%     opts = statset('Display','final');
%     [cidx, ctrs, SUMD, D]= kmeans(data, nC,'Replicates',10,'Options',opts);
% 
% % K=means objective
%     objective = sum(SUMD);
%     k_obj(nC,:) = [nC;objective];
% 
% end
% 
% %%
% 
% figure
% hold on
% plot(k_obj(:,1),k_obj(:,2))
% hold off


%biplot(eigenvectors(:,1:2), 'scores',zscores(:,1:2))
%%

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
Y_c = YTrain(classifier);
%% KNN Error

stay_error=0;
for i=1:ptest_m,
    if YTest(i)~= Y_c(i);
        stay_error=stay_error+1;
    end
end
stay_error_percent = stay_error/size(Classp_test,1) % percent error on those who stayed


leave_error=0;
for i=ptest_m+1:size(Test,1);
    if YTest(i)~= Y_c(i);
        leave_error=leave_error+1;
    end
end
leave_error_percent = leave_error/size(Classm_test,1) % percent error on those who left

total_error = leave_error+stay_error
error_percent = total_error/size(Test,1) % Total error of classifier

%% 

figure
imagesc(ctrs)
title('Cluster Centers')
colorbar

% figure
% imagesc(ctrsTest)
% title('CtrsTest')
% colorbar
% 
% figure
% imagesc(ctrsFull)
% title('CtrsFull')
% colorbar



%% Normal Vector Weight Thing

[num,txt,raw] = xlsread('featurenames.xlsx');
names = txt;
A = wfisher;
[m,n] = size(A);
%A = abs(A);
[I,B] = sort(abs(A),'descend');
feat = [1:43]'
for i = 1:m
    display(sprintf('Feature %d: %s   Score: %d',i, num2str(feat(B(i))),A(B(i))))
end;
%%
trim = 15;
n_feat = features(:,B);

reduced_features = n_feat(:,1:trim);
new = [reduced_features s15];
xlswrite('asdf.xlsx', new) %Write file with top 15 features only


%% Evaluate KNN

EVAL = Evaluate(YTest,Y_c);

%% Evaluate Fisher
[m,n] = size(Classp_test);
[mm,mn] = size(Classm_test);
Test = [Classp_test;Classm_test];


for i = 1:m;
    YTest(i) = 1;
end
for i = m+1:mm;
    YTest(i) = 0;
end;

Y_f = nan(size(Test,1),1);
for i = 1:size(Test,1);
    if Test(i,:)*wfisher<=tfisher;
        Y_f(i) = 0;
    end
    if Test(i,:)*wfisher>=tfisher;
        Y_f(i)=1;
    end
end


EVAL_f = Evaluate(YTest,Y_f);
%% Confusion Matrix for Fisher Test
C_f = confusionmat(YTest,Y_f)


figure
imagesc(C_f)
title('Fisher Confusion Matrix')
colorbar
%% Confusion Matrix for KNN
C_knn = confusionmat(YTest,Y_c);

figure
imagesc(C_knn)
title('KNN Confusion Matrix')
colorbar





