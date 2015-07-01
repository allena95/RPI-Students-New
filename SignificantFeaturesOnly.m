%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;
%% Classification of S15 using significant features from T-Test


[num,txt,raw] = xlsread('DataWQ2.xlsx');

data = num(:,2:end);
want = [data(:,6) data(:,14) data(:,20) data(:,21) data(:,26) data(:,33) data(:,35:37) data(:,40:43) data(:,end) data(:,39)];

want = want(~any(isnan(want),2),:); %remove students with missing data

features = want(:,1:end-1);
labels = want(:,end);


[fisherror4,Train4,Test4,w4,t4,perror4,merror4,error_total4] = ...
classifier(features,labels,.75);

%Train error 17.95%, Test error 15.75%
%Leave error for both train and test < 40%
%%

s=std(features);
a = diag(1./s);
[m,n] = size(features);
one_m = ones(m,m);

features = (features - (1/m)*(ones(m,m)*features))*a;




nC = 4;

% Do k-means with 10 restarts. 
opts = statset('Display','final');
[cidx, ctrs, SUMD, D]= kmeans(features, nC,'Replicates',10,'Options',opts);

% K=means objective
objective = sum(SUMD);

[eigenvectors, scores, eigenvalues] = pca(features);
explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100);
figure
bar(explainedVar)

%%
figure
imagesc(ctrs)
title('Cluster Centers')
colorbar

%% Confusion Matrix for sig S15 prediction
Y_f4 = nan(size(Test4,1),1);
for i = 1:size(Test4,1);
    if Test4(i,1:end-1)*w4<=t4;
        Y_f4(i) = 0;
    end
    if Test4(i,1:end-1)*w4>=t4;
        Y_f4(i)=1;
    end
end


C_f4 = confusionmat(Test4(:,end),Y_f4)


%% List Features for S15 prediction in order of weight

[num,txt,raw] = xlsread('Significant Features S15.xlsx');
names = txt;
A = w4;
[m,n] = size(A);
%A = abs(A);
[I,B] = sort(abs(A),'descend');

for i = 1:m
    display(sprintf('Feature %d: %s   Score: %d',i, char(names(B(i))),A(B(i))))
end;



%% Classification of Ever Left Using Sig Features from T-Test

[num,txt,raw] = xlsread('DataWQ2.xlsx');

data = num(:,2:end);
want = [data(:,6) data(:,8) data(:,14) data(:,20) data(:,21) data(:,26) data(:,33) data(:,35) data(:,37) data(:,41:43) data(:,end) data(:,44)];

want = want(~any(isnan(want),2),:); %remove students with missing data

features5 = want(:,1:end-1);
labels5 = want(:,end);


[fisherror5,Train5,Test5,w5,t5,perror5,merror5,error_total5] = ...
classifier(features5,labels5,.75);
%% Confusion matrix for sig ever leave prediction 

Y_f5 = nan(size(Test5,1),1);
for i = 1:size(Test5,1);
    if Test5(i,1:end-1)*w5<=t5;
        Y_f5(i) = 0;
    end
    if Test5(i,1:end-1)*w5>=t5;
        Y_f5(i)=1;
    end
end


C_f5 = confusionmat(Test5(:,end),Y_f5)






%% List Features in order of weight
[num,txt,raw] = xlsread('Significant FeaturesEver.xlsx');
names = txt;
A = w5;
[m,n] = size(A);
%A = abs(A);
[I,B] = sort(abs(A),'descend');

for i = 1:m
    display(sprintf('Feature %d: %s   Score: %d',i, char(names(B(i))),A(B(i))))
end;
