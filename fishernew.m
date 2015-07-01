%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;

%%
[num,txt,raw] = xlsread('DataWQ.xlsx');

%num = num(~any(isnan(num),2),:); %remove students with missing data
s14 = [num(:,2:34) num(:,36) num(:,43:44) num(:,35)];
f14 = [num(:,2:34) num(:,36) num(:,38) num(:,43:44) num(:,39)];
%%
s14 = s14(~any(isnan(s14),2),:); %remove students with missing data
f14 = f14(~any(isnan(f14),2),:); %remove students with missing data

%features = features(:,33:37) %gpa only
%features = features(:,21);
%features = features(:,32:end);
%features = [features(:,1:32) features(:,35:end)];
%features = features(~any(isnan(features),2),:);

%survey = features(:,1:31);


%% Spring 2014
% 
% [fisherror,train,test,w,t,perror_percent,merror_percent,error_total] = ...
%     classifier(s14(:,1:end-1),s14(:,end),.75);
% 
% %% Fall 2014 
% [fisherror2,train2,test2,w2,t2,perror_percent2,merror_percent2,error_total2] = ...
%     classifier(f14(:,1:end-1),f14(:,end),.75);

%% Determine if students left in general
leave = nan(size(num,1),1);
s14 = num(:,35);
f14 = num(:,39);
s15 = num(:,40);



for i = 1:size(num,1);
    %if s14(i) == 0;
    %    leave(i) = 0;
    %end
    if f14(i) == 0;
        leave(i)=0;
    end
    if s15(i) == 0;
        leave(i)=0;
    end
end

for i = 1:size(num,1);
    if leave(i) ~= 0;
        leave(i) = 1;
    end
end

%%
%want = [num(:,2:34) num(:,36) num(:,38) num(:,42) num(:,43:44) leave];

want = [num(:,2:34) num(:,36) num(:,38) num(:,42) num(:,43:44) leave];

%want = [num(:,2:34) num(:,43:44) leave];

want = want(~any(isnan(want),2),:); %remove students with missing data


[fisherror,train,test,w,t,perror_percent,merror_percent,error_total] = ...
    classifier(want(:,1:end-1),want(:,end),.75);



%%

nC = 4;

% Do k-means with 10 restarts. 
opts = statset('Display','final');
[cidx, ctrs, SUMD, D]= kmeans(train(:,1:end-1), nC,'Replicates',10,'Options',opts);

% K=means objective
objective = sum(SUMD);

[eigenvectors, scores, eigenvalues] = pca(train(:,1:end-1));
explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100);
figure
bar(explainedVar)

figure
imagesc(ctrs)
title('Cluster Centers Train')
colorbar


nC = 4;

% Do k-means with 10 restarts. 
opts = statset('Display','final');
[cidx2, ctrs_test, SUMD, D]= kmeans(test(:,1:end-1), nC,'Replicates',10,'Options',opts);

% K=means objective
objective = sum(SUMD);

figure
imagesc(ctrs_test)
title('Cluster Centers Test')
colorbar


%%
features = want(:,1:end-1);

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

figure
imagesc(ctrs)
title('Cluster Centers All')
colorbar
%%
[num,txt,raw] = xlsread('featurenames3.xlsx');
names = txt;
A = w;
[m,n] = size(A);
%A = abs(A);
[I,B] = sort(abs(A),'descend');

for i = 1:m
    display(sprintf('Feature %d: %s   Score: %d',i, char(names(B(i))),A(B(i))))
end;
%%

figure
imagesc(corr(want))
title('Correlation Matrix All')
colorbar
