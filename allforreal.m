%% DATUM RPI Students - Predict leaving Spring 2015
clear;
close all;


%%

[num,txt,raw] = xlsread('EditedNoCourses.xlsx');

want = num(:,2:end);
math = want(:,21:24);

q21 = math(:,1);
q22 = math(:,2);
q23 = math(:,3);
q24 = math(:,4);

%%

correct = zeros(size(math,1),1);

for i = 1:size(correct,1);
    if q21(i)== 1;
        correct(i) = correct(i) + 1;
    end
    if q22(i) == 2;
        correct(i) = correct(i) + 1;
    end
    if q23(i) == 5;
        correct(i) = correct(i) + 1;
    end
    if q24(i) == 3;
        correct(i) =  correct(i) + 1;
    end
end
%%
data = [want(:,1:20) correct want(:,25:50) want(:,54) want(:,58:60) want(:,51) want(:,55:56)];
data = data(~any(isnan(data),2),:); %remove students with missing data

features = data(:,1:51);

s14 = data(:,52);
f14 = data(:,53);
s15 = data(:,54);
%%
[fisherror,Train,Test,w,t,perror,merror,error_total] = ...
classifier(features,s15,.75);



%%
xlswrite('mathcorrect.xlsx', correct);
%%

nC = 3;

% Do k-means with 10 restarts. 
opts = statset('Display','final');
[cidx, ctrs, SUMD, D]= kmeans(Train(:,1:end-1), nC,'Replicates',10,'Options',opts);;

% K=means objective
objective = sum(SUMD);

[eigenvectors, scores, eigenvalues] = pca(Train(:,1:end-1));
explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100);
figure
bar(explainedVar)

%%

figure
gscatter(scores(:,1),scores(:,2),cidx);

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
imagesc(eigenvectors(1:10,:))
title('Ctrs')
colorbar
%%
