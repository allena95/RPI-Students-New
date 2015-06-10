%%
clear;
close all;


%%

[num,txt,raw] = xlsread('GPA ONLY.xlsx');

data = num(:,1:5);

returns = num(:,6:8);

s14 = num(:,8);
f14 = num(:,6);
s15 = num(:,7);

%%

nC = 3;

s15_pred = [data(:,3) data(:,5)]

% Do k-means with 10 restarts. 
opts = statset('Display','final');
[cidx, ctrs, SUMD, D]= kmeans(s15_pred, nC,'Replicates',10,'Options',opts);

% K=means objective
objective = sum(SUMD)

%%
s15_predtxt = ['may14'; 'f14  '];

%%

[eigenvectors,zscores,eigenvalues] = pca(data)

figure
gscatter(zscores(:,1),zscores(:,2),cidx);


hold on
legend

for j = 1:5
    
    plot(3.5*[0,eigenvectors(j,1)], 3.5*[0,eigenvectors(j,2)])
    %text(3.5*[0,eigenvectors(j,1)], 3.5*[0,eigenvectors(j,2)], s15_predtxt(j))
    
end



% turn off the ticks
%set(gca,'xtick',[])
%set(gca,'ytick',[])

axis square
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('Principal Component Scatter Plot');
hold off


%%
explainedVar = cumsum(eigenvalues./sum(eigenvalues) * 100)
figure
bar(explainedVar)


 %%
% 
% 
% k_obj = ones(15,2);
% 
% for nC = 1:15   
% % Do k-means with 10 restarts. 
%     opts = statset('Display','final');
%     [cidx, ctrs, SUMD, D]= kmeans(s15_pred, nC,'Replicates',10,'Options',opts);
% 
% % K=means objective
%     objective = sum(SUMD)
%     k_obj(nC,:) = [nC;objective]
% 
% end
% %}
% %%
% 
% 
% figure
% hold on
% plot(k_obj(:,1),k_obj(:,2))
% hold off