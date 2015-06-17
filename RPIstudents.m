%% DATUM RPI Students - SAT, First year GPA, Returned Fall 2015 and Two Survey Figures
clear;
close all;

%% Load in the data
[num,txt,raw] = xlsread('Cross out blanks in registered 14.xlsx');
[m,n] = size(num);

%% Mean center SAT score and Find the covariance matrix
count = 0;
for i = 1:m
    if (~isnan(num(i,50)))
       count = count + 1;
    end
end

A = zeros(count, n);

for i = 1:m
    if (~isnan(num(i,50)))
        A(i,:) = num(i,:);
    end
end

a = A(:,1);

A;

[B,I] = sort(abs(a));

keep = I(13:end);
A = A(keep,:);

SAT = A(:,45);
SAT_mean_centered = SAT - mean(SAT);

first_year_GPA = A(:,50);


C = cov(SAT_mean_centered, first_year_GPA);

% imagesc(C)   
% colormap(gray);
% colorbar;
%% Clustering, run K-means for nC=3 clusters (not working yet)
% 
% nC = 3;
% SATreg = num(:,[45,47]);
% 
% % Do k-means with 10 restarts. 
% opts = statset('Display','final');
% [cidx, ctrs, SUMD, D]= kmeans(SATreg, nC,'Replicates',10,'Options',opts);
% 
% % K=means objective
% objective = sum(SUMD);
% 
% figure
% hold on
% set(gca,'YTick',[-1;0;1])
% set(gca,'XTick',[0;900;1600])
% plot(SATreg(:,1),SATreg(:,2),'*');

%% Plot SAT scores, first year GPA and if they returned


figure
hold on 
for i = 1:m
    if num(i,51) == 0
        plot(num(i,45),num(i,50),'*','MarkerSize',5,'Color','r')
    else if num(i,51) == 1
            plot(num(i,45),num(i,50),'*','MarkerSize',5,'Color','b')
            
        end
    end
end

xlabel('SAT Score');
ylabel('First Year GPA');
title('SAT, First year GPA, Returned Fall 2015');

hold off

%% Survey part
SurveyGrades = A(:,[2,3,4]); % take out the parts i think are important
SurveyMath = A(:,[13,14,15,17]);

SurveyGrades = sort(SurveyGrades,'descend');
figure
imagesc(SurveyGrades)
colorbar;
title('Survey Questions 1 2 3');
xlabel('Answers');
ylabel('Students');

% boxplot(SurveyGrades)

figure
SurveyMath = sort(SurveyMath,'descend');
imagesc(SurveyMath)
colorbar
title('Survey Questions 12 13 14 16');
xlabel('Answers');
ylabel('Students');

%% Covariance on survey

AllSurvey = A(:,[2:20,26:36,52]);     % All Survey Questions and if they returned spring 2015
AllSurvey = AllSurvey(~any(isnan(AllSurvey),2),:);


survey_N = size(AllSurvey,1) ;
img_mean = mean(AllSurvey);
Survey_mean_centered = AllSurvey - ones(survey_N,1)*img_mean;

figure
imagesc(corr(Survey_mean_centered))


%% 

        