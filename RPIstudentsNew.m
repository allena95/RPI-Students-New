%% DATUM RPI Students
clear;
close all;

%% Load in the data
[num,txt,raw] = xlsread('DataWoutCat.xlsx');
[m,n] = size(num);

returned15 = num(:,38)
%% Separate the original dataset into training and testing
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,m);
A=num(p,:);
returned15 = returned15(p,:);
%Use trainpct percent of the data for training and the rest for testing.
trainpct=.75;
train_size=ceil(m*trainpct);

Train = A(1:train_size,:);
Test = A(train_size+1:end,:);
YTrain = returned15(1:train_size,:);
YTest = returned15(train_size+1:end,:);

% Break them up into Class 1 and Class 0
% Classp_train = Train(YTrain==1,:);
% Classm_train = Train(YTrain==0,:);
% 
% Classp_test = Test(YTest==1,:);
% Classm_test = Test(YTest==0,:);

%% Correlation image on Train dataset
TrainNew = Train(~any(isnan(Train),2),:)
TrainCorr = corr(TrainNew)
imagesc(TrainCorr)
% heatmap = HeatMap(TrainCov)

%% Sort SOC code
[B,I] = sort(num(:,42),'descend');
sortedmatrix = num(I,:);
SOC = sortedmatrix(41:end,:);
gpaSOC = SOC(:,[41,42]);
gpaSOC = gpaSOC(~any(isnan(gpaSOC),2),:)


%% Percentage of students with SOC code who did well (GPA > 3.2) in college

blah1 = zeros(9,2);
for soc = 1:9
    gpa = gpaSOC(gpaSOC(:,2)==soc);
    blah1(soc,1)=soc;
    count = 0;
    for m = 1:size(gpa,1)
        if gpa(m,1) > 2.67
            count = count +1;
           
        end            
    end
    blah1(soc,2) = count/size(gpa,1);
    
%     X = ['The percentage of students with SOC code ',num2str(soc), ' did well (Cumulative GPA Fall 2014 > 3.2) is ', num2str(count/size(gpa,1))];
%     disp(X)
end


blah2 = zeros(9,2);
for soc = 1:9
    gpa = gpaSOC(gpaSOC(:,2)==soc);
    blah2(soc,1)=soc;
    count = 0;
    for m = 1:size(gpa,1)
        if gpa(m,1) > 3
            count = count +1;
           
        end            
    end
    blah2(soc,2) = count/size(gpa,1);
    
%     X = ['The percentage of students with SOC code ',num2str(soc), ' did well (Cumulative GPA Fall 2014 > 3.2) is ', num2str(count/size(gpa,1))];
%     disp(X)
end


blah3 = zeros(9,2);
for soc = 1:9
    gpa = gpaSOC(gpaSOC(:,2)==soc);
    blah3(soc,1)=soc;
    count = 0;
    for m = 1:size(gpa,1)
        if gpa(m,1) > 3.33
            count = count +1;
           
        end            
    end
    blah3(soc,2) = count/size(gpa,1);
%     
%     X = ['The percentage of students with SOC code ',num2str(soc), ' did well (Cumulative GPA Fall 2014 > 3.2) is ', num2str(count/size(gpa,1))];
%     disp(X)
end


blah4 = zeros(9,2);
for soc = 1:9
    gpa = gpaSOC(gpaSOC(:,2)==soc);
    blah4(soc,1)=soc;
    count = 0;
    for m = 1:size(gpa,1)
        if gpa(m,1) > 3.67
            count = count +1;
           
        end            
    end
    blah4(soc,2) = count/size(gpa,1);
    
%     X = ['The percentage of students with SOC code ',num2str(soc), ' did well (Cumulative GPA Fall 2014 > 3.2) is ', num2str(count/size(gpa,1))];
%     disp(X)
end

%% Subplots of four different Cumulative GPA in Fall 2014

figure
subplot(2,2,1)
line(blah1(:,1),blah1(:,2),'Color',[0.2 0.5 0.8],'linewidth',2.5)
xlabel('Strength of Curriculum')
ylabel('Percentage')
title('SOC Code vs. Percentage of students whose GPA > 2.67')

subplot(2,2,2)
line(blah2(:,1),blah2(:,2),'Color',[0.2 0.5 0.8],'linewidth',2.5)
xlabel('Strength of Curriculum')
ylabel('Percentage')
title('SOC Code vs. Percentage of students whose GPA > 3')

subplot(2,2,3)
line(blah3(:,1),blah3(:,2),'Color',[0.2 0.5 0.8],'linewidth',2.5)
xlabel('Strength of Curriculum')
ylabel('Percentage')
title('SOC Code vs. Percentage of students whose GPA > 3.33')

subplot(2,2,4)
line(blah4(:,1),blah4(:,2),'Color',[0.2 0.5 0.8],'linewidth',2.5)
xlabel('Strength of Curriculum')
ylabel('Percentage')
title('SOC Code vs. Percentage of students whose GPA > 3.67')

%% Playing around with probplot

% probplot('normal',gpaSOC(:,1)) % I have no idea what it means :(

%% Top and Bottom (Based on GPA)
[C,J] = sort(Train(:,41),'descend');
sortedmatrixGPA = Train(J,:);
GPAAna = sortedmatrixGPA(27:end,:);
Top = GPAAna(1:254,:);
Bottom = GPAAna(size(GPAAna)-253:end,:);



% GPA boxplot
figure
subplot(1,2,1)
boxplot(Top(:,[35:37,40,41]))
title('All GPA for top 25%')
% % imagesc(Top(:,[2:32,34:end]))
% 
subplot(1,2,2)
boxplot(Bottom(:,[35:37,40,41]))
title('All GPA for bottom 25%')
% imagesc(Bottom(:,[2:32,34:end]))


