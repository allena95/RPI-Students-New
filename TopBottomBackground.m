%% DATUM RPI Students
clear;
close all;

%% Load in the data
[num,txt,raw] = xlsread('NoCoursesWithGPA.xlsx');
[m,n] = size(num);

names = txt(1,[37:42,52,53]);

% DataNeeded = num(:,[37:42,52,53]);
% DataNeeded = DataNeeded(~any(isnan(DataNeeded),2),:);

%% Analysis

[B,I] = sort(num(:,51),'descend');
sortedmatrixGPA = num(I,:);
GPAAna = sortedmatrixGPA(27:end,:);
Top = GPAAna(1:254,:);
Bottom = GPAAna(size(GPAAna)-253:end,:);
TopBackgr = Top(:,[37:42,52,53,45:47,50,51]);
BottomBackgr = Bottom(:,[37:42,52,53,45:47,50,51]);

%% Top

% Gender
GirlCount = sum(TopBackgr(:,1)==1);
GirlP = GirlCount/size(TopBackgr,1);
BoyCount = sum(TopBackgr(:,1)==2);
BoyP = BoyCount/size(TopBackgr,1);
GenderTop = [GirlP,BoyP];

% Citizenship
NonCount = sum(TopBackgr(:,2)==1);
NonP = NonCount/size(TopBackgr,1);
PermCount = sum(TopBackgr(:,2)==2);
PermP = PermCount/size(TopBackgr,1);
CitizenCount = sum(TopBackgr(:,2)==3);
CitizenP = CitizenCount/size(TopBackgr,1);
CitizenTop = [NonP,PermP,CitizenP];


% Region
NECount = sum(TopBackgr(:,3)==1);
NEP = NECount/size(TopBackgr,1);
NonNECount = sum(TopBackgr(:,3)==2);
NonNEP = NonNECount/size(TopBackgr,1);
RegionTop = [NEP,NonNEP];

% Medalist
MCount = sum(TopBackgr(:,4)==1);
MP = MCount/size(TopBackgr,1);
NonMCount = sum(TopBackgr(:,4)==2);
NonMP = NonMCount/size(TopBackgr,1);
MedalistTop = [MP,NonMP];


% School
ArchiCount = sum(TopBackgr(:,5)==1);
ArchiP = ArchiCount/size(TopBackgr,1);
EngCount = sum(TopBackgr(:,5)==2);
EngP = EngCount/size(TopBackgr,1);
HASSCount = sum(TopBackgr(:,5)==3);
HASSP = HASSCount/size(TopBackgr,1);
ITWSCount = sum(TopBackgr(:,5)==4);
ITWSP = ITWSCount/size(TopBackgr,1);
MngCount = sum(TopBackgr(:,5)==5);
MngP = MngCount/size(TopBackgr,1);
SciCount = sum(TopBackgr(:,5)==6);
SciP = SciCount/size(TopBackgr,1);
UdclrCount = sum(TopBackgr(:,5)==7);
UdclrP = UdclrCount/size(TopBackgr,1);
SchoolTop = [ArchiP,EngP,HASSP,ITWSP,MngP,SciP,UdclrP];


% HS Type
PubCount = sum(TopBackgr(:,6)==1);
PubP = PubCount/size(TopBackgr,1);
PriCount = sum(TopBackgr(:,6)==2);
PriP = PriCount/size(TopBackgr,1);
ParoCount = sum(TopBackgr(:,6)==3);
ParoP = ParoCount/size(TopBackgr,1);
HSTop = [PubP,PriP,ParoP];


% SOC Code
Count1 = sum(TopBackgr(:,7)==1);
P1 = Count1/size(TopBackgr,1);
Count2 = sum(TopBackgr(:,7)==2);
P2 = Count2/size(TopBackgr,1);
Count3 = sum(TopBackgr(:,7)==3);
P3 = Count3/size(TopBackgr,1);
Count4 = sum(TopBackgr(:,7)==4);
P4 = Count4/size(TopBackgr,1);
Count5 = sum(TopBackgr(:,7)==5);
P5 = Count5/size(TopBackgr,1);
Count6 = sum(TopBackgr(:,7)==6);
P6 = Count6/size(TopBackgr,1);
Count7 = sum(TopBackgr(:,7)==7);
P7 = Count7/size(TopBackgr,1);
Count8 = sum(TopBackgr(:,7)==8);
P8 = Count8/size(TopBackgr,1);
Count9 = sum(TopBackgr(:,7)==9);
P9 = Count9/size(TopBackgr,1);
SOCTop = [P1,P2,P3,P4,P5,P6,P7,P8,P9];


% For GPA just use boxplot

%% Bottom

% Gender
GirlBCount = sum(BottomBackgr(:,1)==1);
GirlBP = GirlBCount/size(BottomBackgr,1);
BoyBCount = sum(BottomBackgr(:,1)==2);
BoyBP = BoyBCount/size(BottomBackgr,1);
GenderBottom = [GirlBP,BoyBP];

% Citizenship
NonBCount = sum(BottomBackgr(:,2)==1);
NonBP = NonBCount/size(BottomBackgr,1);
PermBCount = sum(BottomBackgr(:,2)==2);
PermBP = PermCount/size(BottomBackgr,1);
CitizenBCount = sum(BottomBackgr(:,2)==3);
CitizenBP = CitizenBCount/size(BottomBackgr,1);
CitizenBottom = [NonBP,PermBP,CitizenBP];

% Region
NEBCount = sum(BottomBackgr(:,3)==1);
NEBP = NEBCount/size(BottomBackgr,1);
NonNEBCount = sum(BottomBackgr(:,3)==2);
NonNEBP = NonNEBCount/size(BottomBackgr,1);
RegionBottom = [NEBP,NonNEBP];

% Medalist
MBCount = sum(BottomBackgr(:,4)==1);
MBP = MBCount/size(BottomBackgr,1);
NonMBCount = sum(BottomBackgr(:,4)==2);
NonMBP = NonMBCount/size(BottomBackgr,1);
MedalistBottom = [MBP,NonMBP];

% School
ArchiBCount = sum(BottomBackgr(:,5)==1);
ArchiBP = ArchiBCount/size(BottomBackgr,1);
EngBCount = sum(BottomBackgr(:,5)==2);
EngBP = EngBCount/size(BottomBackgr,1);
HASSBCount = sum(BottomBackgr(:,5)==3);
HASSBP = HASSBCount/size(BottomBackgr,1);
ITWSBCount = sum(BottomBackgr(:,5)==4);
ITWSBP = ITWSBCount/size(BottomBackgr,1);
MngBCount = sum(BottomBackgr(:,5)==5);
MngBP = MngBCount/size(BottomBackgr,1);
SciBCount = sum(BottomBackgr(:,5)==6);
SciBP = SciBCount/size(BottomBackgr,1);
UdclrBCount = sum(BottomBackgr(:,5)==7);
UdclrBP = UdclrBCount/size(BottomBackgr,1);
SchoolBottom = [ArchiBP,EngBP,HASSBP,ITWSBP,MngBP,SciBP,UdclrBP];


% HS Type
PubBCount = sum(BottomBackgr(:,6)==1);
PubBP = PubBCount/size(BottomBackgr,1);
PriBCount = sum(BottomBackgr(:,6)==2);
PriBP = PriBCount/size(BottomBackgr,1);
ParoBCount = sum(BottomBackgr(:,6)==3);
ParoBP = ParoBCount/size(BottomBackgr,1);
HSBottom = [PubBP,PriBP,ParoBP];


% SOC Code
Count1B = sum(BottomBackgr(:,7)==1);
P1B = Count1B/size(BottomBackgr,1);
Count2B = sum(BottomBackgr(:,7)==2);
P2B = Count2B/size(BottomBackgr,1);
Count3B = sum(BottomBackgr(:,7)==3);
P3B = Count3B/size(BottomBackgr,1);
Count4B = sum(BottomBackgr(:,7)==4);
P4B = Count4B/size(BottomBackgr,1);
Count5B = sum(BottomBackgr(:,7)==5);
P5B = Count5B/size(BottomBackgr,1);
Count6B = sum(BottomBackgr(:,7)==6);
P6B = Count6B/size(BottomBackgr,1);
Count7B = sum(BottomBackgr(:,7)==7);
P7B = Count7B/size(BottomBackgr,1);
Count8B = sum(BottomBackgr(:,7)==8);
P8B = Count8B/size(BottomBackgr,1);
Count9B = sum(BottomBackgr(:,7)==9);
P9B = Count9B/size(BottomBackgr,1);
SOCBottom = [P1B,P2B,P3B,P4B,P5B,P6B,P7B,P8B,P9B];


% For GPA just use boxplot
%% Figures
Gender = [GenderTop;GenderBottom];
Citizenship = [CitizenTop;CitizenBottom];
Region = [RegionTop;RegionBottom];
Medalist = [MedalistTop;MedalistBottom];
School = [SchoolTop;SchoolBottom];
HS = [HSTop;HSBottom];
SOC = [SOCTop;SOCBottom];

figure
subplot(3,3,1)
Genderb = bar(Gender);
title('Gender')
% legend('Girls','Boys','Location','best')
ylabel('Percentage')
set(gca,'XTickLabel',{'Top 25%','Bottom 25%'})

subplot(3,3,2)
Citizenb = bar(Citizenship);
title('Citizenship')
% legend('Non-Citizen','Permanent Resident','Citizen','Location','best')
ylabel('Percentage')
set(gca,'XTickLabel',{'Top 25%','Bottom 25%'})

subplot(3,3,3)
Regionb = bar(Region);
title('Region')
% legend('Northeast','Non-Northeast','Location','best')
ylabel('Percentage')
set(gca,'XTickLabel',{'Top 25%','Bottom 25%'})

subplot(3,3,4)
Medalistb = bar(Medalist);
title('Medalist')
% legend('Medalist','Non-Medalist','Location','best')
ylabel('Percentage')
set(gca,'XTickLabel',{'Top 25%','Bottom 25%'})

subplot(3,3,5)
Schoolb = bar(School);
title('School')
% legend('Architecture','Engineering','HASS','ITWS','Management','Science','Undeclared','Location','best')
ylabel('Percentage')
set(gca,'XTickLabel',{'Top 25%','Bottom 25%'})

subplot(3,3,6)
HSb = bar(HS);
title('High School Type')
% legend('Public','Private','Parochial','Location','best')
ylabel('Percentage')
set(gca,'XTickLabel',{'Top 25%','Bottom 25%'})

subplot(3,3,7)
SOCb = bar(SOC);
title('Strength of Curriculum')
ylabel('Percentage')
set(gca,'XTickLabel',{'Top 25%','Bottom 25%'})

subplot(3,3,8)
% GPA boxplot
% figure
% boxplot([TopBackgr(:,8:end),BottomBackgr(:,8:end)])
% title('GPA of top 25% vs. GPA of bottom 25%')
% ylabel('GPA')
% xlabel('First 5 are from top 25%, the rest are from bottom 25%')
% % imagesc(Top(:,[2:32,34:end]))
% 
% subplot(1,2,2)
% boxplot(BottomBackgr(:,8:end))
% title('All GPA for bottom 25%')


% subplot(1,2,1)
boxplot([TopBackgr(:,8),BottomBackgr(:,8)],'labels',{'Top 25%','Bottom 25%'})
title('High School GPA of Top 25% and bottom 25%')
ylabel('GPA')


% 
% subplot(1,2,2)
% boxplot(BottomBackgr(:,8))
% title('Bottom 25%')

% figure 
% subplot(1,2,1)
% boxplot(Top(:,[37,39,40]))
% 
% subplot(1,2,2)
% boxplot(Bottom(:,[37,39,40]))
% legend

% figure
% subplot(1,2,1)
% boxplot(TopBackgr,'orientation','horizontal','labels',names)
% title('All background information for top 25%')
% % % imagesc(Top(:,[2:32,34:end]))
% % 
% subplot(1,2,2)
% boxplot(BottomBackgr,'orientation','horizontal','labels',names)
% title('All background information for bottom 25%')

%% Decision tree
% [num2,txt2,raw2] = xlsread('train.xlsx');
% [num3,txt3,raw3] = xlsread('test.xlsx');
% 
% X = num2(:,[2:47,49:53]);
% Y = num2(:,48);
% tree = fitctree(X,Y);
% view(tree,'Mode','graph')
% 
% test = num3(:,[2:47,49:53]);
% 
% yfit = predict(tree, test);     % From prediction result
% 
% result = num3(:,48);        % From original data set
% 
% countaccurate = 0;
% posacc = 0;
% negacc = 0;
% 
% for i = 1:size(yfit,1)
%     
%     
%     if and(result(i) ~= yfit(i),result(i)==1)   % positive class
%         posacc = posacc + 1;
%         countaccurate = countaccurate + 1;
%     elseif and(result(i) ~= yfit(i),result(i)==0) % negative class
%         negacc = negacc + 1;
%         countaccurate = countaccurate + 1;
%     end
% 
% end
% 
% posaccuracy = posacc/sum(result==1)
% negaccuracy = negacc/sum(result==0)
% 
% accuracy = countaccurate/size(yfit,1)


%% Treebagger

[num2,txt2,raw2] = xlsread('train.xlsx');
[num3,txt3,raw3] = xlsread('test.xlsx');

X = num2(:,[2:47,49:53]);
Y = num2(:,48);

NTrees = 35
tree = TreeBagger(NTrees,X,Y,'oobpred','on','NVarToSample', 50);
treeb = oobError(tree);

figure
plot(treeb)
% view(tree,'Mode','graph')

test = num3(:,[2:47,49:53]);

[yfit,score] = predict(tree, test); % From prediction result
yfit = str2double(yfit)

result = num3(:,48);        % From original data set


EVAL = Evaluate(yfit,result);

posclass = 1;

[CX,CY,T,AUC] = perfcurve(yfit,score(:,1),posclass);

AUC

figure
plot(CX,CY)

countaccurate = 0;
posacc = 0;
negacc = 0;

for i = 1:size(yfit,1)
    
    
    if and(result(i) ~= yfit(i),result(i)==1)   % positive class
        posacc = posacc + 1;
        countaccurate = countaccurate + 1;
    elseif and(result(i) ~= yfit(i),result(i)==0) % negative class
        negacc = negacc + 1;
        countaccurate = countaccurate + 1;
    end

end

posaccuracy = posacc/sum(result==1)
negaccuracy = negacc/sum(result==0)

accuracy = countaccurate/size(yfit,1)



