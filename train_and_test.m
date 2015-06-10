[num,txt,raw] = xlsread('NoCoursesWithGPA.xlsx');

% Set random number to an initial seed
[r,c]=size(num);
s=RandStream('mt19937ar','Seed',550);
%generate a permutation of the data
p=randperm(s,r);
num=num(p,:);
%Y=s15(p);
%Use trainpct percent of the data for training and the rest for testing.
trainpct=.75;
train_size=ceil(r*trainpct);

% Grab training and test data
Train = num(1:train_size,:);
Test = num(train_size+1:end,:);

xlswrite('train.xlsx', Train);
xlswrite('test.xlsx', Test);