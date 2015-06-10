%%


[num,txt,raw] = xlsread('OnlyCourses.xlsx');

course_1 = max(num(:,6), num(:,7));
course_2 = max(num(:,8), num(:,9));
course_3 = max(num(:,10), num(:,11));
course_4 = max(num(:,12), num(:,13));
course_5 = max(num(:,14), num(:,15));
course_6 = max(num(:,16), num(:,17));
 %%
 courses_1_6 = [course_1 course_2 course_3 course_4 course_5 course_6];
 
 
 
 
%%

xlswrite('courses1_6.xlsx',courses_1_6)
