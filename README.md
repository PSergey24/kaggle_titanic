# kaggle_titanic
My decision for kaggle "Titanic".
https://www.kaggle.com/c/titanic


1) I splitted train.csv into train and test sample (the ratio of 8 to 2 and 7 to 3). Results was same.
2) There are idea that not all сolumns affect the result. For example, 'PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'

3) I checked several different models using SKlearn library
i got next results:<br />
LinearRegression ~44,4% for test sample<br />
DecisionTreeClassifier ~79,8%<br />
RandomForestClassifier ~81,56%<br />

4) Using GridSearchCV i got best params for RandomForestClassifier.<br />
I got {'criterion': 'entropy', 'max_depth': 6, 'max_features': 'auto', 'n_estimators': 200}<br />
same result ~81,56%
'gini' instead of 'entropy': 82,12%

5) Idea with title was taken here: https://www.kaggle.com/blastchar/titanic-the-machine-learning-fix-some-mistakes <br />

6) The 'Age' column was filled with mean values for all gaps<br />
Other decision is to get distribution law for people without relatives and after that to generate Ages for such people.<br /> For people with relatives to generate Ages using titles. <br /> After that accuracy score increased to 91-94%