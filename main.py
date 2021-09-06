import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class KaggleTitanic:
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    pd.options.mode.chained_assignment = None

    def main(self):
        df = self.data_read()
        x, y = self.data_process(df)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        model_tree = DecisionTreeClassifier(max_depth=3)
        model_random_forest = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, max_features='auto', criterion='gini')
        # self.find_better_params(x_train, y_train)
        model_tree.fit(x_train, y_train)
        y_predict = model_tree.predict(x_test)

        # depth = []
        # for i in range(3, 20):
        #     clf = DecisionTreeClassifier(max_depth=i)
        #     scores = cross_val_score(estimator=clf, X=x, y=y, cv=7, n_jobs=4)
        #     depth.append((i, scores.mean()))
        # print(depth)

        print(accuracy_score(y_test, y_predict))

    @staticmethod
    def data_read():
        return pd.read_csv('data/train.csv')

    def data_process(self, df):
        df = df.drop(['PassengerId', 'Ticket', 'Fare', 'Cabin', 'Pclass'], axis=1)

        df = self.sex_process(df)
        df = self.place_embarked_process(df)
        df = self.title_process(df)
        df = self.ages_fill(df)
        df = self.relatives_process(df)

        df = df.drop(['Name', 'Title'], axis=1)

        # print(df.head)
        x, y = self.answer_separate(df)
        return x, y

    @staticmethod
    def sex_process(df):
        sex_mapping = {'male': 0, 'female': 1}
        df['Sex'] = df['Sex'].map(sex_mapping)
        return df

    @staticmethod
    def place_embarked_process(df):
        embarked_dummies = pd.get_dummies(df['Embarked'], prefix='port', dummy_na=False)
        df = pd.concat([df, embarked_dummies], axis=1)
        df = df.drop(['Embarked'], axis=1)
        return df

    def title_process(self, df):
        df['Title'] = df['Name'].apply(lambda x: self.name_to_title(x))
        # title_dummies = pd.get_dummies(df['Title'], prefix='title', dummy_na=False)
        # df = pd.concat([df, title_dummies], axis=1)
        return df

    @staticmethod
    def name_to_title(name):
        crew = ['Capt', 'Col', 'Dr', 'Major', 'Rev']
        royalty = ['Dona', 'Lady', 'the Countess', 'Sir', 'Jonkheer', 'Don']
        title = re.findall(r',\s[A-Za-z\s]+\.', name)
        title = 'NaN' if len(title) == 0 else title[0][2:-1]
        if title in royalty:
            title = 'Royalty'
        elif title in crew:
            title = 'Crew'
        elif title == 'Mlle' or title =='Ms':
            title = 'Miss'
        elif title == 'Mme':
            title = 'Mrs'
        return title

    def ages_fill(self, df):
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        pd.options.mode.chained_assignment = None

        age_null = df.loc[df['Age'].isnull()]
        age_not_null = df.loc[~df['Age'].isnull()]

        age_null_wo_relative = self.fill_wo_relatives(age_null, age_not_null)
        age_null_with_relative = self.fill_by_titles(age_null, age_not_null)

        df = pd.concat([age_null_wo_relative, age_null_with_relative])
        df['Age'] = df['Age'].astype(np.float64)
        return df

    def fill_wo_relatives(self, age_null, age_not_null):
        age_not_null_wo_relative = age_not_null.loc[(age_not_null['SibSp'] == 0) & (age_not_null['Parch'] == 0)]
        age_null_wo_relative = age_null.loc[(age_null['SibSp'] == 0) & (age_null['Parch'] == 0)]

        list_ages = self.get_distribution_law(age_not_null_wo_relative['Age'])
        age_null_wo_relative['Age'] = self.generate_ages(age_null_wo_relative['Age'], list_ages)
        return age_null_wo_relative

    def fill_by_titles(self, age_null, age_not_null):
        pd.options.display.max_columns = None
        pd.options.display.max_rows = None
        age_not_null_with_relative = age_not_null.loc[(age_not_null['SibSp'] > 0) | (age_not_null['Parch'] > 0)]
        age_null_with_relative = age_null.loc[(age_null['SibSp'] > 0) | (age_null['Parch'] > 0)]

        list_titles = ['Master', 'Mr', 'Miss', 'Mrs']
        answer_null = None
        for title in list_titles:
            current_title_not_null = age_not_null_with_relative.loc[age_not_null_with_relative['Title'] == title]
            current_title_null = age_null_with_relative.loc[age_null_with_relative['Title'] == title]
            list_ages = self.get_distribution_law(current_title_not_null['Age'])
            current_title_null['Age'] = self.generate_ages(current_title_null['Age'], list_ages)

            if answer_null is None:
                answer_null = current_title_null
            else:
                answer_null = pd.concat([answer_null, current_title_null])
        return answer_null

    @staticmethod
    def fill_mean(series):
        return series.fillna(value=series.mean())

    @staticmethod
    def get_distribution_law(series):
        max_v = int(series.max())
        min_v = int(series.min())
        series = series.astype(int)
        series_dict = dict(series.value_counts())
        list_distribution = []

        for i in range(min_v, max_v, 1):
            if i not in series_dict:
                series_dict[i] = 1
            for j in range(series_dict[i]):
                list_distribution.append(i)
        return list_distribution

    @staticmethod
    def generate_ages(series, list_distribution):
        series = dict(series)
        for key, item in series.items():
            series[key] = list_distribution[random.randint(0, len(list_distribution)-1)]
        return pd.Series(series)

    def relatives_process(self, df):
        df['Family'] = df.apply(lambda x: self.union_relatives(x['SibSp'], x['Parch']), axis=1)
        df = df.drop(['SibSp', 'Parch'], axis=1)
        return df

    @staticmethod
    def union_relatives(sibsp, parch):
        if sibsp + parch == 0:
            relatives = 1
        else:
            relatives = sibsp + parch + 1
        return relatives

    @staticmethod
    def answer_separate(df):
        y = df['Survived']
        x = df.drop(['Survived'], axis=1)
        return x, y


    @staticmethod
    def find_better_params(X, y):
        rfc = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt', 'log2'],
            'max_depth': [4, 5, 6, 7, 8],
            'criterion': ['gini', 'entropy']
        }
        GS_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        GS_rfc.fit(X, y)
        print(GS_rfc.best_params_)


if __name__ == '__main__':
    kaggle_titanic = KaggleTitanic()
    kaggle_titanic.main()
