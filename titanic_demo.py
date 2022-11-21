from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd


def titan_decisioncls():
    titan = pd.read_csv("titanic.csv")
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    x['age'].fillna(x['age'].mean(), inplace=True)
    dic = DictVectorizer(sparse=False)
    x = dic.fit_transform(x.to_dict(orient="records"))

    print(dic.get_feature_names_out())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    dc = DecisionTreeClassifier()
    dc.fit(x_train, y_train)

    print("随机森林预测的准确率为：", gc.score(x_test, y_test))
    print("决策树预测的准确率为：", dc.score(x_test, y_test))
    export_graphviz(dict, out_file="./titan_tree.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd',
                                                                      'pclass=3rd', '女性', '男性'])

    return None


if __name__ == "__main__":
    titan_decisioncls()
