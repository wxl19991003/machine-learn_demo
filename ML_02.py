from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def knn_iris_gscv():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.transform(x_test)
    knn = KNeighborsClassifier()
    param = {"n_neighbors": [1, 3, 5, 7, 9, 11]}
    knn = GridSearchCV(knn, param_grid=param, cv=10)
    knn.fit(x_train, y_train)
    y_predict = knn.predict(x_test)
    print(y_predict == y_test)  # 查看是否与预测值相符
    print(knn.score(x_test, y_test))  # 准确率
    # 网格搜索结果
    print(knn.best_params_)
    print(knn.best_estimator_)
    print(knn.best_score_)

    return None


def nbcls():
    news = fetch_20newsgroups(subset='all')
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.3)
    tf = TfidfVectorizer()
    x_train = tf.fit_transform(x_train)
    x_test = tf.transform(x_test)
    mlb = MultinomialNB(alpha=1.0)
    mlb.fit(x_train, y_train)
    y_predict = mlb.predict(x_test)

    print("预测每篇文章的类别：", y_predict)
    print("预测准确率为：", mlb.score(x_test, y_test))

    return None


def decisiontree_iris():
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    deci = DecisionTreeClassifier(criterion="entropy")
    deci.fit(x_train, y_train)
    y_predict = deci.predict(x_test)
    print(y_predict)
    print("预测准确率为：", deci.score(x_test, y_test))
    export_graphviz(deci, out_file="iris_tree.dot", feature_names=iris.feature_names)


# if __name__ == "__main__":
    # knn_iris_gscv()
    # nbcls()
    decisiontree_iris()

