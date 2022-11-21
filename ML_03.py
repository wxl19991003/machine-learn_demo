from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


housing = fetch_california_housing()
print(housing.data.shape)
x_train, x_test, y_train, y_test = train_test_split(housing.data, housing.target, random_state=42)
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)


def linear1():
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print("权重为：", estimator.coef_)
    print("偏置为：", estimator.intercept_)
    error = mean_squared_error(y_predict, y_test)
    print("均方误差为：", error)


def linear2():
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)
    y_predict = estimator.predict(x_test)
    print("权重为：", estimator.coef_)
    print("偏置为：", estimator.intercept_)
    error = mean_squared_error(y_predict, y_test)
    print("均方误差为：", error)


def linear3():
    # estimator = Ridge()
    # estimator.fit(x_train, y_train)
    # joblib.dump(estimator, "my_ridge.pkl")
    estimator = joblib.load("my_ridge.pkl")
    y_predict = estimator.predict(x_test)
    print("权重为：", estimator.coef_)
    print("偏置为：", estimator.intercept_)
    error = mean_squared_error(y_predict, y_test)
    print("均方误差为：", error)


if __name__ == "__main__":
    # linear1()
    # linear2()
    linear3()
