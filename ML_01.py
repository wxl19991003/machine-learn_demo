from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA


def datasets_demo():
    # 加载数据集
    iris = load_iris()
    print("鸢尾花数据集:\n", iris)

    # 分割数据集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)
    print("x_train=: \n", x_train)
    print("y_train=: \n", y_train)

    return None


# 文本提取
def dict_demo():
    data = [{'city': '北京', 'temp': 100}, {'city': '上海', 'temp': 80}, {'city': '深圳', 'temp': 60}]
    transfer = DictVectorizer(sparse=False)
    data = transfer.fit_transform(data)
    print("数据", data)
    print("名称", transfer.get_feature_names_out())

    return None


def text_count_demo():
    data = ['life is short,i use use python', 'life is long,i dislike python']
    transfer = CountVectorizer()
    data = transfer.fit_transform(data)
    print(data.toarray())
    print(transfer.get_feature_names_out())

    return None


def chinese_text():
    data = ['人生 苦 短,我 选 Python', '人生 很长,我 拒绝 Python']
    transfer = CountVectorizer()
    data = transfer.fit_transform(data)
    print(transfer.get_feature_names_out())
    print(data.toarray())

    return None


def cut_text(text):
    text = " ".join(list(jieba.cut(text)))

    return text


def chinese_text_02():
    data = ['一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
            '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
            '如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。']
    text_list = []
    for sent in data:
        text_list.append(cut_text(sent))
    transfer = CountVectorizer()
    data = transfer.fit_transform(text_list)
    print(transfer.get_feature_names_out())
    print(data.toarray())

    return None


def tfidf_demo():
    data = ['一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。',
            '我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。',
            '如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。']
    text_list = []
    for sent in data:
        text_list.append(cut_text(sent))
    transfer = TfidfVectorizer(stop_words=['不要', '我们', '发出'])
    data = transfer.fit_transform(text_list)
    print(transfer.get_feature_names_out())
    print(data.toarray())

    return None


# 归一化
def minmax_demo():
    data = pd.read_csv("dating.txt")
    print(data)
    transfer = MinMaxScaler()
    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print("最小值最大值归一化处理的结果：\n", data)

    return None


# 标准化
def stand_demo():
    data = pd.read_csv("dating.txt")
    print(data)
    transfer = StandardScaler()
    data = transfer.fit_transform(data[['milage', 'Liters', 'Consumtime']])
    print("最小值最大值标准化处理的结果：\n", data)

    return None


# 低方差特征过滤
def variance_demo():
    data = pd.read_csv("factor_returns.csv")
    print(data)
    transfer = VarianceThreshold(threshold=1)
    # 2、调用fit_transform
    data = transfer.fit_transform(data.iloc[:, 1:10])  # python list:[0,1,2...-3,-2,-1](左闭右开）
    print("删除低方差特征的结果：\n", data)
    print("形状：\n", data.shape)

    return None


# 皮尔森相关系数
def pearsonr_demo():
    data = pd.read_csv("factor_returns.csv")

    factor = ['pe_ratio', 'pb_ratio', 'market_cap', 'return_on_asset_net_profit', 'du_return_on_equity', 'ev',
              'earnings_per_share', 'revenue', 'total_expense']

    for i in range(len(factor)):
        for j in range(i, len(factor) - 1):
            print(
                "指标%s与指标%s之间的相关性大小为%f" % (factor[i], factor[j + 1], pearsonr(data[factor[i]], data[factor[j + 1]])[0]))

    return None


def pca_demo():
    data = [[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]]
    transfer = PCA(n_components=0.9)
    data1 = transfer.fit_transform(data)
    print("保留90%的信息，降维结果为：\n", data1)

    transfer2 = PCA(n_components=3)
    data2 = transfer2.fit_transform(data)
    print("降维到3维的结果：\n", data2)

    return None

if __name__ == "__main__":
    # datasets_demo()
    # dict_demo()
    # text_count_demo()]
    # chinese_text()
    # chinese_text_02()
    # tfidf_demo()
    # minmax_demo()
    # stand_demo()
    # variance_demo()
    # pearsonr_demo()
    pca_demo()