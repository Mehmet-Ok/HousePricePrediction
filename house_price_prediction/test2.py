import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import missingno as msno
from datetime import date

from clyent import color
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    MinMaxScaler,
    LabelEncoder,
    StandardScaler,
    RobustScaler,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: "%.3f" % x)
pd.set_option("display.width", 500)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"
    ]
    cat_but_car = [
        col
        for col in dataframe.columns
        if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"
    ]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe.isnull().sum().sort_values(ascending=False)
    ratio = (dataframe.isnull().sum() / dataframe.shape[0] * 100).sort_values(
        ascending=False
    )
    missing_df = pd.concat(
        [n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"]
    )
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(
            temp_df[col].isnull(), 1, 0
        )  # eksik olanlar = 1, olmayanalar = 0
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(
            pd.DataFrame(
                {
                    "TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                    "COUNT": temp_df.groupby(col)[target].count(),
                }
            ),
            end="\n\n",
        )


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


def cat_summary(dataframe, col_name, plot=False):
    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)
    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list


def cat_to_target_mean(dataframe, col_name, target):
    print(dataframe.groupby(col_name)[target].mean())


def target_to_num_mean(dataframe, col_name, target):
    print(dataframe.groupby(target).agg({col_name: "mean"}))


def outlier_trasholds(dataframe, col_name, q1=0.25, q3=0.75):
    qrt1 = dataframe[col_name].quantile(q1)
    qrt3 = dataframe[col_name].quantile(q3)
    iqr = qrt3 - qrt1
    upper = qrt3 + 1.5 * iqr
    lower = qrt1 - 1.5 * iqr
    return lower, upper


def check_outlier(dataframe, col_name):
    lo, up = outlier_trasholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < lo) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe.isnull().sum().sort_values(ascending=False)
    ratio = (dataframe.isnull().sum() / dataframe.shape[0] * 100).sort_values(
        ascending=False
    )
    missing_df = pd.concat(
        [n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"]
    )
    print(missing_df, end="\n")
    if na_name:
        return na_columns


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(
            temp_df[col].isnull(), 1, 0
        )  # eksik olanlar = 1, olmayanalar = 0
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(
            pd.DataFrame(
                {
                    "TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                    "COUNT": temp_df.groupby(col)[target].count(),
                }
            ),
            end="\n\n",
        )


def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def load_data():
    df = pd.read_csv("datasets/diabetes.csv")
    return df


df = load_data()
x = df.drop(columns="Outcome",axis=1)
y = df["Outcome"]
# plt.scatter(x=x,y=y)
# plt.show(block=True)

for col in x.columns:
    plt.scatter(x=col, y=y)
    plt.show(block=True)

sns.scatterplot(data=df.drop(columns="Outcome"),y=df["Outcome"])
plt.show(block=True)


df.columns = [col.lower() for col in df.columns]
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if col not in "outcome"]

# kat degisken_target
[cat_to_target_mean(df, col_name, "outcome") for col_name in cat_cols]

# target_num degisken
[target_to_num_mean(df, col_name, "outcome") for col_name in num_cols]

# aykırı gözlem
[print(col_name, ": ", check_outlier(df, col_name)) for col_name in num_cols]

# eksik gözlem
"""
veri setinde eksik gözlemler 0 değerini almış olabilir
"""

cat_cols, num_cols, cat_but_car = grab_col_names(df)

[print(col, ": ", df[col].nunique()) for col in num_cols]

msno.bar(df)
plt.show(block=True)

for col in num_cols:
    if df[col].nunique() > 20:
        df.loc[df[col] == 0, col] = np.nan

na_cols = missing_values_table(df, True)
missing_vs_target(df, target="outcome", na_columns=na_cols)

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
dff.head()

scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)
dff.head()

from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

# missing_values_table(dff)
dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

dff.isnull().sum()
for col in num_cols:
    dff.loc[df[col] == np.nan, col] = 0

# korelasyon analizi
import scipy.stats as stats


def compare_coefficients(x, y):
    """Compares the coefficients of two variables.

    Args:
      x: The first variable.
      y: The second variable.

    Returns:
      The t-statistic and p-value for the difference in coefficients.
    """

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_std = np.std(x)
    y_std = np.std(y)

    t_statistic = (x_mean - y_mean) / (x_std / np.sqrt(len(x)))
    p_value = 1 - stats.t.cdf(t_statistic, len(x) - 1)

    return t_statistic, p_value


# yeni degiskenler

# yeni kategorikler
bins_bmi = [0, 18.5, 24.9, 29.9, 100]
labels_bmi = ['underweight', 'normal weight', 'overweight', 'obese']
dff["new_bmi_cat"] = pd.cut(dff["bmi"], bins=bins_bmi, labels=labels_bmi)
dff["new_bloodpressure_cat"] = pd.qcut(dff["bloodpressure"], labels=["low", "normal", "high"], q=3)
dff["new_have_child_cat"] = np.where(df["pregnancies"] == 0, 0, 1)

age_bins = [0, 21, 30, 60, 90]
age_labels = ['young_adults', 'adults', 'middle_aged', 'elderly']
dff["new_age_cat"] = pd.cut(dff["age"], bins=age_bins, labels=age_labels)

# yeni nümerikler
dff["new_insulin_glucose"] = dff["insulin"] * dff["glucose"]  # ??? belki???
dff["new_bmi_skin"] = dff["bmi"] * dff["skinthickness"]
dff["new_age_bloodpressure"] = dff["age"] * dff["bloodpressure"]  # yaş ve kan basıncı iliskisi
dff["new_skin_insulin"] = dff["skinthickness"] * dff["insulin"]
dff["new_pregnancies-skinthickness"] = dff["pregnancies"] * dff["skinthickness"]
dff["new_pregnancies_insulin"] = dff["pregnancies"] * dff["insulin"]
dff["new_bloodpressure_insulin"] = dff["bloodpressure"] * dff["insulin"]

dff.head()

new_num_values = [col for col in dff.columns if col.startswith("new_") and "_cat" not in col]
new_cat_values = [col for col in dff.columns if "_cat" in col]

[target_to_num_mean(dff, col, "outcome") for col in new_num_values]
[cat_to_target_mean(dff, col, "outcome") for col in new_cat_values]

cat_cols, num_cols, cat_but_car = grab_col_names(dff)
cat_cols = [col for col in cat_cols if "outcome" not in col]
# encoding


rare_analyser(dff, "outcome", cat_cols)
# rare encodinge gerek yok gibi??

# label encode
binary_cols = [col for col in dff.columns if dff[col].dtype not in ["int64", "float64"] and dff[col].nunique() == 2]

for col in binary_cols:
    dff = label_encoder(dff, col)

# one-hot encode
ohe_cols = [col for col in dff.columns if 10 >= dff[col].nunique() > 2]

dff = one_hot_encoder(dff,ohe_cols)

#stantardlastirma
#min-max
dff_standardized = dff.copy()

scaler = MinMaxScaler()

dff_standardized = pd.DataFrame(scaler.fit_transform(dff_standardized),columns=dff_standardized.columns)

#model kurulumu
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

y = dff_standardized["outcome"]
X = dff_standardized.drop("outcome",axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=17)

model = RandomForestClassifier(random_state=42).fit(X_train,y_train)
y_predict = model.predict(X_test)
accuracy_score(y_predict,y_test)

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

plot_importance(model, X, save=True)


from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

accuracy_score(y_test,y_pred_dt)
plot_importance(dt_model, X, save=True)





# data = load_data()
# y = data["Outcome"]
# X = data.drop("Outcome",axis=1)
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=17)
# #
# # model = RandomForestClassifier(random_state=42).fit(X_train,y_train)
# # y_predict = model.predict(X_test)
# # accuracy_score(y_predict,y_test)
#
# dt_model = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
# y_pred_dt = model.predict(X_test)
# accuracy_score(y_test,y_pred_dt)
# plot_importance(dt_model, X, save=True)