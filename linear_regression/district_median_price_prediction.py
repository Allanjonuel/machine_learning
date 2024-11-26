from pathlib import Path
from zlib import crc32

import numpy as np
import pandas as pd
import tarfile
import urllib.request

from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")

    return pd.read_csv("datasets/housing/housing.csv")


def shuffle_and_split_data(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    train_indices = shuffled_indices[test_set_size:]
    test_indices = shuffled_indices[:test_set_size]

    return data.iloc[train_indices], data.iloc[test_indices]


def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32


def split_data_with_id_hash(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing = load_housing_data()
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

housing_with_id = housing.reset_index()  # adds an `index` column
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

print("Features in the dataset")
print(housing.columns)

print("Size of the dataset:")
print(housing.shape)

print("First five rows of the dataset:")
print(housing.head())

print("Info:")
print(housing.info())

print("Summary Statistics:")
print(housing.describe())

print("Correlations:")
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# cleaning data
# housing.dropna(subset=["total_bedrooms"], inplace=True)  # option 1

# housing.drop("total_bedrooms", axis=1)  # option 2

imputer = SimpleImputer(strategy="median")
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

#
# housing['income_cat'] = pd.cut(housing["median_income"],
#                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
#                                labels=[1, 2, 3, 4, 5])
# # extra code – the next 5 lines define the default font sizes
# plt.rc('font', size=14)
# plt.rc('axes', labelsize=14, titlesize=14)
# plt.rc('legend', fontsize=14)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
#
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1, grid=True)

# housing.hist(bins=50, figsize=(12, 8))
# housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
# plt.xlabel("Income category")
# plt.ylabel("Number of districts")

# housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
#              s=housing["population"] / 100, label="population",
#              c="median_house_value", cmap="jet", colorbar=True,
#              legend=True, sharex=False, figsize=(10, 7))
plt.show()
