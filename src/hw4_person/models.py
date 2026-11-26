"""本作业内复用的模型构建与输出转换。"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .data import CLASSES
from .data import NUMERIC, CATEGORICAL


def build_model(name, seed=42, epochs=30, trees=100, features='word'):
    numeric = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True), StandardScaler())
    preprocess = ColumnTransformer([('numeric', numeric, NUMERIC),
                                    ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL)])
    estimators = {'logistic': LogisticRegression(max_iter=1000, random_state=seed),
                  'extra_trees': ExtraTreesClassifier(n_estimators=trees, min_samples_leaf=3, random_state=seed, n_jobs=-1),
                  'mlp': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=epochs, random_state=seed)}
    return make_pipeline(preprocess, estimators[name])


def predict_values(model, x):
    return model.predict(x)


def training_target(y):
    return y
