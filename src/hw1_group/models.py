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
    estimators = {'gradient_boosting': HistGradientBoostingRegressor(max_iter=trees, max_leaf_nodes=31, early_stopping=False, random_state=seed),
                  'random_forest': RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, n_jobs=-1, random_state=seed),
                  'mlp': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=epochs, random_state=seed)}
    return make_pipeline(preprocess, estimators[name])


def predict_values(model, x):
    return np.maximum(0.0, np.expm1(np.clip(model.predict(x), -30, 30)))


def training_target(y):
    return np.log1p(y)
