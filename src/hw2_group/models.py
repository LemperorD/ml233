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


def build_model(name='logistic', seed=42, epochs=30, trees=100, features='word'):
    word = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, sublinear_tf=True)
    char = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True)
    vectorizer = {'word': word, 'char': char, 'both': FeatureUnion([('word', word), ('char', char)])}[features]
    estimators = {'logistic': LogisticRegression(max_iter=1000, C=3, random_state=seed),
                  'naive_bayes': MultinomialNB(alpha=0.5),
                  'mlp': MLPClassifier(hidden_layer_sizes=(64,), max_iter=epochs, random_state=seed)}
    return make_pipeline(vectorizer, estimators[name])


def predict_values(model, x):
    probabilities = model.predict_proba(x)
    return probabilities[:, [list(model.classes_).index(name) for name in CLASSES]]


def training_target(y):
    return y
