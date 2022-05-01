# %%
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from df_preprocessor import load_and_process
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

# %%
first_run = 0

# %%
keep = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university', 'education_level',
        'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']
df, test_df = load_and_process("csv/train_input.csv", "csv/test_input.csv", keep)
y_train = df['target']
train = df.drop('target', axis=1).values
y_test = test_df['target']
test = test_df.drop('target', axis=1).values
# %%
gnb = GaussianNB()
y_pred = gnb.fit(train, y_train).predict(test)
print("f1:", f1_score(y_test, y_pred))
print("score:", gnb.score(test, y_test))
# %%
bnb = BernoulliNB()
y_pred = bnb.fit(train, y_train).predict(test)
print("f1:", f1_score(y_pred, y_test))
print("score:", bnb.score(test, y_test))
# %%
cnb = CategoricalNB()
y_pred = cnb.fit(train, y_train).predict(test)
print("f1:", f1_score(y_pred, y_test))
print("score:", cnb.score(test, y_test))
# %%
conb = ComplementNB()
y_pred = conb.fit(train, y_train).predict(test)
print("f1:", f1_score(y_pred, y_test))
print("score:", conb.score(test, y_test))
# %%
mnb = MultinomialNB()
y_pred = mnb.fit(train, y_train).predict(test)
print("f1:", f1_score(y_pred, y_test))
print("score:", mnb.score(test, y_test))
# %%
rfc = RandomForestClassifier()
y_pred = rfc.fit(train, y_train).predict(test)
print("f1:", f1_score(y_pred, y_test))
print("score:", rfc.score(test, y_test))
# %%

cfig = {
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample'],
    'max_depth': [13, 14, 15]
}

clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), cfig, scoring='f1', cv=10, refit=True, return_train_score=True)
clf.fit(df.drop('target', axis=1).values, y_train)
print('Best-params:', clf.best_params_)
print('Best-score:', clf.best_score_)

# Best-params: {'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': 14, 'max_features': 'log2'}
# Best-score: 0.5823347531177112

# %%

cfig = {
    'criterion': ['gini', 'entropy'],
    'max_features': ['auto', 'sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample'],
    'max_depth': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
}

clf = GridSearchCV(RandomForestClassifier(n_jobs=-1), cfig, scoring='roc_auc', cv=10, refit=True,
                   return_train_score=True)
clf.fit(df.drop('target', axis=1).values, y_train)
print('Best-params:', clf.best_params_)
print('Best-score:', clf.best_score_)

# Best-params: {'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 10, 'max_features': 'auto'}
# Best-score: 0.7799342490327067

# %%

tuned_params = {'random_state': list(range(1, 101))}
cfig = {'class_weight': 'balanced_subsample', 'criterion': 'gini', 'max_depth': 14, 'max_features': 'log2'}
classifiers = []
for v in tuned_params['random_state']:
    rfc = RandomForestClassifier(random_state=v, n_jobs=-1, class_weight='balanced_subsample', criterion='gini',
                                 max_depth=14, max_features='log2')
    rfc.fit(train, y_train)
    classifiers.append(rfc)
y_pred = rfc.predict(test)
print("f1:", f1_score(y_pred, y_test))

# f1: 0.5859375000000001
# f1: 0.5823347531177112 W/o random

#%%
rfc = RandomForestClassifier(class_weight='balanced_subsample', criterion='gini', max_depth=14, max_features='log2')
y_pred = rfc.fit(train, y_train).predict(test)
print("f1:", f1_score(y_pred, y_test))
print("score:", rfc.score(test, y_test))

#%%
print(train.shape)
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(train, y_train)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(train)
print(X_new.shape)

#%%
test_on_less = False
for i in range(0, 4):
    keep = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university', 'education_level',
            'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']
    df, test_df = load_and_process("csv/train_input.csv", "csv/test_input.csv", keep)
    df = df.sample(frac=1-0.25*i)
    if test_on_less:
        test_df = test_df.sample(frac=1-0.25*i)
    print("Number of samples:", len(df))
    y_train = df['target']
    train = df.drop('target', axis=1).values
    y_test = test_df['target']
    test = test_df.drop('target', axis=1).values
    rfc = RandomForestClassifier(class_weight='balanced_subsample', criterion='gini', max_depth=14, max_features='log2')
    y_pred = rfc.fit(train, y_train).predict(test)
    print("f1:", f1_score(y_pred, y_test))

# Number of samples: 11285
# f1: 0.5831702544031312
# Number of samples: 8464
# f1: 0.5859375000000001
# Number of samples: 5642
# f1: 0.5776031434184676
# Number of samples: 2821
# f1: 0.5675146771037182

#%%
print(train.shape)
rfc = RandomForestClassifier(class_weight='balanced_subsample', criterion='gini', max_depth=14, max_features='log2')\
    .fit(train, y_train)
model = SelectFromModel(rfc, prefit=True)
train_new = model.transform(train)
new_test = model.transform(test)
print(train_new.shape)

y_pred = RandomForestClassifier(class_weight='balanced_subsample', criterion='gini', max_depth=10, max_features='log2') \
    .fit(train_new, y_train).predict(new_test)
print("f1:", f1_score(y_pred, y_test))
# (11285, 139)
# (11285, 18)
# f1: 0.5831702544031312
# Dimension is lowered but the score is about the same
#%%

