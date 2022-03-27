# %%
from sklearn.naive_bayes import GaussianNB, BernoulliNB, CategoricalNB, ComplementNB, MultinomialNB
from df_preprocessor import load_and_process
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

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
print("score:", mnb.score(test, y_test))
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
