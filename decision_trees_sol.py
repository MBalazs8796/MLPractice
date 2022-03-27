#%%
import pandas as pd
from sklearn import tree
from df_preprocessor import load_and_process
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
#%%
keepList = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']
df, test_df = load_and_process("csv/train_input.csv", "csv/test_input.csv", keepList)
#%%
y_train = df['target']
model = tree.DecisionTreeClassifier(max_depth=4)
model = model.fit(df.drop('target', axis=1).values, y_train.values)
y_test = test_df['target']
y_pred = model.predict(test_df.drop('target', axis=1).values)
#%%
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('accuracy: ' + str(accuracy))
print('f1: ' + str(f1))
#%%
params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [4, 5, 6],
    'max_features': [None, 'auto', 'log2'],
    'class_weight': [None, 'balanced']
}
clf = GridSearchCV(tree.DecisionTreeClassifier(), params, scoring='f1', cv=10, refit=True, return_train_score=True)
clf.fit(df.drop('target', axis=1).values, y_train)
print('Best-params:', clf.best_params_)
print('Best-score:', clf.best_score_)
print("Accuracy: " + str(accuracy_score(y_test, clf.predict(test_df.drop('target', axis=1).values))))
print("F1 score: " + str(f1_score(y_test, clf.predict(test_df.drop('target', axis=1).values))))
