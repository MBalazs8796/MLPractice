from df_preprocessor import load_and_process
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV


keepList = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']
train, test = load_and_process("csv/train_input.csv", "csv/test_input.csv", keepList)

x_train = train.drop('target', axis=1)
x_test = test.drop('target', axis=1)

y_train = train['target']
y_test = test['target']

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

preds = rf.predict(x_test)

print(accuracy_score(y_test,preds))
print(f1_score(y_test, preds))

params = {
	'max_depth': [10, 11, 12, 13, 14, 15, 20, 25, 30, 40, 50, 75, 100, 1000],
	'n_estimators': [10, 25, 50, 75, 90, 100, 110, 125, 150, 200, 300, 500, 1000],
	'criterion': ['gini', 'entropy']
}

gs = GridSearchCV(RandomForestClassifier(), params, scoring='f1', cv=10, refit=True, return_train_score=True, n_jobs=-1, verbose=10)
gs.fit(x_train, y_train)

print(gs.best_params_)
print(gs.best_score_)

"""
{'criterion': 'entropy', 'max_depth': 14, 'n_estimators': 110}
0.5334207409343039
"""