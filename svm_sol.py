from df_preprocessor import load_and_process
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV


keepList = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']
train, test = load_and_process("csv/train_input.csv", "csv/test_input.csv", keepList)

x_train = train.drop('target', axis=1)
x_test = test.drop('target', axis=1)

y_train = train['target']
y_test = test['target']

model = SVC(C=.003, kernel="linear")
model.fit(x_train, y_train)

preds = model.predict(x_test)

print(accuracy_score(y_test,preds))
print(f1_score(y_test, preds))

params = {
	'kernel': ['linear', 'poly', 'rbf'],
	'C': [0.003, 0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0],
	'gamma': ['scale', 'auto']
}

gs = GridSearchCV(SVC(), params, scoring='f1', cv=10, refit=True, return_train_score=True, n_jobs=-1, verbose=10)
gs.fit(x_train, y_train)

print(gs.best_params_)
print(gs.best_score_)

"""
{'C': 0.7, 'gamma': 'scale', 'kernel': 'linear'}
0.5796673393692781
"""