from df_preprocessor import load_and_process
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV


keepList = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']
train, test = load_and_process("csv/train_input.csv", "csv/test_input.csv", keepList)

x_train = train.drop('target', axis=1)
x_test = test.drop('target', axis=1)

y_train = train['target']
y_test = test['target']

param = {
	'class_weight': 'balanced_subsample',
	'criterion': 'gini',
	'max_depth': 11,
	'max_features': 'sqrt',
	'n_estimators': 50
}
rf = RandomForestClassifier(**param)

param = {
	'C': 0.7,
	'gamma': 'scale',
	'kernel': 'linear'
}
svm = BaggingClassifier(SVC(**param), max_samples=1/10, n_estimators=10, n_jobs=10)

param = param = {
    'max_depth': 4,
    'eta' : 0.1,
    'objective' : 'binary:hinge',
    'min_child_weight' : 3,
    'grow_policy' : 'lossguide',
    'max_leaves' : 5,
    'tree_method' : 'gpu_hist',
    'max_bin' : 256
}
xgb = XGBClassifier(**param)

voting = VotingClassifier(estimators=
	[('rf', rf), ('svm', svm), ('xgb', xgb)],
	voting='soft',
	n_jobs=7
)
voting.fit(x_train, y_train)

preds = voting.predict(x_test)
print(accuracy_score(y_test,preds))
print(f1_score(y_test, preds))