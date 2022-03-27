#%%
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
from df_preprocessor import load_and_process
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import binom_test
from sklearn.model_selection import GridSearchCV
#%%
keepList = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university', 'education_level', 'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']
df, test_df = load_and_process("csv/train_input.csv", "csv/test_input.csv", keepList)
#%%
y_train = df['target']
train = xgb.DMatrix(df.drop('target', axis=1).values, label=y_train.values)
#%%
y_test = test_df['target']
test = xgb.DMatrix(test_df.drop('target', axis=1).values, label=y_test.values)
# %%
param = {
    'max_depth': 4,
    'eta' : 0.15,
    'objective' : 'binary:hinge',
    'min_child_weight' : 3
}
epochs = 10
model = xgb.train(param, train, epochs)
preds = model.predict(test)
#f1_score(y_test, np.zeros(preds.shape[0]))

test_acc = accuracy_score(y_test,preds)
n_success = np.sum(y_test==preds)
p=0.90
interval = binom_test(n_success,y_test.shape[0],p=p)
print("Test-acc: ",test_acc,"+/-",interval,"(",p*100,"%)")
print("F1 score: " + str(f1_score(y_test, preds)))

#%%

params = [{
    'max_depth': [4,5,6],
    'eta' : [0.5, 0.1, 0.15, 0.2],
    'min_child_weight' : [1,2,3]
}]
clf = GridSearchCV(XGBClassifier(objective='binary:hinge'), params, scoring='f1',cv=10, refit = True, return_train_score=True)
clf.fit(df.drop('target', axis=1).values,y_train)
print('Best-params:',clf.best_params_)
print('Best-score:',clf.best_score_)
print("F1 score: " + str(f1_score(y_test, clf.predict(test))))
