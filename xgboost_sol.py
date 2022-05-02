# %%
from matplotlib import pyplot as plt
import time
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
from df_preprocessor import load_and_process
from sklearn.metrics import f1_score, accuracy_score
from scipy.stats import binom_test
from sklearn.model_selection import GridSearchCV
# %%
PCA = False

keepList = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university',
            'major_discipline', 'experience', 'company_size', 'company_type',  'training_hours', 'target', 'gender']

if PCA:
    df, y_train, test_df, y_test = load_and_process(
        "csv/train_input.csv", "csv/test_input.csv", keepList, PCAtarget=90)
else:
    df, test_df = load_and_process(
        "csv/train_input.csv", "csv/test_input.csv", keepList)

# %%
if PCA:
    train = xgb.DMatrix(df, label=y_train.values)
    test = xgb.DMatrix(test_df, label=y_test.values)
else:
    y_test = test_df['target']
    test = xgb.DMatrix(test_df.drop(
        'target', axis=1).values, label=y_test.values)
    y_train = df['target']
    train = xgb.DMatrix(df.drop('target', axis=1).values, label=y_train.values)
# %%
param = {
    'max_depth': 4,
    'eta': 0.1,
    'objective': 'binary:hinge',
    'min_child_weight': 3,
    'grow_policy': 'lossguide',
    'max_leaves': 5,
    # 'tree_method' : 'gpu_hist',
    'max_bin': 256
}
epochs = 100
model = xgb.train(param, train, epochs)
preds = model.predict(test)
#f1_score(y_test, np.zeros(preds.shape[0]))

test_acc = accuracy_score(y_test, preds)
n_success = np.sum(y_test == preds)
p = 0.90
interval = binom_test(n_success, y_test.shape[0], p=p)
print("Test-acc: ", test_acc, "+/-", interval, "(", p*100, "%)")
print("F1 score: " + str(f1_score(y_test, preds)))

# %%

params = [{
    'max_depth': [1, 2, 3, 4, 5, 6],
    'eta': [0.1, 0.01, 0.2],
    'min_child_weight': [1, 2, 3],
    'grow_policy': ['depthwise', 'lossguide'],
    'max_leaves': [10],
    'max_bin': [256],
    'num_parallel_tree': [1],
    # 'tree_method' : ['gpu_hist']
}]
if PCA:
    clf = GridSearchCV(XGBClassifier(objective='binary:hinge'), params,
                       scoring='f1', cv=10, refit=True, return_train_score=True)
    clf.fit(df, y_train)
else:
    clf = GridSearchCV(XGBClassifier(objective='binary:hinge'), params,
                       scoring='f1', cv=10, refit=True, return_train_score=True)
    clf.fit(df.drop('target', axis=1).values, y_train)
print('Best-params:', clf.best_params_)
print('Best-score:', clf.best_score_)

"""
Test-acc:  0.8553398058252427
F1 score: 0.5895316804407713
"""
# %%
# %%
# test smaller data sets
param = {
    'max_depth': 4,
    'eta': 0.1,
    'objective': 'binary:hinge',
    'min_child_weight': 3,
    'grow_policy': 'lossguide',
    'max_leaves': 5,
    # 'tree_method' : 'gpu_hist',
    'max_bin': 256
}
y_test = test_df['target']
test = xgb.DMatrix(test_df.drop('target', axis=1).values, label=y_test.values)
epochs = 100
fractions = np.linspace(0.2, 1, 10)
fittimes = []
accuracies = []
f1s = []
accuracies_test = []
f1s_test = []
for frac in fractions:
    train_sampled = df.sample(frac=frac)
    y_train = train_sampled['target']
    train = xgb.DMatrix(train_sampled.drop(
        'target', axis=1).values, label=y_train.values)
    starttime = time.time()
    model = xgb.train(param, train, epochs)
    fittimes += [time.time()-starttime]
    preds_train = model.predict(train)
    preds = model.predict(test)
    accuracies += [accuracy_score(y_train, preds_train)]
    f1s += [f1_score(y_train, preds_train)]
    accuracies_test += [accuracy_score(y_test, preds)]
    f1s_test += [f1_score(y_test, preds)]
plt.clf()
plt.plot(fractions, fittimes, 'o')
plt.xlabel('data fraction')
plt.ylabel('fit time (s)')
plt.savefig('subset_fittime.png')
# plt.show()
plt.clf()
plt.plot(fractions, accuracies, 'o')
plt.plot(fractions, accuracies_test, 'o')
plt.legend(['train', 'test'])
plt.xlabel('data fraction')
plt.ylabel('accuracy')
plt.savefig('subset_accuracy.png')
# plt.show()
plt.clf()
plt.plot(fractions, f1s, 'o')
plt.plot(fractions, f1s_test, 'o')
plt.legend(['train', 'test'])
plt.xlabel('data fraction')
plt.ylabel('f1 score')
plt.savefig('subset_f1.png')
# plt.show()

# %%
