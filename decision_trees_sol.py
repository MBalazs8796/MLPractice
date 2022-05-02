# %%
import time
import numpy
from sklearn import tree
from df_preprocessor import load_and_process
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
# %%
keepList = ['city', 'city_development_index', 'relevent_experience', 'enrolled_university', 'education_level',
            'major_discipline', 'experience', 'company_size', 'company_type', 'last_new_job', 'training_hours', 'target']
df, test_df = load_and_process(
    "csv/train_input.csv", "csv/test_input.csv", keepList)
# %%
y_train = df['target']
#model = tree.DecisionTreeClassifier(max_depth=4)
model = tree.DecisionTreeClassifier(
    max_depth=3, class_weight='balanced', criterion='entropy', min_samples_leaf=10)
model = model.fit(df.drop('target', axis=1).values, y_train.values)
y_test = test_df['target']
y_pred = model.predict(test_df.drop('target', axis=1).values)
# %%
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('accuracy: ' + str(accuracy))
print('f1: ' + str(f1))
# %%
# accuracy: 0.835, f1: 0.587
params = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 5, 10],
    'max_features': [None, 'auto', 'log2'],
    'class_weight': [None, 'balanced']
}
clf = GridSearchCV(tree.DecisionTreeClassifier(), params,
                   scoring='f1', cv=10, refit=True, return_train_score=True)
clf.fit(df.drop('target', axis=1).values, y_train)
print('Best-params:', clf.best_params_)
print('Best-score:', clf.best_score_)
print("Accuracy: " + str(accuracy_score(y_test,
      clf.predict(test_df.drop('target', axis=1).values))))
print("F1 score: " + str(f1_score(y_test,
      clf.predict(test_df.drop('target', axis=1).values))))

# %%
model = tree.DecisionTreeClassifier(
    max_depth=3, class_weight='balanced', criterion='entropy', min_samples_leaf=10)
path = model.cost_complexity_pruning_path(
    df.drop('target', axis=1).values, y_train.values)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(
        max_depth=3, class_weight='balanced', criterion='entropy', min_samples_leaf=10, ccp_alpha=ccp_alpha)
    clf.fit(df.drop('target', axis=1).values, y_train.values)
    clfs.append(clf)


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

train_scores = [
    clf.score(df.drop('target', axis=1).values, y_train.values) for clf in clfs]
test_scores = [clf.score(test_df.drop(
    'target', axis=1).values, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o",
        label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o",
        label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# %%
# accuracy: 0.81, f1: 0.574
model = ExtraTreesClassifier(
    criterion='entropy', max_depth=3, min_samples_leaf=10, class_weight='balanced')
model = model.fit(df.drop('target', axis=1).values, y_train.values)
y_test = test_df['target']
y_pred = model.predict(test_df.drop('target', axis=1).values)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('accuracy: ' + str(accuracy))
print('f1: ' + str(f1))

# %%
# accuracy: 0.833, f1: 0.584
params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 5, 10],
    'n_jobs': [4],
    'max_features': [None, 'auto', 'log2'],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

clf = GridSearchCV(ExtraTreesClassifier(), params, scoring='f1',
                   cv=5, refit=True, return_train_score=True)
clf.fit(df.drop('target', axis=1).values, y_train)
print('Best-params:', clf.best_params_)
print('Best-score:', clf.best_score_)
print("Accuracy: " + str(accuracy_score(y_test,
      clf.predict(test_df.drop('target', axis=1).values))))
print("F1 score: " + str(f1_score(y_test,
      clf.predict(test_df.drop('target', axis=1).values))))

# %%
n = df.drop('target', axis=1).values.shape[1]
pca = PCA(n_components=n)
pca.fit(df.drop('target', axis=1).values)

percentages = numpy.cumsum(pca.explained_variance_ratio_[:10])
plt.plot(percentages, 'o')
plt.ylim([0, 1.01])
plt.show()

print(pca.explained_variance_ratio_[:3])

# %%
# accuracy: 0.73, f1: 0.13
pca = PCA(n_components=1)
model = tree.DecisionTreeClassifier(
    max_depth=3, class_weight='balanced', criterion='entropy', min_samples_leaf=10)
pipe = Pipeline(steps=[('pca', pca), ('dt', model)])

pipe.fit(df.drop('target', axis=1).values, y_train.values)
y_pred = pipe.predict(test_df.drop('target', axis=1).values)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print('accuracy: ' + str(accuracy))
print('f1: ' + str(f1))

# %%
params = {
    'dt__criterion': ['entropy'],
    'dt__max_depth': [3],
    'dt__min_samples_leaf': [10],
    'dt__class_weight': ['balanced'],
    'pca__n_components': [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50]
}

clf = GridSearchCV(pipe, params, scoring='f1', cv=5,
                   refit=True, return_train_score=True)
clf.fit(df.drop('target', axis=1).values, y_train)
print('Best-params:', clf.best_params_)
print('Best-score:', clf.best_score_)
print("Accuracy: " + str(accuracy_score(y_test,
      clf.predict(test_df.drop('target', axis=1).values))))
print("F1 score: " + str(f1_score(y_test,
      clf.predict(test_df.drop('target', axis=1).values))))
# %%
n_components = [1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 100, 139]
fittimes = []
accuracies = []
f1s = []
accuracies_test = []
f1s_test = []
for i in n_components:
    pipe.set_params(pca__n_components=i)
    starttime = time.time()
    pipe.fit(df.drop('target', axis=1).values, y_train)
    fittimes += [time.time()-starttime]
    accuracies += [accuracy_score(y_train,
                                  pipe.predict(df.drop('target', axis=1).values))]
    f1s += [f1_score(y_train, pipe.predict(df.drop('target', axis=1).values))]
    accuracies_test += [accuracy_score(y_test,
                                       pipe.predict(test_df.drop('target', axis=1).values))]
    f1s_test += [f1_score(y_test,
                          pipe.predict(test_df.drop('target', axis=1).values))]
plt.clf()
plt.plot(n_components, fittimes, 'o')
plt.xlabel('number of features')
plt.ylabel('fit time (s)')
plt.savefig('PCA_fittime.png')
plt.clf()
plt.plot(n_components, accuracies, 'o')
plt.plot(n_components, accuracies_test, 'o')
plt.legend(['train', 'test'])
plt.xlabel('number of features')
plt.ylabel('accuracy')
plt.savefig('PCA_accuracy.png')
plt.clf()
plt.plot(n_components, f1s, 'o')
plt.plot(n_components, f1s_test, 'o')
plt.legend(['train', 'test'])
plt.xlabel('number of features')
plt.ylabel('f1 score')
plt.savefig('PCA_f1.png')
# %%
