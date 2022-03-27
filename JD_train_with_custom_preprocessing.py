#%%
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
# %%
def preprocess(df:pd.DataFrame, fulldata:pd.DataFrame):
    processed=pd.DataFrame()
    processed["is_city_103"]=df["city"]=="city_103"
    processed["is_city_21"]=df["city"]=="city_21"
    processed["is_city_16_or_114"]=(df["city"]=="city_16") | (df["city"]=="city_114")
    citydev=np.array(df["city_development_index"])
    citydev-=fulldata["city_development_index"].min()
    citydev/=fulldata["city_development_index"].max()-fulldata["city_development_index"].min()
    processed["city_development_index"]=citydev
    processed["is_male"]=df["gender"]=="Male"
    processed["is_female"]=df["gender"]=="Female"
    processed["is_other"]=df["gender"]=="Other"
    processed["has_relevant_experience"]=df["relevent_experience"]=="Has relevent experience"
    processed["is_full_time"]=df["enrolled_university"]=="Full time course"
    processed["education_level"]=df["education_level"].replace({
        "Phd":5/5,
        "Masters":4/5,
        "Graduate":3/5,
        "High School":2/5,
        "Primary School":1/5,
    }).fillna(0)
    processed["is_arts"]=df["major_discipline"]=="Arts"
    processed["is_business"]=df["major_discipline"]=="Business Degree"
    processed["is_humanities"]=df["major_discipline"]=="Humanities"
    processed["is_no_major"]=df["major_discipline"]=="No Major"
    processed["is_other_discipline"]=df["major_discipline"]=="Other"
    processed["is_STEM"]=df["major_discipline"]=="STEM"
    processed["is_unknown_major"]=df["major_discipline"].isna()
    temp_experience=df["experience"].replace({
        "<1":0/21,
        "1":1/21,
        "2":2/21,
        "3":3/21,
        "4":4/21,
        "5":5/21,
        "6":6/21,
        "7":7/21,
        "8":8/21,
        "9":9/21,
        "10":10/21,
        "11":11/21,
        "12":12/21,
        "13":13/21,
        "14":14/21,
        "15":15/21,
        "16":16/21,
        "17":17/21,
        "18":18/21,
        "19":19/21,
        "20":20/21,
        ">20":21/21,
    })
    processed["experience"]=temp_experience.fillna(
        np.average(temp_experience[~temp_experience.isna()]))
    processed["is_funded_startup"]=df["company_type"]=="Funded Startup"
    processed["is_PVT"]=df["company_type"]=="Pvt Ltd"
    processed["is_company_large"]=df["company_size"].replace({
        "<10":0/7,
        "10-49":1/7,
        "10/49":1/7,
        "50-99":2/7,
        "100-500":3/7,
        "500-999":4/7,
        "1000-4999":5/7,
        "5000-9999":6/7,
        "10000+":7/7
    }).fillna(0)
    processed["is_company_small"]=df["company_size"].replace({
        "<10":7/7,
        "10-49":6/7,
        "10/49":6/7,
        "50-99":5/7,
        "100-500":4/7,
        "500-999":3/7,
        "1000-4999":2/7,
        "5000-9999":1/7,
        "10000+":0/7
    }).fillna(0)
    processed["job_stability"]=df["last_new_job"].replace({
        "never":1/6,
        "1":2/6,
        "2":3/6,
        "3":4/6,
        "4":5/6,
        ">4":6/6,
    }).fillna(0)
    processed.replace({False: 0, True: 1}, inplace=True)
    return processed.astype(float)
#%%
#Load data
df = pd.read_csv("csv/train_input.csv")
train=preprocess(df.tail(-2000),df)
train_target=df["target"].tail(-2000)
develop=preprocess(df.tail(2000),df)
develop_target=df["target"].tail(2000)
# %%
def gaussian_fit(train_features, train_target):
    not_target_model=GaussianMixture(n_components=20,covariance_type="full",n_init=10)
    not_target_model.fit(train_features[train_target==0])
    target_model=GaussianMixture(n_components=20,covariance_type="full",n_init=10)
    target_model.fit(train_features[train_target==1])
    def predict(features):
        is_target_pred=target_model.score_samples(features)
        is_not_target_pred=not_target_model.score_samples(features)
        final_pred=is_target_pred>is_not_target_pred
        return final_pred.astype(int)
    return predict
# %%
from sklearn.metrics import f1_score, accuracy_score
def print_statistics(truth,predict):
    #results=pd.DataFrame({"pred":predict==1,"truth":truth==1}).value_counts()
    print("Accuracy = ", accuracy_score(truth,predict))
    print("f1 = ",f1_score(truth,predict))
    #print(results)
# %%
def try_many_methods(train_features,train_target,features,target):
    print("--- Dummy most frequent ---")
    dummy_m_f=DummyClassifier(strategy="most_frequent")
    dummy_m_f.fit(train_features,train_target)
    print_statistics(target,dummy_m_f.predict(features))
    print("--- Dummy random ---")
    dummy_str=DummyClassifier(strategy="stratified")
    dummy_str.fit(train_features,train_target)
    print_statistics(target,dummy_str.predict(features))
    print("--- Gaussian Naive Bayes ---")
    naive=GaussianNB()
    naive.fit(train_features,train_target)
    print_statistics(target,naive.predict(features))
    print("--- Bernoulli Naive Bayes ---")
    naive2=BernoulliNB()
    naive2.fit(train_features,train_target)
    print_statistics(target,naive2.predict(features))
    print("--- K Nearest ---")
    neighbor=KNeighborsClassifier(n_neighbors=10)
    neighbor.fit(train_features,train_target)
    print_statistics(target,neighbor.predict(features))
    print("--- GMM ---")
    print_statistics(target,gaussian_fit(train_features,train_target)(features))
# %%
print("### On train ###")
try_many_methods(train,train_target,train,train_target)
print("### On development ###")
try_many_methods(train,train_target,develop,develop_target)
# %%
