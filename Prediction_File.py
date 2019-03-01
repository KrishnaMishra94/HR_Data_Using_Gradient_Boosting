import pandas as pd
import os
from sklearn.externals import joblib

gradient_boost_model = joblib.load('Serialized_Data//gradient_boosting_model.pkl')
data_columns = joblib.load('Serialized_Data//data_columns.pkl')
cols = list(data_columns)
test = pd.read_csv('test_2umaH9m.csv')


def column_rename(data):
    data.rename(columns={'KPIs_met >80%':'is_KPI_met_80','awards_won?':'is_award_won'},inplace=True)
    return data

def treat_misssing_values(data):
    data['previous_year_rating'].fillna(value=data['previous_year_rating'].mean(),inplace=True)
    data.loc[(data['age'] <=25) & (data['education'].isnull()),'education']='Below Secondary'
    data.loc[(data['age'] > 25) & (data['education'].isnull()),'education']="Bachelor's"
    return data

def text_encoding(data):
    nominal_cols = ['department','recruitment_channel']
    ## BINARY ENCODING
    data = pd.get_dummies(data,prefix=nominal_cols,columns=nominal_cols,drop_first=True)
    ## MANUAL LABEL ENCODING    
    education_label = {"Below Secondary":0,"Bachelor's":1, "Master's & above":2}
    data['education'] = data['education'].replace(education_label)
    return data


######## ALL THE PRE-PROCESSING OF TEST DATA ##############
test = test.drop(['gender','region'],axis=1)
test = column_rename(test)
test = treat_misssing_values(test)
test = text_encoding(test)
test = test.reindex(columns=cols,fill_value=0)
##########################################################


predict = gradient_boost_model.predict(test)
test_predict = pd.concat([test['employee_id'],pd.Series(predict,name='is_promoted')],axis=1)
test_predict.to_csv('Test_Data_With_Prediction.csv',index=False)




