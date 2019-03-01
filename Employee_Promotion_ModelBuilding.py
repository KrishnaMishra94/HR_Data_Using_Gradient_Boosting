import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib


pd.set_option('display.max_rows',500) ## WIDENING OUTPUT DISPLAY

############ LOADING DATA #############
train = pd.read_csv('train_LZdllcl.csv')
test  = pd.read_csv('test_2umaH9m.csv')
test_orig  = pd.read_csv('test_2umaH9m.csv')
train_orig = train.copy()
#######################################
train.shape # 54808, 14
test.shape # 23490, 13
train.columns

train.iloc[:5,:8]

train.iloc[:5,8:15]

sns.countplot(train['recruitment_channel'])
len(train['region'].unique())
sns.countplot(train['region'])


############# CREATING COUNT PLOT FOR EACH COLUMN ###############
int_cols = ['no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met >80%','awards_won?','avg_training_score','is_promoted']

train.rename(columns={'KPIs_met >80%':'is_KPI_met_80','awards_won?':'is_award_won'},inplace=True)
cat_cols = ['education','department','no_of_trainings','previous_year_rating','is_KPI_met_80','is_award_won','is_promoted']


plt.figure(figsize=(12,4))
i=0
for col in cat_cols:
    plt.figure(i)
    sns.countplot(x=col,data=train)
    name = col+'_Count_Plot.png'
    plt.savefig('Plots//{}'.format(name))
    i = i+1

## THERE IS VERY HIGH BIAS IN OUTPUT DATA => 10% and 90% -------> Going Under Sampling of Majority Data
    
###############################################################


############### UNDER-SAMPLING OF DATA #####################


promoted_indices = np.array(train[train['is_promoted']==1].index)
not_promoted_indices = np.array(train[train['is_promoted']==0].index)

train_under_sampled = under_sample_data(not_promoted_indices,promoted_indices,2,train)

def under_sample_data(majority_indices,minority_indices,multiple,data):
    # PARAMTER majority_indices MEANS INDICES FOR MAJORITY OUTPUT DATA 
    # PARAMTER minority_indices MEANS INDICES FOR MINORITU OUTPUT DATA
    # PARAMETER multiple MEANS under_sampled_majority = WHAT *multiple* OF minority
    under_sampled_indices = np.random.choice(majority_indices,multiple*len(minority_indices))
    total_indices = np.concatenate([under_sampled_indices,minority_indices])
    under_sampled_data =  data.iloc[total_indices,:]
    
    # THE RESULTANT DATAFRAME HAS BOTH MINORITY AND UNDERSAMPLED MAJORITY DATA
    return(under_sampled_data)
#############################################################


#################### COLUMN EXCLUSION & DATA SPLIT ########################

## TO CHECK IF COLUMNS HAVE ANY BIAS OVER GETTING PROMOTED
text_cols = ['department','region','education','gender','recruitment_channel']


for cols in text_cols:
    print('\n')
    print('For Feature --------------> {}'.format(cols))
    col_values = train_under_sampled[cols].unique().tolist()
    for value in col_values:
        percent = (train_under_sampled[(train_under_sampled[cols]==value) & train_under_sampled['is_promoted']==1].shape[0]/train_under_sampled[(train_under_sampled[cols]==value)].shape[0])*100
        percent = round(percent,2)
        print('Value--->{} & Promoted Percentage-->{}'.format(value,percent))


## HENCE THERE IS NO BIAS OVER COLUMNS 'gender'. SO EXCLUDING IT FROM FEATURES

X = train_under_sampled.drop(['gender','region','is_promoted'],axis=1)

y = train_under_sampled['is_promoted']

##########################################################################

################ PRE-PROCESSING STEP 1 : RENAMING #################

X = column_rename(X)
def column_rename(data):
    data.rename(columns={'KPIs_met >80%':'is_KPI_met_80','awards_won?':'is_award_won'},inplace=True)
    return data



###################################################################    

################# PRE-PROCESSING STEP 2 :TREATING MISSING VALUES ###################   
cf.draw_null_values_table(X)

## column 'previous_year_rating' has 8% missing values
## column 'education' has 5% missing values
X['previous_year_rating'].hist()

X[X['previous_year_rating'].isnull()]['age'].hist() 
X[X['previous_year_rating'].isnull()]['length_of_service'].hist()
## SEEMS LIKE 'previous_year_rating' IS MISSING FOR EMPLOYEES THAT ARE EITHER  FRESHER OR ARE NEW TO THE COMPANY


X[X['education'].isnull()].iloc[:5,:11]
X[X['education'].isnull()].iloc[:5,11:14]
X[X['education'].isnull()]['is_KPI_met_80'].hist() 
X[X['education'].isnull()]['age'].hist() 

X['education'].value_counts()

X[X['education']=="Bachelor's"]['age'].hist()
X[X['education']=="Master's & above"]['age'].hist()
X[X['education']=="Below Secondary"]['age'].hist()
## 'education' IS MISSING MOSTLY FOR AGE VALUES LESS THAN 35.
## FOR AGE VALUE < 25 'education' CAN BE FILLED WITH "Below Secondary"
## FOR AGE VALUE > 25 AND < 35 'education' CAN BE FILLED WITH "Bachelor's"



##### FILLING MISSING VALUES WITH MEAN #######

X = treat_misssing_values(X)

def treat_misssing_values(data):
    data['previous_year_rating'].fillna(value=data['previous_year_rating'].mean(),inplace=True)
    data.loc[(data['age'] <=25) & (data['education'].isnull()),'education']='Below Secondary'
    data.loc[(data['age'] > 25) & (data['education'].isnull()),'education']="Bachelor's"
    return data

    

###################################################################


################ PRE-PROCESSING STEP 3 : ENCODING OF TEXT DATA ##################

X = text_encoding(X)

def text_encoding(data):
    nominal_cols = ['department','recruitment_channel']
    ## BINARY ENCODING
    data = pd.get_dummies(data,prefix=nominal_cols,columns=nominal_cols,drop_first=True)
    ## MANUAL LABEL ENCODING    
    education_label = {"Below Secondary":0,"Bachelor's":1, "Master's & above":2}
    data['education'] = data['education'].replace(education_label)
    return data
    
###############################################################################


##################  TRAIN TEST SPLIT #####################

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
##########################################################

############################## GRADIENT BOOSTING CLASSIFIER ################################

y_test.value_counts()

##### GRID SEARCH #######

n_estimators = [100, 200, 300, 400, 500]
learning_rate = [0.1,0.3,0.5,0.7]

for tree_count in n_estimators:
    for learn_rate in learning_rate:
        for feature_count in np.arange(9,11):
            print('For Tree Count -->{} & Learning Rate -->{} & Max_Features -->{}'.format(tree_count,learn_rate,feature_count))                
            gradient_boost_model = GradientBoostingClassifier(n_estimators=tree_count,learning_rate=learn_rate,max_features=feature_count,subsample=0.8)
            gradient_boost_model.fit(X_train,y_train)
            predict = gradient_boost_model.predict(X_test)
            confusion_mat = confusion_matrix(y_test,predict)
            print("the recall for this model is :",confusion_mat[1,1]/(confusion_mat[1,1]+confusion_mat[1,0]))


## 70% Recall with n_estimators=500,learning_rate=0.7,max_features=9
#########################  
            
##### MODEL & COLUMN DUMPS #####

gradient_boost_model = GradientBoostingClassifier(n_estimators=500,learning_rate=0.7,max_features=9,subsample=0.8)
gradient_boost_model.fit(X_train,y_train)
joblib.dump(gradient_boost_model,'Serialized_Data//gradient_boosting_model.pkl')
joblib.dump(X_train.columns,'Serialized_Data//data_columns.pkl')

###############################

############################################################################################


############# DRAWING FEATURE IMPORTANCE ##############

draw_rf_feature_imp_plot(X_train,gradient_boost_model)

def draw_rf_feature_imp_plot(df,model):
    features = df.columns
    feature_imp = model.feature_importances_
    indices = np.argsort(feature_imp)
    plt.figure(figsize=(15,10))
    plt.barh(range(len(indices)),feature_imp[indices],color='b',align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Relative Feature Importance')
    plt.savefig('Plots//Feature_Importance.png')
    plt.show()
#######################################################







    



