import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import datetime

def abs_MAPE_error(y_pred,y_true):
    return np.mean(np.mean(np.abs(y_pred-y_true)/np.abs(y_true)) )
def abs_MAPE_error_element(y_pred,y_true):
    return np.abs(y_pred-y_true)/np.abs(y_true)
def abs_error(y_pred,y_true):
    return np.mean(np.mean(np.abs(y_pred-y_true)))
def abs_error_element(y_pred,y_true):
    return np.abs(y_pred-y_true)


X = pd.read_csv('C:/Users/Administrator/PycharmProjects/Traffic/Feature/X.csv')
Xtest = pd.read_csv('C:/Users/Administrator/PycharmProjects/Traffic/Feature/Xtest.csv')
Y = pd.read_csv('C:/Users/Administrator/PycharmProjects/Traffic/Feature/Y.csv')
# 去不相干列--
cols_drop = ['TOLLGATE_ID','INTERSECTION_ID','TRN_STA','TRN_END','TST_STA','XTST_END']
X.drop(cols_drop,axis=1,inplace=True)
Xtest.drop(cols_drop,axis=1,inplace=True)
# 归一化---
for a in X.columns:
    scaler = StandardScaler()
    X[a] = scaler.fit_transform(X[a])
for a in Xtest.columns:
    scaler = StandardScaler()
    Xtest[a] = scaler.fit_transform(Xtest[a])
# 一阶训练 保留90%的数据，去除脏数据
error_list = []
Ytrain_all = pd.DataFrame()
XGBR = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=500)
TRAIN_TST_C = ['SB00', 'SB01', 'SB02', 'SB03', 'SB04', 'SB05']
for ind in range(6):
    time1 = time.time()
    XGBR.fit(X.values, Y.values[:, ind])
    y_true = Y.values[:, ind]
    y_pred = XGBR.predict(X.values)
    time2 = time.time()
    print(str(ind) + '_error:' + str(abs_MAPE_error(y_pred, y_true)) + '__time:' + str(time2 - time1))
    error_list.append(abs_MAPE_error(y_pred, y_true))

    Ytrain = XGBR.predict(X.values)
    Ytrain_df = pd.DataFrame(Ytrain)
    Ytrain_all = pd.concat((Ytrain_all, Ytrain_df), axis=1)
print(np.mean(error_list))

#%%
Ytrain_true = pd.DataFrame(Y)
Y_error = abs_MAPE_error_element(Ytrain_all.values,Ytrain_true.values  )
Good_ind =  Y_error.sum(axis=1).argsort()[0:np.int(0.90*len(X))]
#%%

# 二阶训练
XGBR = xgb.XGBRegressor(max_depth = 5,learning_rate=0.01,n_estimators=1600,reg_alpha=1,reg_lambda=0)
Ytest_all = pd.DataFrame()
error_list = []
Ytrain_all = pd.DataFrame()
XGBR = xgb.XGBRegressor(max_depth = 3,learning_rate=0.1,n_estimators=500)
TRAIN_TST_C = ['SB00', 'SB01', 'SB02', 'SB03', 'SB04', 'SB05']
for ind in range(6):
    time1 = time.time()
    XGBR.fit(X.values[Good_ind],Y.values[Good_ind,ind] )
    y_true = Y.values[Good_ind,ind]
    y_pred = XGBR.predict(X.values[Good_ind])
    time2 = time.time()
    print(str(ind)+ '_error:' + str(abs_MAPE_error(y_pred,y_true  ) ) + '__time:'+ str(time2 - time1) )
    error_list.append(abs_MAPE_error(y_pred, y_true  ))
    Ytest = XGBR.predict(Xtest.values)
    Ytest_df = pd.DataFrame(Ytest)
    Ytest_all = pd.concat((Ytest_all,Ytest_df),axis = 1)
print(np.mean(error_list))

# Gernerate Final Result
Xtest = pd.read_csv('C:/Users/Administrator/PycharmProjects/Traffic/Feature/Xtest.csv')
Ytest_all.columns = [0,1,2,3,4,5]
result = pd.concat([Ytest_all,Xtest],axis=1)
result = result.loc[:,[0,1,2,3,4,5,'TOLLGATE_ID','INTERSECTION_ID','TRN_STA','TRN_END']]
result['TRN_END'] = result['TRN_STA'].apply(lambda x :
str((datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S') + datetime.timedelta(hours=2))))

# Final submit format
sub_result = pd.DataFrame()
for x in range(result.shape[0]):
    t = result.loc[x,:]
    for y in range(6):
        end_time = str(datetime.datetime.strptime(t['TRN_END'],'%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=20*(y+1)))
        start_time = str((datetime.datetime.strptime(t['TRN_END'],'%Y-%m-%d %H:%M:%S') + datetime.timedelta(minutes=20*y)))
        timewindow = '['+start_time+','+end_time+')'
        each_result = pd.DataFrame({'intersection_id':[t['INTERSECTION_ID']],'tollgate_id':[t['TOLLGATE_ID']]
        ,'time_window':[timewindow] ,'avg_travel_time':[t[y]]})
        sub_result = pd.concat((sub_result,each_result),axis=0)
cols = ['intersection_id','tollgate_id','time_window','avg_travel_time']
sub_result = sub_result.ix[:, cols]
sub_result.to_csv('XGBOOST1.csv',index=False,)
