import numpy as np
import datetime
import pandas as pd
# read in raw data
AVG_TIME20 = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/training_20min_avg_travel_time-Copy1.csv")
AVG_TIME20['START_TIME'] = AVG_TIME20['time_window'].apply(lambda x :(x.split(','))[0][1:])
AVG_TIME20.drop(['time_window'],axis=1,inplace=True)

AVG_TIME20['START_TIME'] = pd.DataFrame(AVG_TIME20['START_TIME'])

MIN20_N = 24*3*91
end_date = datetime.datetime.strptime('2016-10-17 23:40:00','%Y-%m-%d %H:%M:%S')
date_list = [str((end_date - datetime.timedelta(minutes=20*x))) for x in range(MIN20_N)]
date_list.reverse()
date_list_df = pd.DataFrame(date_list)
date_list_df = date_list_df.rename(columns={0:'START_TIME'})
AVG_TIME20_FULL = pd.DataFrame(columns={'START_TIME','intersection_id','tollgate_id','avg_travel_time'})
for key,values in AVG_TIME20.groupby(['intersection_id','tollgate_id']):
    values = pd.merge(date_list_df,values,on='START_TIME',how='left')
    values['intersection_id'].fillna(method='backfill',inplace=True)
    values['tollgate_id'].fillna(method='backfill',inplace=True)
    AVG_TIME20_FULL = pd.concat([AVG_TIME20_FULL,values],axis=0)


AVG_TIME20_TAB = pd.pivot_table(AVG_TIME20_FULL, values=['avg_travel_time'], index=['intersection_id','tollgate_id'],columns=['START_TIME'], aggfunc=np.sum)

# train data time
TRN_N = 6
TST_N = 6
MIN20_N = 24*3*91

# zip 用法a=[1,2,3] b=[7,7,7,7,7,7] list(zip(a,b)) [(1, 7), (2, 7), (3, 7)]
TRAIN = pd.DataFrame()
train_date_zip = list(zip(date_list[0:MIN20_N-(TRN_N+TST_N)+1],date_list[TRN_N-1:MIN20_N-TST_N+1],date_list[TRN_N:MIN20_N-TST_N+2],date_list[TRN_N+TST_N-1:MIN20_N]))
train_date_zip_df = pd.DataFrame(train_date_zip)
train_date_zip_df.columns = ['TRN_STA','TRN_END','TST_STA','TST_END']
for TRN_STA,TRN_END,TST_STA,TST_END in train_date_zip:
    TRAIN_temp = AVG_TIME20_TAB.loc[:,TRN_STA:TST_END]
    TRAIN_temp.columns = np.arange(TRAIN_temp.shape[1])
    TRAIN_temp.reset_index(level=0, inplace=True)
    TRAIN_temp.loc[:,'TRN_STA']  = str(TRN_STA)
    TRAIN_temp.loc[:,'TRN_END'] = str(TRN_END)
    TRAIN_temp.loc[:,'TST_STA']  = str(TST_STA)
    TRAIN_temp.loc[:,'TST_END'] = str(TST_END)
    TRAIN = pd.concat( [TRAIN,TRAIN_temp],)
TRAIN = TRAIN.reset_index()
TRAIN_TRN_C =  list(map(lambda x:'SA'+ str(x).zfill(2), np.arange(TRN_N)))
TRAIN_TST_C =  list(map(lambda x:'SB'+ str(x).zfill(2), np.arange(TST_N)))
TRAIN.columns = ['TOLLGATE_ID','INTERSECTION_ID']  + TRAIN_TRN_C + TRAIN_TST_C  + ['TRN_STA','TRN_END','TST_STA','TST_END']
## #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #############################################
TRAIN_OK = TRAIN[TRAIN.loc[:,TRAIN_TST_C].isnull().sum(axis = 1)==0]
# IF TRAIN CONTAIN NAN --> DEL
TRAIN_OK = TRAIN_OK[TRAIN_OK.loc[:,TRAIN_TRN_C].isnull().sum(axis = 1)<=(TRN_N-6)]
TRAIN_OK.to_csv("TRAIN_OK.csv",index=None,encoding='utf-8')
# ###  TEST DATA GENERATE\
TEST = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/test1_20min_avg_travel_time.csv")
TEST['START_TIME'] = TEST['time_window'].apply(lambda x :(x.split(','))[0][1:])
TEST.drop(['time_window'],axis=1,inplace=True)
DAY_N = 7
MIN20_N = 24*3*DAY_N
end_date = datetime.datetime.strptime('2016-10-24 23:40:00','%Y-%m-%d %H:%M:%S')
date_list = [str((end_date- datetime.timedelta(minutes=20*x))) for x in range(MIN20_N)]
date_list.reverse()
date_list_datetime = pd.to_datetime(date_list)
morning_start = datetime.datetime.strptime('2016-10-18 06:00:00','%Y-%m-%d %H:%M:%S')
morning_end = datetime.datetime.strptime('2016-10-18 07:40:00','%Y-%m-%d %H:%M:%S')
afternoon_start = datetime.datetime.strptime('2016-10-18 15:00:00','%Y-%m-%d %H:%M:%S')
afternoon_end = datetime.datetime.strptime('2016-10-18 16:40:00','%Y-%m-%d %H:%M:%S')

def GetDate(date_list,date_list_datetime,morning_start):
    temp_index = date_list_datetime.slice_locs(morning_start)
    return date_list[temp_index[0]]
def GetDateList(date_list,date_list_datetime,morning_start,morning_end):
    temp_index = date_list_datetime.slice_locs(morning_start,morning_end)
    return date_list[temp_index[0]:temp_index[1]]

test_date_zip_df = pd.DataFrame(columns={'START_TIME'})
for x in range(DAY_N):
    test_date_list_A_temp =pd.DataFrame(GetDateList(date_list,date_list_datetime,afternoon_start + datetime.timedelta(days=x),afternoon_end + datetime.timedelta(days=x)),columns={'START_TIME'})
    test_date_list_M_temp =pd.DataFrame(GetDateList(date_list,date_list_datetime,morning_start + datetime.timedelta(days=x),morning_end + datetime.timedelta(days=x)),columns={'START_TIME'})
    test_date_zip_df = pd.concat([test_date_zip_df,test_date_list_M_temp],axis=0)
    test_date_zip_df = pd.concat([test_date_zip_df,test_date_list_A_temp],axis=0)

test_date_list = []
for x in range(DAY_N):
    test_date_list_M_start_temp = GetDate(date_list, date_list_datetime, morning_start + datetime.timedelta(days=x))
    test_date_list_M_end_temp = GetDate(date_list, date_list_datetime, morning_end + datetime.timedelta(days=x))

    test_date_list_A_start_temp = GetDate(date_list, date_list_datetime, afternoon_start + datetime.timedelta(days=x))
    test_date_list_A_end_temp = GetDate(date_list, date_list_datetime, afternoon_end + datetime.timedelta(days=x))
    test_date_list.append([test_date_list_M_start_temp, test_date_list_M_end_temp, test_date_list_A_start_temp,
                           test_date_list_A_end_temp])


test_date_zip_df = test_date_zip_df.reset_index(level=0,drop=True)
TEST_FULL = pd.DataFrame(columns={'START_TIME','intersection_id','tollgate_id','avg_travel_time'})
for key,values in TEST.groupby(['intersection_id','tollgate_id']):
    values = pd.merge(test_date_zip_df,values,on='START_TIME',how='left')
    values['intersection_id'].fillna(method='backfill',inplace=True)
    values['tollgate_id'].fillna(method='backfill',inplace=True)
    TEST_FULL = pd.concat([TEST_FULL,values],axis=0)
TEST_FULL = TEST_FULL.reset_index(level=0,drop=True)
TEST_FULL_TAB = pd.pivot_table(TEST_FULL, values=['avg_travel_time'], index=['intersection_id','tollgate_id'],columns=['START_TIME'], aggfunc=np.sum)
TEST_FULL_TAB.columns = test_date_zip_df.START_TIME
TEST_FULL_TAB_FILL = TEST_FULL_TAB.fillna(method='ffill',axis=1)

TEST_DF = pd.DataFrame()
for TRN_STA_M, TRN_END_M, TRN_STA_A, TRN_END_A in test_date_list:
    TRAIN_M_temp = TEST_FULL_TAB_FILL.loc[:, TRN_STA_M:TRN_END_M]
    TRAIN_A_temp = TEST_FULL_TAB_FILL.loc[:, TRN_STA_A:TRN_END_A]
    TRAIN_M_temp.columns = np.arange(TRAIN_M_temp.shape[1])
    TRAIN_A_temp.columns = np.arange(TRAIN_A_temp.shape[1])

    TRAIN_M_temp.loc[:, 'TRN_STA'] = str(TRN_STA_M)
    TRAIN_M_temp.loc[:, 'TRN_END'] = str(TRN_END_M)

    TRAIN_A_temp.loc[:, 'TRN_STA'] = str(TRN_STA_A)
    TRAIN_A_temp.loc[:, 'TRN_END'] = str(TRN_END_A)

    TRAIN_M_temp.reset_index(level=0, inplace=True)
    TRAIN_A_temp.reset_index(level=0, inplace=True)

    TEST_DF = pd.concat([TEST_DF, TRAIN_M_temp], )
    TEST_DF = pd.concat([TEST_DF, TRAIN_A_temp], )
TEST_DF.reset_index(level=0, inplace=True)
TRAIN_TRN_C =  list(map(lambda x:'SA'+ str(x).zfill(2), np.arange(6)))
TEST_DF.columns = ['TOLLGATE_ID','INTERSECTION_ID']  + TRAIN_TRN_C + ['TRN_STA','TRN_END']
TEST_DF.to_csv("TEST_DF.csv",index=None,encoding='utf-8')
###############MERGE ALL FEATURE ################################################################################################################
TEST_DF['TOLLGATE_ID'] = TEST_DF['TOLLGATE_ID'].apply(lambda x:int(x))
TRAIN_OK['TOLLGATE_ID'] = TRAIN_OK['TOLLGATE_ID'].apply(lambda x:int(x))
ROUTES_FEATURES = pd.read_csv('ROUTES_FEATURES.csv')
TRAIN_OK_ROUTES = pd.merge(TRAIN_OK,ROUTES_FEATURES,on=['INTERSECTION_ID','TOLLGATE_ID'],how='left')
TEST_DF_ROUTES = pd.merge(TEST_DF,ROUTES_FEATURES,on=['INTERSECTION_ID','TOLLGATE_ID'],how='left')
ALL_FEATURE_LIST = ['TOLLGATE_ID','INTERSECTION_ID','SA00','SA01','SA02','SA03','SA04','SA05','TRN_STA','TRN_END','SUM_LENGTH'
                    ,'SI00','SI01','SI02','SI03','SI04','SI05','SI06','SI07','SI08','FIX','RATIO','ROUTE_HOUR_head'
                    ,'ROUTE_PCT_head','ROUTE_MIN_TTIME','ROUTE_MAX_TTIME','ROUTE_LAST_TTIME','ROUTE_MEAN_TTIME']
TRAIN_TST_C = ['SB00', 'SB01', 'SB02', 'SB03', 'SB04', 'SB05']
X = TRAIN_OK_ROUTES[ALL_FEATURE_LIST]
Y = TRAIN_OK_ROUTES[TRAIN_TST_C]
X_test = TEST_DF_ROUTES[ALL_FEATURE_LIST]
X.to_csv('X.csv', index = False)
Y.to_csv('Y.csv', index = False)
X_test.to_csv('Xtest.csv', index = False)
###############################ADD WEATHER FEATURE########################################################################################
X = pd.read_csv('X.csv')
Y = pd.read_csv('Y.csv')
Xtest = pd.read_csv('Xtest.csv')
TRAIN_WEATHER_UPDATE_FULL = pd.read_csv('TRAIN_WEATHER_UPDATE_FULL.csv')
TEST_WEATHER_FULL = pd.read_csv('TEST_WEATHER_FULL.csv')

TRAIN_WEATHER_UPDATE_FULL['concrete_date']=pd.to_datetime(TRAIN_WEATHER_UPDATE_FULL['concrete_date'])
TRAIN_WEATHER_UPDATE_FULL_TAB = pd.pivot_table(TRAIN_WEATHER_UPDATE_FULL, values=['pressure','sea_pressure','wind_direction','precipitation','SSD'],index=['concrete_date'])
TEST_WEATHER_FULL['concrete_date']=pd.to_datetime(TEST_WEATHER_FULL['concrete_date'])
TEST_WEATHER_FULL_TAB = pd.pivot_table(TEST_WEATHER_FULL, values=['pressure','sea_pressure','wind_direction','precipitation','SSD'],index=['concrete_date'])

X['TST_STA']=X['TRN_END'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')+ datetime.timedelta(minutes=20))
X['XTST_END']=X['TRN_END'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')+ datetime.timedelta(hours=2))
Xtest['TST_STA']=Xtest['TRN_END'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')+ datetime.timedelta(minutes=20))
Xtest['XTST_END']=Xtest['TRN_END'].apply(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')+ datetime.timedelta(hours=2))

TRN_WEATHER = pd.DataFrame()
TST_WEATHER = pd.DataFrame()
for x in range(X.shape[0]):
    each_row = X.iloc[x]
    TRN_weather = TRAIN_WEATHER_UPDATE_FULL_TAB.loc[each_row['TRN_STA']:each_row['TRN_END']]
    TRN_WEATHER = pd.concat([TRN_WEATHER,pd.DataFrame(TRN_weather.mean()).T],axis=0)
    TST_weather = TRAIN_WEATHER_UPDATE_FULL_TAB.loc[each_row['TST_STA']:each_row['XTST_END']]
    TST_WEATHER = pd.concat([TST_WEATHER,pd.DataFrame(TST_weather.mean()).T],axis=0)
TRN_WEATHER.columns = [(lambda x:('TRNW'+ str(x).zfill(2))) (x)  for x in range(TRN_WEATHER.shape[1])]
TST_WEATHER.columns = [(lambda x:('TSTW'+ str(x).zfill(2))) (x)  for x in range(TST_WEATHER.shape[1])]
W= pd.concat([TST_WEATHER,TRN_WEATHER],axis=1)
W.reset_index(drop=True,inplace=True)
X_F = pd.concat([X,W],axis=1)


TRN_WEATHER = pd.DataFrame()
TST_WEATHER = pd.DataFrame()
for x in range(Xtest.shape[0]):
    each_row = Xtest.iloc[x]
    TRN_weather = TEST_WEATHER_FULL_TAB.loc[each_row['TRN_STA']:each_row['TRN_END']]
    TRN_WEATHER = pd.concat([TRN_WEATHER,pd.DataFrame(TRN_weather.mean()).T],axis=0)
    TST_weather = TEST_WEATHER_FULL_TAB.loc[each_row['TST_STA']:each_row['XTST_END']]
    TST_WEATHER = pd.concat([TST_WEATHER,pd.DataFrame(TST_weather.mean()).T],axis=0)
TRN_WEATHER.columns = [(lambda x:('TRNW'+ str(x).zfill(2))) (x)  for x in range(TRN_WEATHER.shape[1])]
TST_WEATHER.columns = [(lambda x:('TSTW'+ str(x).zfill(2))) (x)  for x in range(TST_WEATHER.shape[1])]
W= pd.concat([TST_WEATHER,TRN_WEATHER],axis=1)
W.reset_index(drop=True,inplace=True)
Xtest_F = pd.concat([Xtest,W],axis=1)

X_F.to_csv('X.csv', index = False)
Xtest_F.to_csv('Y.csv', index = False)