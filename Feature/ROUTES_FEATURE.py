import pandas as pd

# read in rotues data
ROUTES = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/routes (table 4).csv")
LINKS = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/links (table 3).csv")

ROUTES['link_seq'] = ROUTES['link_seq'].apply(lambda x :(x.split(',')))
LINK_SEQ = pd.DataFrame(ROUTES['link_seq'])
LINKS.index = LINKS.link_id
# colculate each rotues length

SUM_LENGTH = []
for link_seq in ROUTES['link_seq']:
    temp_sum = 0
    for link in link_seq:
        temp_sum = temp_sum + LINKS.loc[int(link),:]['length']
    SUM_LENGTH.append(temp_sum)
SUM_LENGTH = pd.DataFrame(SUM_LENGTH)
LINK_SEQ = pd.concat([LINK_SEQ,SUM_LENGTH],axis=1)
LINK_SEQ.rename(columns={0:"SUM_LENGTH"},inplace=True)
ROUTES = pd.concat([ROUTES,LINK_SEQ],axis=1)
ROUTES.drop(['link_seq'],axis=1,inplace=True)

ROUTES.columns = ['INTERSECTION_ID','TOLLGATE_ID','SUM_LENGTH']
# read in holi data
import datetime
HOLI = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/HOLI.csv")
HOLI['DATE'] =HOLI['DATE'].apply(lambda x:datetime.datetime.strptime(str(x),'%Y%m%d').date() )
# read in raw data
AVG_TIME20 = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/training_20min_avg_travel_time-Copy1.csv")
AVG_TIME20['START_TIME'] = AVG_TIME20['time_window'].apply(lambda x :(x.split(','))[0][1:])

AVG_TIME20['TIME'] = pd.DatetimeIndex(AVG_TIME20.START_TIME).time
AVG_TIME20['DATE'] = pd.DatetimeIndex(AVG_TIME20.START_TIME).date
AVG_TIME20['HOUR'] = pd.DatetimeIndex(AVG_TIME20.START_TIME).hour
AVG_TIME20['DAY'] = pd.DatetimeIndex(AVG_TIME20.DATE).dayofweek

AVG_TIME20.drop(['time_window','START_TIME'],axis=1,inplace=True)
AVG_TIME20 = pd.merge(ROUTES,AVG_TIME20,on=['intersection_id','tollgate_id'],how='right')
AVG_TIME20 = pd.merge(AVG_TIME20,HOLI,on=['DATE'],how='left')
#%%  calculate top hours
TOP_N = 1
INTERSECTION_ID = []
TOLLGATE_ID = []
ROUTE_HOUR_head = []
ROUTE_PCT_head = []
ROUTE_MIN_TTIME = []
ROUTE_MAX_TTIME = []
ROUTE_MEAN_TTIME = []
ROUTE_LAST_TTIME = []

for key, values in AVG_TIME20.groupby(['intersection_id', 'tollgate_id']):
    values.reset_index(inplace=True)
    # group top hour
    tt = values.groupby(['HOUR'], as_index=False).sum()
    tt2 = tt.sort_values('avg_travel_time', ascending=False, inplace=False)
    tt3 = tt2.head(TOP_N)['HOUR'].values
    tt4 = tt2.head(TOP_N)['avg_travel_time'].values / tt2['avg_travel_time'].sum()
    # group date
    tt5 = values.groupby(['DATE'], as_index=False).min().mean()
    tt6 = values.groupby(['DATE'], as_index=False).max().mean()
    values['MEAN'] = values['avg_travel_time'] * values['HOUR']

    INTERSECTION_ID.append(values['intersection_id'][0])
    TOLLGATE_ID.append(values['tollgate_id'][0])
    ROUTE_HOUR_head.append(tt3)
    ROUTE_PCT_head.append(tt4)

    ROUTE_MIN_TTIME.append(tt5.HOUR)
    ROUTE_MAX_TTIME.append(tt6.HOUR)
    ROUTE_MEAN_TTIME.append(values['MEAN'].sum() / values['avg_travel_time'].sum())
    ROUTE_LAST_TTIME.append(tt6.HOUR - tt5.HOUR)

INTERSECTION_ID_df = pd.DataFrame(INTERSECTION_ID)
TOLLGATE_ID_df = pd.DataFrame(TOLLGATE_ID)
ROUTE_HOUR_head_df = pd.DataFrame(ROUTE_HOUR_head)
ROUTE_PCT_head_df = pd.DataFrame(ROUTE_PCT_head)
ROUTE_MIN_TTIME_df = pd.DataFrame(ROUTE_MIN_TTIME)
ROUTE_MAX_TTIME_df = pd.DataFrame(ROUTE_MAX_TTIME)
ROUTE_LAST_TTIME_df = pd.DataFrame(ROUTE_LAST_TTIME)
ROUTE_MEAN_TTIME_df = pd.DataFrame(ROUTE_MEAN_TTIME)
ROUTE_HOUR_INFO = pd.concat([INTERSECTION_ID_df,TOLLGATE_ID_df,ROUTE_HOUR_head_df,ROUTE_PCT_head_df,ROUTE_MIN_TTIME_df,ROUTE_MAX_TTIME_df,ROUTE_LAST_TTIME_df,ROUTE_MEAN_TTIME_df],axis = 1)
ROUTE_HOUR_INFO.columns = ['INTERSECTION_ID','TOLLGATE_ID','ROUTE_HOUR_head','ROUTE_PCT_head','ROUTE_MIN_TTIME','ROUTE_MAX_TTIME','ROUTE_LAST_TTIME','ROUTE_MEAN_TTIME']

#%%  ratio of weekday and weekend median
INTERSECTION_ID = []
TOLLGATE_ID = []
WORK_DAY_MEDIAN = []
WEEKEND_MEDIAN = []

for key,values in AVG_TIME20.groupby(['intersection_id','tollgate_id']):
    values.reset_index(inplace=True)
    #get wk(weekend)/wd(work day)
    wd = values[values['HOLI']==0].median()
    wk = values[values['HOLI']>=1].median()
    INTERSECTION_ID.append(values['intersection_id'][0])
    TOLLGATE_ID.append(values['tollgate_id'][0])
    WEEKEND_MEDIAN.append(wd.avg_travel_time)
    WORK_DAY_MEDIAN.append(wk.avg_travel_time)


INTERSECTION_ID_df = pd.DataFrame(INTERSECTION_ID)
TOLLGATE_ID_df = pd.DataFrame(TOLLGATE_ID)
WEEKEND_MEDIAN_df = pd.DataFrame(WEEKEND_MEDIAN)
WORK_DAY_MEDIAN_df = pd.DataFrame(WORK_DAY_MEDIAN)
ROUTE_WORK_WEEKEND_INFO = pd.concat([INTERSECTION_ID_df,TOLLGATE_ID_df,WEEKEND_MEDIAN_df,WORK_DAY_MEDIAN_df],axis = 1)

ROUTE_WORK_WEEKEND_INFO.columns = ['intersection_id','tollgate_id','WEEKEND_MEDIAN','WORK_DAY_MEDIAN'];

#%%
HOLI_list = [0,0,0,0,0,1,1]
DofW_list = [0,1,2,3,4,5,6]

for holi_ind, DofW_ind in zip(HOLI_list,DofW_list):
    DAYOFWEEK = AVG_TIME20[(AVG_TIME20['HOLI']==holi_ind)&(AVG_TIME20['DAY']==DofW_ind)].groupby(['intersection_id','tollgate_id'],as_index = False).median()
    DAYOFWEEK = DAYOFWEEK[['intersection_id','tollgate_id','avg_travel_time']].rename(columns = {'avg_travel_time':'d'+str(DofW_ind)})
    ROUTE_WORK_WEEKEND_INFO = pd.merge(ROUTE_WORK_WEEKEND_INFO, DAYOFWEEK, on=['intersection_id','tollgate_id'],how = 'left')

ROUTE_SI =  [(lambda x:('SI'+ str(x).zfill(2))) (x)  for x in range(9)]
ROUTE_WORK_WEEKEND_INFO.columns = ['INTERSECTION_ID','TOLLGATE_ID'] + ROUTE_SI
ROUTE_WORK_WEEKEND_INFO['FIX'] = ROUTE_WORK_WEEKEND_INFO[ROUTE_SI].mean(axis = 1)
ROUTE_WORK_WEEKEND_INFO[ROUTE_SI] = ROUTE_WORK_WEEKEND_INFO[ROUTE_SI].div( ROUTE_WORK_WEEKEND_INFO['FIX'],axis = 0)
# del ROUTE_WORK_WEEKEND_INFO['FIX']
ROUTE_WORK_WEEKEND_INFO['RATIO'] = ROUTE_WORK_WEEKEND_INFO['SI00']/ ROUTE_WORK_WEEKEND_INFO['SI01']

# Merge aLL route features
ROUTES = pd.merge(ROUTES,ROUTE_WORK_WEEKEND_INFO, on=['INTERSECTION_ID','TOLLGATE_ID'],how = 'left')
ROUTES = pd.merge(ROUTES,ROUTE_HOUR_INFO, on=['INTERSECTION_ID','TOLLGATE_ID'],how = 'left')

#%%
ROUTES.to_csv('ROUTES_FEATURES.csv',index = False)