# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import util as util

train = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/training_20min_avg_travel_time-Copy1.csv")
train['start_time'] = train['time_window'].apply(lambda x :(x.split(','))[0][1:])
train['start_time'] = pd.to_datetime(train['start_time'])
# Turn A,B,C to 1,2,3
train.replace('B',2,inplace=True)
train.replace('A',1,inplace=True)
train.replace('C',3,inplace=True)
train.drop(['time_window'],axis=1,inplace=True)

test_Time_Morn = np.array([pd.datetime(2016,10,11,6,0,0),pd.datetime(2016,10,11,7,40,0)])
test_Time_After = np.array([pd.datetime(2016,10,11,15,0,0),pd.datetime(2016,10,11,16,40,0)])

Test_Data = pd.DataFrame({'avg_travel_time':[],'start_time':[],'intersection_id':[],'tollgate_id':[]})
for intersection_id,intersection_id_group in train.groupby(['intersection_id']):
        for tollgate_id,tollgate_id_group  in intersection_id_group.groupby(['tollgate_id']):
            Morning =  util.generateTestData(intersection_id,tollgate_id,train,test_Time_Morn)
            Afternoon =  util.generateTestData(intersection_id,tollgate_id,train,test_Time_After)
            Test_Data = Test_Data.merge(Morning,how='outer')
            Test_Data = Test_Data.merge(Afternoon,how='outer')
test_Time_Morn = np.array([pd.datetime(2016,10,11,8,0,0),pd.datetime(2016,10,11,9,40,0)])
test_Time_After = np.array([pd.datetime(2016,10,11,17,0,0),pd.datetime(2016,10,11,18,40,0)])

True_Data = pd.DataFrame({'avg_travel_time':[],'start_time':[],'intersection_id':[],'tollgate_id':[]})
for intersection_id,intersection_id_group in train.groupby(['intersection_id']):
        for tollgate_id,tollgate_id_group  in intersection_id_group.groupby(['tollgate_id']):
            Morning =  util.generateTestData(intersection_id,tollgate_id,train,test_Time_Morn)
            Afternoon =  util.generateTestData(intersection_id,tollgate_id,train,test_Time_After)
            True_Data = True_Data.merge(Morning,how='outer')
            True_Data = True_Data.merge(Afternoon,how='outer')
Sum = pd.DataFrame({'avg_travel_time':[],'start_time':[],'intersection_id':[],'tollgate_id':[]})
for intersection_id,intersection_id_group in train.groupby(['intersection_id']):
        for tollgate_id,tollgate_id_group  in intersection_id_group.groupby(['tollgate_id']):
            Morning =  util.RunOnCv(intersection_id,tollgate_id,train,True,Test_Data)
            Afternoon =  util.RunOnCv(intersection_id,tollgate_id,train,False,Test_Data)
            Sum = Sum.merge(Morning,how='outer')
            Sum = Sum.merge(Afternoon,how='outer')
print Sum
