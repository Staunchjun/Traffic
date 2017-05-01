# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


# get all date list ,check out which date are missing
import knn as knn


def get_date_list(start, end, toFormat):
    date_list = []
    date = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')
    end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S')
    while date <= end:
        date_list.append(date.strftime(toFormat))
        date = date + datetime.timedelta(minutes=20)
    return date_list


def GetTimeSeries(train, intersection_id, tollgate_id):
    '''
    Get intersection_id & tollgate_id 's Time Series
    '''
    Tag_Data = train[train.intersection_id == intersection_id]
    Tag_Data = Tag_Data[Tag_Data.tollgate_id == tollgate_id]

    MinTime = min(Tag_Data.start_time)
    MaxTime = max(Tag_Data.start_time)
    DataTimeRange = pd.date_range(start=MinTime, end=MaxTime, freq='20Min')
    ts_0 = pd.Series([0] * len(DataTimeRange), index=DataTimeRange)
    ts = pd.Series(Tag_Data.avg_travel_time.values, index=Tag_Data.start_time)
    TS = ts_0 + ts
    #   先用前一个时刻的数据做填充。。。暂时想法。也是比较方便的//或者使用相近的历史数据进行填充/
    TS = TS.fillna(method='pad')
    #     TS = TS.fillna(0)
    return TS


def DrawTsList(ts_list):
    ts_list.T.plot(kind='line', marker='o', legend=False, figsize=[24, 12])
    plt.show()


def Get_Part_of_TimeSeries(TS, TimeRange):
    '''
    Input [start_time,end_time]
    '''
    return TS[TimeRange[0]:TimeRange[1]]


def GenerateTs_0(Time):
    '''
    Input [start_time,end_time]
    '''
    timerange = pd.date_range(start=Time[0], end=Time[1], freq='20Min')
    ts = pd.Series(np.zeros(len(timerange)), index=timerange)
    return ts


def TsList(train, intersection_id, tollgate_id, Time):
    '''
    Input [start_time,end_time]
    '''
    ts_list = []
    ts = GetTimeSeries(train, intersection_id, tollgate_id)
    for i in range(0, 92):
        TimeRange = Time + datetime.timedelta(i)
        ts_part = Get_Part_of_TimeSeries(ts, TimeRange)
        if len(ts_part) == 0 or ts_part.isnull().any():
            ts_list.append(GenerateTs_0(TimeRange))
        else:
            ts_list.append(ts_part)
    c = []
    for x in ts_list:
        a = list(np.array(x))
        c.append(a)
    # return np.array(ts_list)
    return c


def TrueFalseListCombine(TFlist1, TFlist2):
    return [l1 and l2 for l1, l2 in zip(TFlist1, TFlist2)]


def ExceptOutlier(ts_list):
    Mean = pd.DataFrame([np.mean(i) for i in ts_list])
    mean_low = Mean > Mean.quantile(0.2)
    mean_up = Mean < Mean.quantile(0.8)
    TF = TrueFalseListCombine(mean_low.values, mean_up.values)
    mean_index = Mean[TF].index.values
    Std = pd.DataFrame([np.std(i) for i in ts_list])
    std_low = Std > Std.quantile(0.2)
    std_up = Std < Std.quantile(0.8)
    TF = TrueFalseListCombine(std_low.values, std_up.values)
    std_index = Std[TF].index.values
    valid_index = list(set(mean_index) & set(std_index))
    return valid_index

def generateTestData(intersection_id,tollgate_id,train,test_Time):
    td = TsList(train,intersection_id,tollgate_id,test_Time)
    td = pd.DataFrame(td)
    td = td[0:7]
    start_time=[]
    for i in range(0,7):
            TimeRange = test_Time + datetime.timedelta(i)
            time_lists =get_date_list(str(TimeRange[0]),str(TimeRange[1]),'%Y-%m-%d %H:%M:%S')
            for x in time_lists:
                start_time.append(x)

    avg_travel_time =[]
    for  x in  td.values:
         for a in x:
            avg_travel_time.append(a)
    test = pd.DataFrame(columns=["avg_travel_time"],data=avg_travel_time)
    test['start_time'] = start_time
    test['intersection_id']=intersection_id
    test['tollgate_id']=tollgate_id
    return test


def RunOnCv(intersection_id, tollgate_id, train,isMorning,Test_Data ):
    start_time = []
    end_time = []
    if (isMorning):
        train_Time = np.array([pd.datetime(2016, 7, 18, 6, 0, 0), pd.datetime(2016, 7, 18, 9, 40, 0)])
        test_Time = np.array([pd.datetime(2016, 10, 11, 6, 0, 0), pd.datetime(2016, 10, 11, 9, 40, 0)])
        predict = np.array([pd.datetime(2016, 10, 11, 8, 0, 0), pd.datetime(2016, 10, 11, 9, 40, 0)])
    else:
        train_Time = np.array([pd.datetime(2016, 7, 18, 15, 0, 0), pd.datetime(2016, 7, 18, 18, 40, 0)])
        test_Time = np.array([pd.datetime(2016, 10, 11, 15, 0, 0), pd.datetime(2016, 10, 11, 18, 40, 0)])
        predict = np.array([pd.datetime(2016, 10, 11, 17, 0, 0), pd.datetime(2016, 10, 11, 18, 40, 0)])
    ts_list = TsList(train, intersection_id, tollgate_id, train_Time)
    # 最后7天是测试用的
    ts_list = ts_list[0:len(ts_list) - 7]
    # 此处的过滤策略是：对每天特定时间段的数据求均值与标准差，然后将均值与标准差落在10%分位数以下和90%分位数以上的日子去除
    # 我觉得，去除异常的时候，应该把缺省值给补充回来
    valid_index = ExceptOutlier(ts_list)
    ts_list = pd.DataFrame(ts_list)
    ts_list = ts_list.ix[valid_index]

    td = TsList(Test_Data, intersection_id, tollgate_id, test_Time)
    td_list = pd.DataFrame(td)
    #     td_list = td_list.fillna(method='pad')
    td_list = td_list[0:7]
    col = [0, 1, 2, 3, 4, 5]
    label1 = [6]
    label2 = [7]
    label3 = [8]
    label4 = [9]
    label5 = [10]
    label6 = [11]

    label = [label2, label3, label4, label5, label6]
    train = np.array(ts_list[col])
    target = np.array(ts_list[label1])
    test = np.array(td_list[col])
    model = knn.NonparametricKNN(n_neighbors=3, loss='MAPE')
    # Train

    model.fit(train, target)
    # Predict
    Y_predict = model.predict(test)
    result = pd.DataFrame(Y_predict)
    result.rename(columns={0: label1[0]}, inplace=True)
    for eachLabel in label:
        train = np.array(ts_list[col])
        target = np.array(ts_list[eachLabel])
        test = np.array(td_list[col])
        model = knn.NonparametricKNN(n_neighbors=3, loss='MAPE')
        # Train
        model.fit(train, target)
        # Predict
        Y_predict = model.predict(test)
        Y_predict = pd.DataFrame(Y_predict)
        Y_predict.rename(columns={0: eachLabel[0]}, inplace=True)
        result = result.join(Y_predict)

    avg_travel_time = []
    for x in result.values:
        for a in x:
            avg_travel_time.append(a)
    result = pd.DataFrame(columns=["avg_travel_time"], data=avg_travel_time)

    start_time = []
    for i in range(0, 7):
        TimeRange = predict + datetime.timedelta(i)
        time_lists = get_date_list(str(TimeRange[0]), str(TimeRange[1]), '%Y-%m-%d %H:%M:%S')
        for x in time_lists:
            start_time.append(x)

    result['start_time'] = start_time
    result['intersection_id'] = intersection_id
    result['tollgate_id'] = tollgate_id
    return result