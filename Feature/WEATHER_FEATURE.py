import pandas as pd
import datetime

def SSD(Temp,Velo,Humi):
    score = (1.818*Temp+18.18) * (0.88+0.002*Humi) + 1.0*(Temp -32)/(45-Temp) - 3.2*Velo  + 18.2
    return score

TEST_WEATHER = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/weather (table 7)_test1.csv")
TRAIN_WEATHER_UPDATE = pd.read_csv("C:/Users/Administrator/PycharmProjects/Traffic/Data/weather (table 7)_training_update.csv")

MIN20_N = 109*24*3
end_date = datetime.datetime.strptime('2016-10-17 23:40:00','%Y-%m-%d %H:%M:%S')
date_list = [str((end_date - datetime.timedelta(minutes=x*20))) for x in range(MIN20_N)]
date_list.reverse()
date_list_df = pd.DataFrame(date_list,columns={'concrete_date'})
date_list_df['hour'] = date_list_df['concrete_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour)
date_list_df['date'] = date_list_df['concrete_date'].apply(lambda x : str(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date()))

TRAIN_WEATHER_UPDATE_FULL = pd.merge(date_list_df,TRAIN_WEATHER_UPDATE,on=['hour','date'],how='outer')
TRAIN_WEATHER_UPDATE_FULL.fillna(method='ffill',inplace=True)
TRAIN_WEATHER_UPDATE_FULL['SSD'] = SSD(TRAIN_WEATHER_UPDATE_FULL['temperature'],TRAIN_WEATHER_UPDATE_FULL['wind_speed'],TRAIN_WEATHER_UPDATE_FULL['rel_humidity'])
TRAIN_WEATHER_UPDATE_FULL.drop(['temperature','rel_humidity','wind_speed'],axis=1,inplace=True)

TRAIN_WEATHER_UPDATE_FULL.to_csv("TRAIN_WEATHER_UPDATE_FULL.csv",index=False,encoding='utf-8')
MIN20_N = 7
end_date = datetime.datetime.strptime('2016-10-24 23:40:00','%Y-%m-%d %H:%M:%S')
date_list = [str((end_date - datetime.timedelta(minutes=x*20))) for x in range(MIN20_N)]
date_list.reverse()
date_list_df = pd.DataFrame(date_list,columns={'concrete_date'})
date_list_df['hour'] = date_list_df['concrete_date'].apply(lambda x : datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').hour)
date_list_df['date'] = date_list_df['concrete_date'].apply(lambda x : str(datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S').date()))

TEST_WEATHER_FULL = pd.merge(date_list_df,TEST_WEATHER,on=['hour','date'],how='outer')
TEST_WEATHER_FULL.fillna(method='ffill',inplace=True)
TEST_WEATHER_FULL['SSD'] = SSD(TEST_WEATHER_FULL['temperature'],TEST_WEATHER_FULL['wind_speed'],TEST_WEATHER_FULL['rel_humidity'])
TEST_WEATHER_FULL.drop(['temperature','rel_humidity','wind_speed'],axis=1,inplace=True)

TEST_WEATHER_FULL.to_csv("TEST_WEATHER_FULL.csv",index=False,encoding='utf-8')
