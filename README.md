HW1_House-Predict
-----------
# 工作環境
    1.Ubuntu
    2.Python3.6
    3.Keras
    4.Spyder
# 引入函式庫
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn import preprocessing
# 讀取資料並取出輸入與輸出
    dataframe=pd.read_csv('train-v3.csv')
    dataframe=dataframe.drop(['sale_yr','sale_month','sale_day','lat','long','zipcode','yr_renovated'],axis=1)
    dataset=dataframe.values
    features=dataset[:,2:]
    prices=dataset[:,1:2]
    dataframe=pd.read_csv('test-v3.csv')
    dataframe=dataframe.drop(['sale_yr','sale_month','sale_day','lat','long','zipcode','yr_renovated'],axis=1)
    dataset=dataframe.values
    test_features=dataset[:,1:]
# 標準化數據
    scaler = preprocessing.StandardScaler().fit(features)
    features=scaler.transform(features)
    test_features=scaler.transform(test_features)
# 建立模型
    model = Sequential()
    model.add(Dense(30, input_dim=14, kernel_initializer='normal',activation='relu' )) 
    model.add(Dense(30, kernel_initializer='normal',activation='relu' ))
    model.add(Dense(30, kernel_initializer='normal',activation='relu' ))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_absolute_error', optimizer='adam')
# 測試模型
    history = model.fit(features,prices, nb_epoch= 1000 , batch_size= 50 )
    output=model.predict(test_features)
    np.savetxt('test.csv',output,delimiter=',')
# 執行過程
![image](https://github.com/106318143yenyuhsiang/HW1_House-Predict/blob/master/run.JPG)
# Kaggle排名
![image](https://github.com/106318143yenyuhsiang/HW1_House-Predict/blob/master/rank.JPG)
