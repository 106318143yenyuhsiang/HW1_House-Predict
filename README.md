-----------
HW1_House-Predict
-----------
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn import preprocessing
# 讀取資料並取出輸入與輸出
    dataframe=pd.read_csv(r'D:\Python\train-v3.csv')
    dataframe=dataframe.drop(['sale_yr','sale_month','sale_day','lat','long','zipcode','yr_renovated'],axis=1)
    dataset=dataframe.values
    features=dataset[:,2:]
    prices=dataset[:,1:2]
    dataframe=pd.read_csv(r'D:\Python\test-v3.csv')
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
    print(output)
