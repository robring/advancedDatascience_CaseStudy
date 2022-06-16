import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from sklearn.cluster import DBSCAN
import json
#Regressions Modelle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from itertools import chain, combinations

class Model():
    def __init__(self, model, data, comb, df, features):
        self.model = model
        self.data = data
        self.comb = comb
        self.df = df
        self.features = features

    def set_model(self, model):
        self.model = model
    def get_model(self):
        return self.model
    def set_data(self, X_train, X_test, y_train, y_test):
        self.data = (X_train, X_test, y_train, y_test)
    def get_data(self):
        return self.data
    def set_comb(self, comb):
        self.comb = comb
    def get_comb(self):
        return self.comb
    def set_df(self, df):
        self.df = df
    def get_df(self):
        return self.df
    def set_features(self, features):
        self.features = features
    def get_features(self):
        return self.features
    def train(self):
        X_train, y_train = self.get_data()[0], self.get_data()[2]
        return self.get_model().fit(X_train, y_train)
    def pred(self):
        return self.get_model().predict(self.get_data()[1])
    def summary(self):
        X_test, y_test, y_pred = self.get_data()[1], self.get_data()[3], self.pred()
        mae = mean_absolute_error(y_test, y_pred)
        score = self.get_model().score(X_test, y_test)
        return np.round(mae, 2), np.round(score, 6)
    def getDF(self):
        X_train, X_test, y_train, y_test = self.data
        X = np.append(X_train, X_test, axis=0)
        y = np.append(y_train, y_test.reshape(len(y_test), 1), axis=0)
        X_y = np.append(X, y, axis=1)
        features = df.columns.to_list()
        features.remove("price")
        df = pd.DataFrame(X_y, columns=features)
        return df

    def __repr__(self):
        return (f'{bold("Model:")} {blue(self.model)} {bold("Combination:")} {blue(self.comb)} {bold("Features:")} {blue(len(self.df.columns)-1)}') #-1 bei columns wegen preis

maeList = []

def green(txt):
    return f"\x1b[32m{txt}\x1b[0m"
def red(txt):
    return f"\x1b[31m{txt}\x1b[0m"
def blue(txt):
    return f"\x1b[36m{txt}\x1b[0m"
def bold(txt):
        return f"\x1b[1m{txt}\x1b[0m"

def dropMissingValues(df):
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df

# def reg_train_test(X_train, X_test, y_train, y_test):
#     '''Function for building Basic Regression Model'''
#     # fit the model
#     model = LinearRegression()
#     model.fit(X_train, y_train)

#     # evaluate the model
#     ypred = model.predict(X_test)
    
#     # evaluate predictions
#     mae = mean_absolute_error(y_test, ypred)
#     maeList.append(np.round(mae))
#     #print(f'{bold("Mean Absolute Error")}: {blue(np.round(mae))}\n')

#     # #print(bold(f'MAE_List expanded:'))
#     # for i, m in enumerate(maeList):
#     #     if i+1 == len(maeList):
#     #         print(f'model_{bold(i)} - "Mean Absolute Error:" {blue(m)}') 
#     #     else:
#     #         print(f'model_{bold(i)} - "Mean Absolute Error:" {m}')
#     print(np.round(mae))
#     return model

def splitData(df, test_size = 0.25, outlier_index_list = [], method = None, replace_index_list=[]):
    '''function for splitting the data from a given df into the given test_size proportions'''
    listLenOriginal = len(outlier_index_list) #just for for output
    # Select price as label and remove price_data from list
    X, y = df.drop(columns=["price"]), df["price"]
    # Transform Column to a numeric value
    if 'date' in df:
        X[["date"]] = X[["date"]].apply(pd.to_numeric)
    # Dataframes in numpy-Arrays konvertieren
    X,y  = np.array(X.values.tolist()), np.array(y.values.tolist())
    #split Data and train the model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=1)

    if method != None:
        #create columnList to transform X_Train
        column_list = df.columns.to_list().remove('price')

        #transfrom train Data into df to drop the outliers
        df_X_Train = np_to_df(X_train, column_list)
        df_y_Train = np_to_df(y_train, ['price'])

        #calculate max index --> we only want to delete the outliers below this threshold
        maxIndex = df_X_Train.index.stop

        for o in list(outlier_index_list):
            #rint(type(o))
            if o >= maxIndex:
                outlier_index_list.remove(o) 
        
        # if method == "drop":
        #     #drop the outlierts from the dfs 
        #     df_X_Train = df_X_Train.drop(df_X_Train.index[outlier_index_list])
        #     df_y_Train = df_y_Train.drop(df_y_Train.index[outlier_index_list])
        #     print(f'dropped {red(len(outlier_index_list))} / {listLenOriginal} rows')
        # elif method == "replace":
        #     mean = np.round(df["price"].mean())
        #     df_y_Train.loc[outlier_index_list,'price'] = mean
        #     print(f'replaced {red(len(outlier_index_list))} / {listLenOriginal} rows')

        # else:
        outlier_index_list = list(set(outlier_index_list) - set(replace_index_list))
        mean = np.round(df["price"].mean())
        df_y_Train.loc[outlier_index_list,'price'] = mean
        df_X_Train = df_X_Train.drop(df_X_Train.index[outlier_index_list])
        df_y_Train = df_y_Train.drop(df_y_Train.index[outlier_index_list])
        print(f'dropped {red(len(outlier_index_list))} / {listLenOriginal} rows')
        #print(f'replaced {red(len(replace_index_list))} / {listLenOriginal} rows')
            
        #transfrom back trainigdata to np_arrays
        X_train = df_to_np(df_X_Train)
        y_train = df_to_np(df_y_Train)
    
    return X_train, X_test, y_train, y_test

def np_to_df(numpy_arr, column_list):
    df = pd.DataFrame(numpy_arr, columns=column_list)
    return df

def df_to_np(df):
    np_arr = df.to_numpy()
    return np_arr

def get99(df):
    list99 = df.index[df['price'] == 9999999.9].tolist()
    list90 = df.index[df['price'] == 99999999.0].tolist()
    list99_combined =  list(set(list99) | set(list90))
    return list99_combined


def drop99_all(df, outlier_index_list):
    return df.drop(df.index[outlier_index_list])

def z_score(df, std_multiply=3):
    '''Univariate outlier detection based on descriptive statistics (three standard deviations)
    can be useful to identify extreme outliers'''

    feature_list=['price', 'bedrooms', 'bathrooms', 'sqft_living',
        'sqft_lot', 'floors', 'dis_super', 'view', 'condition',
        'grade', 'sqft_above', 'sqft_basement',
        'sqft_living15', 'sqft_lot15']

    outliers_dict = {}#dict for storing outlierts for an outlier summary df
    outlier_list_unique = []
    #print(bold("Potential Outliers:"))
    for feature in feature_list:
        feature_data = df[feature]

        df_feature = pd.concat([feature_data], axis=1)
        df_feature["outlier"] = 0

        three_std=feature_data.std()*std_multiply
        mean=feature_data.mean()

        inlier_low=mean-three_std
        inlier_high=mean+three_std

        outlier_list = [] #list for storing indexes of outliers
        for i, value in enumerate(feature_data):
            if value < inlier_low or value > inlier_high:
                outlier_list.append(i)
                df_feature.iloc[i,1] = 1      

        #print(f'{bold(feature)} detected: {blue(len(outlier_list))}')
        
        if not len(outlier_list) == 0:
            outliers_dict[str(feature)]=outlier_list
            outlier_list_unique =  list(set(outlier_list_unique) | set(outlier_list))
    
    return outlier_list_unique

def z_score_individual(df):
       if "id" in df.columns:
              df = df.drop(columns=["id"])

       limit = {'date':3, 'price':10, 'bedrooms':7, 'bathrooms':8, 'sqft_living':8, 'sqft_lot':17,
              'floors':5, 'waterfront':100, 'dis_super':100, 'view':100, 'condition':100, 'grade':100,
              'sqft_above':6, 'sqft_basement':6, 'yr_built':3, 'yr_renovated':100, 'zipcode':3,
              'lat':3, 'long':5, 'sqft_living15':5, 'sqft_lot15':16, 'ahf1':3, 'ahf2':4, 'ahf3':3}
       x=0
       outlier_indice = []
       z_score_mask = df.assign(outlier = False).outlier
       for i in df.columns:
              local_mask = df.assign(outlier = np.logical_or(df[i] > df[i].mean() + df[i].std() * limit[i], df[i] < df[i].mean() - df[i].std() * limit[i])).outlier
              z_score_mask = np.logical_or(z_score_mask, local_mask)
              for e in z_score_mask[z_score_mask == True].index:
                     if e not in outlier_indice:
                            outlier_indice.append(e)

              #print(f"{i}: {local_mask.sum()}", end = " ")
              x+=1
       #print(f"\n\nEs wurden {blue(str(len(outlier_indice)))} Ausreißer gefunden. Sie sind auf den Graphen gelb dargestellt.")
       return outlier_indice

def outliers_knn(df, k=3, num_outliers=181, split_size=0.25):
    #X_train needed
    X_train, X_test, y_train, y_test = splitData(df, split_size)

    #normalize data to identify outliers
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X_train)

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    outlier_indices=np.argpartition(distances[:,1],-num_outliers)[-num_outliers:]
 
    return outlier_indices

def outliers_dbscan(df, k=3, num_outliers=181, eps=0.42, min_samples=5, split_size=0.25):

    #need distances
    X_train, X_test, y_train, y_test = splitData(df, split_size)
    scaler = preprocessing.MinMaxScaler()
    X = scaler.fit_transform(X_train)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    inliers_list=[]
    outliers_list=[]
    index_upper=distances[:,1].size

    for index in range (0,index_upper):
        if clustering.labels_[index] == -1:
            outliers_list.append(index)
        else:
            inliers_list.append(index)

    return outliers_list

def summary(maeList):
    baseline = maeList[0]
    for i, model in enumerate(maeList):
        if model > baseline:
            print(red(f'model: {i}: {model}'))
        elif model == baseline:
            print(f'model: {i}: {model}')
        else:
            print(green(f'model: {i}: {model}'))

#Split DataSet into data and target
def getNoise(df, cv=5): #TODO zscore variable gestalten 

    df_noise = df.drop(['date'], axis = 1)
    x = df_noise.iloc[:,2:]
    y = df_noise.iloc[:,1]

    reg1 = GradientBoostingRegressor(random_state=1)
    reg2 = BayesianRidge()
    reg3 = DecisionTreeRegressor(max_depth=5, random_state=1)

    reg1.fit(x,y)
    reg2.fit(x,y)
    reg3.fit(x,y)

    ereg = VotingRegressor([('gb', reg1),('brr',  reg2),('dtr', reg3)])

    ereg.fit(x, y)

    y_pred=cross_val_predict(ereg,x,y, cv=cv)

    xt = x[:20]
    #real = y[:20]
    pred1 = reg1.predict(xt)
    pred2 = reg2.predict(xt)
    pred3 = reg3.predict(xt)
    pred4 = ereg.predict(xt)
    y_pred20 = y_pred[:20]

    mae=mean_absolute_error(y_pred,y)
    noise_id=[]
    for i, e in enumerate(y):
        if y_pred[i] > e+mae*10:
            noise_id.append(i)
        elif y_pred[i] < e-mae*10:
            noise_id.append(i)    

    #print(f"Bei dem 10fachen MAE kann man bis zu {red(len(noise_id))} Noise Sätze finden")
    noise_index_list = df.iloc[noise_id,].index
    noise_index_list = noise_index_list.to_list()
    return noise_index_list

def getRelFeatures(df, threshold=0.3):
    corr =df.corr(method="spearman")
    rel_features =[]
    corr_fig = corr["price"]
    ix = corr.sort_values('price', ascending=False).index
    for i in ix:
        #print(i)
        if corr_fig[i]>= threshold or corr_fig[i]<=-threshold:
            rel_features.append(i)
    return rel_features

def drop_features(df, feature_list):
    try:
        return df[feature_list]
    except:
        print(f'Error while trying to drop features')

def getCombinations(iterable):
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def get_unique_list(outlier_dict_all, combination):
    lists_rel = []
    #print(combination)
    for c in combination:
        lists_rel.append(outlier_dict_all[c])

    list_unique = lists_rel[0]
    if len(lists_rel) >= 2:
        for i, l in enumerate(lists_rel):
            if not i == 0:
                uniques = list(set().union(list_unique, l))
                list_unique = uniques
    return list_unique   

def getBestModel(model_obj_list, df_summary, i):
    for m in model_obj_list:
        mae, score = m.summary()
        if mae == df_summary["mae"][i]:
        #if m.get_comb() == df_summary["combo"][i]:
            return m
        
def train_test_to_df(X_train, X_test, y_train, y_test, columns):
    X = np.append(X_train, X_test, axis=0)
    y = np.append(y_train, y_test.reshape(len(y_test), 1), axis=0)
    #y = np.append(y_train, y_test, axis=0)
    X_y = np.append(X, y, axis=1)
    #features = df.columns.to_list()
    df = pd.DataFrame(X_y, columns=columns)
    return df