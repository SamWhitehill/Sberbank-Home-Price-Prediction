import pandas as pd
import numpy as np

# visualization
#import seaborn as sns
import matplotlib as plt
#%matplotlib inline
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from scipy.optimize import differential_evolution
from HamnerMetrics import rmsle
from sklearn import preprocessing

FILE_PATH='' # '../input/'

if FILE_PATH=='':
    from HamnerMetrics import rmsle
    

def clearNAs(pDf):
    pDf.replace([np.inf, -np.inf], np.nan,inplace=True)
    for fld in pDf.columns:
        pDf[fld].fillna(0, inplace=True)
    return pDf


def fEncodeValues(pData):
    for c in pData.columns:
        if pData[c].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(pData[c].values)) 
            pData[c] = lbl.transform(list(pData[c].values))
    
    return pData


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''
    
    # Initialize new output DataFrame
    output = pd.DataFrame(index = X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():
        
        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object or col[:3]=='ID_':
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)  
        
        # Collect the revised columns
        output = output.join(col_data)
    
    return output

    #X_all = preprocess_features(X_all)
    #print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))



def getPCAFeatures(trainData,n_components=20,pPCA=None):
    if pPCA ==None:
        pca = PCA(n_components=n_components).fit(trainData)
    else:
        pca=pPCA
    if True:
        result = pca.transform(trainData)


    return result, pca
        
def setupData():
    np.random.seed(10)
    pd.set_option('display.max_columns', 30)

    macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
    "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
    "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]

    lstFeatures=['full_sq'	,
    'life_sq'	,
    'floor'	,
    'build_year'	,
    'life_pct'	,
    'eurrub'	,
    'micex_cbi_tr'	,
    'micex_rgbi_tr'	,
    'kindergarten_km'	,
    'max_floor'	,
    'railroad_km'	,
    'public_transport_station_km'	,
    'green_zone_km'	,
    'rel_floor'	,
    'rel_kitch'	,
    'public_healthcare_km'	,
    'metro_km_avto'	,
    'mortgage_rate'	,
    'metro_min_avto'	,
    'workplaces_km'	,
    'mosque_km'	,
    'bus_terminal_avto_km'	,
    'big_market_km'	,
    'state'	,
    'swim_pool_km'	,
    'balance_trade'	,
    'big_road2_km'	,
    'basketball_km'	,
    'market_shop_km'	,
    'big_road1_km'	,
    'mortgage_value'	,
    'thermal_power_plant_km'	,
    'additional_education_km'	,
    'incineration_km'	,
    'area_m'	,
    'week_year_cnt'	,
    'park_km'	,
    'school_km'	,
    'cemetery_km'	,
    'fitness_km'	,
    'industrial_km'	,
    'catering_km'	,
    'railroad_station_walk_km'	,
    'ts_km'	,
    'income_per_cap'	,
    'oil_chemistry_km'	,
    'hospice_morgue_km'	,
    'preschool_km'	,
    'power_transmission_line_km'	,
    'detention_facility_km'	,
    'exhibition_km'	,
    'green_part_1000'	,
    'water_treatment_km'	,
    'church_synagogue_km'	,
    'nuclear_reactor_km'	,
    'railroad_station_avto_km'	,
    'water_km'	,
    'trc_sqm_5000'	,
    'mkad_km'	,
    'kitch_sq'	,
    'ice_rink_km'	,
    'green_part_5000'	,
    'big_church_km'	,
    'theater_km'	,
    'stadium_km'	,
    'green_part_1500'	,
    'prom_part_3000'	,
    'green_part_2000'	,
    'green_zone_part'	,
    'month_year_cnt'	,
    'shopping_centers_km'	,
    'green_part_500'	,
    'balance_trade_growth'	,
    'radiation_km'	,
    'cafe_sum_500_min_price_avg'	,
    'university_km'	,
    'prom_part_5000'	,
    'office_km'	,
    'rent_price_4+room_bus'	,
    'metro_min_walk'	,
    'deposits_rate'	,
    'cafe_sum_3000_min_price_avg'	,
    'dow'	,
    'green_part_3000'	,
    'zd_vokzaly_avto_km'	,
    'indust_part'	,
    'ttk_km'	,
    'office_sqm_2000'	,
    'average_provision_of_build_contract'	,
    'num_room'	,
    'material'	,
    'museum_km'	,
    'cafe_sum_1000_min_price_avg'	,
    'raion_popul'	,
    'trc_sqm_1500'	,
    'month'	,
    'cafe_sum_1500_min_price_avg'	,
    'cafe_sum_3000_max_price_avg'	,
    'preschool_quota'	,
    'cafe_sum_500_max_price_avg'	,
    'hospital_beds_raion'	,
    'railroad_station_avto_min'	,
    'cafe_sum_2000_min_price_avg'	,
    'trc_sqm_2000'	,
    'cafe_count_5000'	,
    'build_count_block'	,
    'ID_metro'	,
    'prom_part_1500'	,
    'prom_part_2000'	,
    'cafe_sum_5000_min_price_avg'	,
    'cafe_sum_1000_max_price_avg'	,
    'office_sqm_1500'	,
    'prom_part_1000'	,
    'raion_build_count_with_material_info'	,
    'trc_count_1500'	,
    'ID_railroad_station_walk'	,
    'trc_sqm_1000'	,
    'product_type'	,
    'trc_sqm_3000'	,
    'metro_km_walk'	,
    'sport_count_3000'	,
    'trc_count_3000'	,
    'cafe_sum_1500_max_price_avg'	,
    'cafe_count_5000_price_2500'	,
    'sport_objects_raion'	,
    'public_transport_station_min_walk'	,
    'cafe_count_1000'	,
    'cafe_count_2000_price_1500'	,
    'cafe_count_2000'	,
    'build_count_monolith'	,
    'trc_count_2000'	,
    'market_count_1000'	,
    'ID_big_road2'	,
    'market_count_3000'	,
    'cafe_avg_price_5000'	,
    'railroad_station_walk_min'	,
    'cafe_count_1500'	,
    'office_sqm_500'	,
    'shopping_centers_raion'	,
    'cafe_sum_2000_max_price_avg'	,
    'office_sqm_5000'	,
    'sport_count_2000'	,
    'cafe_count_500'	,
    'prom_part_500'	,
    'bulvar_ring_km'	,
    'cafe_avg_price_3000'	,
    'cafe_count_1500_price_1500'	,
    'sport_count_500'	,
    'sub_area'	,
    'cafe_avg_price_500'	,
    'market_count_5000'	,
    'ekder_female'	,
    'sport_count_1000'	,
    'ID_railroad_station_avto'	,
    'sadovoe_km'	,
    'cafe_count_3000_price_500'	,
    'church_count_5000'	,
    'cafe_count_500_price_1500'	,
    'cafe_count_1000_price_1000'	,
    'cafe_count_1000_price_high'	,
    'cafe_avg_price_1000'	,
    'cafe_count_2000_price_500'	,
    'museum_visitis_per_100_cap'	,
    'build_count_brick'	,
    'full_all'	,
    'cafe_count_5000_price_high'	,
    'school_quota'	,
    'office_sqm_1000'	,
    '16_29_all'	,
    'cafe_count_3000'	,
    'build_count_1971-1995'	,
    'church_count_500'	,
    'build_count_1946-1970'	,
    'cafe_sum_5000_max_price_avg'	,
    'cafe_avg_price_2000'	,
    'children_preschool'	,
    'cafe_count_1500_price_1000'	,
    'cafe_count_5000_price_500'	,
    'ekder_male'	,
    'ID_big_road1'	,
    'cafe_count_2000_price_1000'	,
    'office_count_2000'	,
    'build_count_1921-1945'	,
    'church_count_3000'	,
    'build_count_frame'	,
    'cafe_count_5000_na_price'	,
    'cafe_count_1000_price_1500'	,
    'cafe_count_2000_price_2500'	,
    'school_education_centers_raion'	,
    'office_count_5000'	,
    'cafe_avg_price_1500'	,
    'big_church_count_3000'	,
    'sport_count_1500'	,
    'leisure_count_3000'	,
    'office_count_500'	,
    'cafe_count_500_price_2500'	,
    'church_count_2000'	,
    'office_count_1500'	,
    'build_count_wood'	,
    'male_f'	,
    'ekder_all'	,
    'cafe_count_1000_na_price'	,
    'office_count_1000'	,
    'office_sqm_3000'	,
    'build_count_mix'	,
    'cafe_count_1000_price_4000'	,
    'cafe_count_2000_na_price'	,
    'market_count_1500'	,
    'office_count_3000'	,
    'cafe_count_3000_price_1000'	,
    'trc_count_1000'	,
    'cafe_count_3000_na_price'	,
    'raion_build_count_with_builddate_info'	,
    'leisure_count_5000'	,
    'cafe_count_5000_price_4000'	,
    'children_school'	,
    'cafe_count_3000_price_2500'	,
    'big_church_count_2000'	,
    'cafe_count_5000_price_1000'	,
    'cafe_count_1000_price_500'	,
    'sport_count_5000'	,
    'trc_count_5000'	,
    'cafe_count_1500_price_4000'	,
    'cafe_count_3000_price_1500'	,
    'female_f'	,
    'mosque_count_5000'	,
    'ID_bus_terminal'	,
    'trc_sqm_500'	,
    'work_female'	,
    'market_count_500'	,
    'work_male'	,
    'kremlin_km'	,
    'church_count_1500'	,
    'work_all'	,
    'preschool_education_centers_raion'	,
    'church_count_1000'	,
    'leisure_count_1000'	,
    'build_count_after_1995'	,
    '7_14_male'	,
    'office_raion'	,
    'cafe_count_500_price_4000'	,
    'apartment_build'	,
    'university_top_20_raion'	,
    'build_count_panel'	,
    'cafe_count_500_na_price'	,
    'cafe_count_5000_price_1500'	,
    '16_29_male'	,
    'cafe_count_500_price_500'	,
    'cafe_count_1500_price_500'	,
    '16_29_female'	,
    'additional_education_raion'	]

    lstFeatures =lstFeatures[:115]

    #try selectkbest or variancethreshold
    #http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html#sphx-glr-auto-examples-plot-compare-reduction-py

    train_df = pd.read_csv(FILE_PATH+'train.csv',parse_dates=['timestamp'])

    train_df.reindex(np.random.permutation(train_df.index))

    #train_df=train_df[:10000]
    print ("ONLY Using " + str(len(train_df)) +" rows of train data, but in random order!")

    test_df = pd.read_csv(FILE_PATH+'test.csv',parse_dates=['timestamp'])
    macro_df=pd.read_csv(FILE_PATH+'macro.csv', parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)
    #macro_df=pd.read_csv('../input/macro.csv',parse_dates=['timestamp'])
    train_df.shape

    train_df = pd.merge_ordered(train_df, macro_df, on='timestamp', how='left')
    result_df = pd.merge_ordered(test_df, macro_df, on='timestamp', how='left')
    combine=[train_df,result_df]
    train_df.head()


    #Cleaning
    for df in combine:
        df.ix[df.life_sq<2, 'life_sq'] = np.nan
        df.ix[df.build_year<1500,'build_year']=np.nan
        df.ix[df.max_floor<df.floor,'max_floor']=np.nan
        df.ix[df.full_sq<2,'full_sq']=df.ix[df.full_sq<2,'life_sq']
        df.ix[df.full_sq<df.life_sq,'life_sq']=np.nan
        df.ix[df.kitch_sq>df.life_sq,'kitch_sq']=np.nan
        df.ix[df.kitch_sq<2,'kitch_sq']=np.nan
        df.ix[df.floor==0,'floor']=np.nan
        df.ix[df.max_floor==0,'max_floor']=np.nan
        df.ix[df.max_floor>70,'max_floor']=np.nan
        df.ix[df.num_room==0,'num_room']=np.nan
    #In [5]:
    #New features
    for df in combine:
        df['life_pct']=df['life_sq']/df['full_sq'].astype(float)
        df['rel_kitch']=df['kitch_sq']/df['full_sq'].astype(float)
        df['rel_floor']=df['floor']/df['max_floor'].astype(float)
        
        
    #In [6]:
    for df in combine:
        month_year = (df.timestamp.dt.month + df.timestamp.dt.year * 100)
        month_year_cnt_map = month_year.value_counts().to_dict()
        df['month_year_cnt'] = month_year.map(month_year_cnt_map)

        week_year = (df.timestamp.dt.weekofyear + df.timestamp.dt.year * 100)
        week_year_cnt_map = week_year.value_counts().to_dict()
        df['week_year_cnt'] = week_year.map(week_year_cnt_map)
        df['dow'] = df.timestamp.dt.dayofweek
        df['month'] = df.timestamp.dt.month
    #In [7]:
    train_df[['product_type']].describe(include=['O'])

    #train_df[train_df.price_doc==1000000].groupby('month_year_cnt')[['timestamp','full_sq','life_sq','floor','product_type']].count()


    #train_df[(train_df.price_doc==2000000)&(train_df.product_type!='Investment')][['timestamp','full_sq','life_sq','floor','product_type']].head(30)

    if False:
        train_df_numeric = train_df.select_dtypes(exclude=['object'])
        train_df_obj = train_df.select_dtypes(include=['object']).copy()

    
        for column in train_df_obj:
            train_df_obj[column] = pd.factorize(train_df_obj[column])[0]

        train_df_values = pd.concat([train_df_numeric, train_df_obj], axis=1)[:]
        #test_df_values = pd.concat([train_df_numeric, train_df_obj], axis=1)[25001:] #validation set
        #In [11]:
        result_df_numeric = result_df.select_dtypes(exclude=['object'])
        result_df_obj = result_df.select_dtypes(include=['object']).copy()

        for column in result_df_obj:
            result_df_obj[column] = pd.factorize(result_df_obj[column])[0]

        result_df_values = pd.concat([result_df_numeric, result_df_obj], axis=1)
    #In [12]:

    #X_train = train_df_values[(train_df_values.full_sq<1000)&(train_df_values.price_doc!=1000000)&(train_df_values.price_doc!=2000000)].drop(['price_doc','id','timestamp'],axis=1)

    #Y_train = np.log1p(train_df_values[(train_df_values.full_sq<1000)&(train_df_values.price_doc!=1000000)&(train_df_values.price_doc!=2000000)]['price_doc'].values.reshape(-1,1))

    #X_train.shape
    #Out[12]:
    #(28945, 309)
    #In [13]:
    #X_train = train_df_values.drop(['price_doc','id','timestamp'],axis=1)
    X_train = train_df.drop(['price_doc','id','timestamp'],axis=1)

    X_train=X_train[lstFeatures]

    X_train=clearNAs(X_train)
    
    #pre process made for SVR
    X_train=preprocess_features(X_train)
    #X_train=fEncodeValues(X_train)

    Y_train = (train_df['price_doc'].values.reshape(-1,1))
    X_train.shape
    #Out[13]:
    #(30471, 309)
    
    #In [14]:
    X_result = result_df.drop(['id','timestamp'],axis=1)
    X_result=X_result[lstFeatures]
    id_test = result_df['id']
    X_result.shape

    X_result=clearNAs(X_result)

    X_result=preprocess_features(X_result)
    #X_result=fEncodeValues(X_result)
    #Out[14]:
    #(7662, 309)
    #In [15]:
    dtrain = xgb.DMatrix(X_train[:24000], Y_train[:24000])
    dtest = xgb.DMatrix(X_train[24000:], Y_train[24000:])
    
    #In [16]:
    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': .7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',  
        'eval_metric': 'rmse',
        'silent': 0
    }
    # Uncomment to tune XGB `num_boost_rounds`
    #model = xgb.cv(xgb_params, dtrain, num_boost_round=200,
                      #early_stopping_rounds=30, verbose_eval=10)


    y_train =Y_train #X_train["price_doc"]
    x_train =X_train #.drop('price_doc', axis=1, inplace=True)


    if False:
        '''************************************ '''
        '''#scale features'''
        '''************************************ '''
        dataset = x_train.values
        dataset = dataset.astype('float64')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train = scaler.fit_transform(dataset)

        '''Scale test data '''
        X_result
        dataset = X_result.values
        dataset = dataset.astype('float64')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_result = scaler.fit_transform(dataset)

    
    
    #getPCAFeatures
    ''' ***************************************************** '''
    ''' PCA feature reduction '''
    ''' ***************************************************** '''
    nFeatures=90
    #copyXTrain =x_train.copy(deep=True)
    #x_train, pca=getPCAFeatures(x_train,nFeatures)
    
    #test data
    #X_result, pca=getPCAFeatures(X_result,nFeatures,pca)
        
    dresult=xgb.DMatrix(X_result)

    # Train/Valid split
    split = .28 #len(x_train)
    #xx_train, yy_train, xx_valid, yy_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

    xx_train, xx_valid, yy_train, yy_valid = train_test_split(x_train, y_train, test_size=split, random_state=42)


    if False:
        dtrain = xgb.DMatrix(xx_train, yy_train, feature_names=xx_train.columns.values)
        dvalid = xgb.DMatrix(xx_valid, yy_valid, feature_names=xx_valid.columns.values)

        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    if False:
        model = xgb.train(dict(xgb_params), dtrain, 400, watchlist,verbose_eval=50, \
            feval=rmsle, early_stopping_rounds=100)

    return xx_train, yy_train, xx_valid, yy_valid, dresult,id_test,X_result

''' SVR prediction'''
def runSVR(x, *args):
    #xx_train, yy_train, xx_valid, yy_valid, X_result, gamma, C
    #best svr (1254668.3131053746, 0.17444006241096333, 'C, gamma')
    gamma= x[0]
    C=x[1]
    #print ('take out hard coded svr parms')
    print ("SVR requires dummies!")
    xx_train, yy_train, xx_valid, yy_valid, dresult,id_test, X_result =args

    clfSVR =SVR(kernel='rbf',gamma=gamma,C=C,epsilon =.0001)

    lstG =[1e-04, 4e-04, 1e-04, 1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10]

    parameters={'gamma':lstG,'C':[.5,1,2,4,7,10,100]}
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)

    if False:
        clf = GridSearchCV(clfSVR, parameters, scoring =rmsle_scorer,verbose=1,n_jobs=2)

        clf.fit(xx_train, yy_train)
        print(clf.best_params_)
        #{'C': 100, 'gamma': 0.0004}
        #{'C': 10, 'gamma': 1e-07}, kaggle notebook
    else:
        clfSVR.fit(xx_train, yy_train)

    pred =clfSVR.predict(xx_valid)
    #def rmsle(predicted, actual):
    score =rmsle(pred,yy_valid,False)
    print (C,gamma, "C, gamma")
    print('validation rmsle ',score)

    #{'C': 10, 'gamma': 0.04}
    #from kaggle notebook
    #parameters={'gamma':[.00001,.01,.04,.06,.08,.1,.3,.7,2],'C':[.5,1,2,4,7,10,100]}
    #10k training split
    #gives: {'C': 2, 'gamma': 2}

    #clfSVR.fit(xx_train, yy_train)
    #model = xgb.train(xgb_params, dtrain, num_boost_round=1000,feval=rmsle,
    #                 verbose_eval=20, early_stopping_rounds=20, evals=[(dtest,'test')])


    #num_round=model.best_iteration
    #print(num_round)
    #225
    #In [18]:

    #fig, ax = plt.subplots(1, 1, figsize=(8, 16))
    #xgb.plot_importance(model,  height=0.5, ax=ax) #max_num_features=50,

    #fig.show()

    if False:
        df=pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)

        #print feature importances
        for index, row in df.iterrows():
            print  (row['feature'], row['importance'])


    xgb_params = {
        'eta': 0.05,
        'max_depth': 5,
        'subsample': .7,
        'colsample_bytree': 0.7,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }

    # Uncomment to tune XGB `num_boost_rounds`
    #model = xgb.cv(xgb_params, dtrain, num_boost_round=200,
                      #early_stopping_rounds=30, verbose_eval=10)
    #dtrain = xgb.DMatrix(X_train, Y_train)

    #model = xgb.train(xgb_params, dtrain, num_boost_round=240)
    
    if True:
        model=clfSVR

        #y_pred=model.predict(dresult) # using XGBoost
        y_pred=model.predict(X_result) #using SVR
    
        #must convert log price into original price
        #y_pred =np.expm1(y_pred)

        output=pd.DataFrame(data={'price_doc':y_pred-1},index=id_test)
        #In [21]:
        #output.head()
        #SVR scored .55 on kaggle, vs. xgboost of .31
        output.to_csv('output.csv',header=True)

    return score



def runXGBoost(x, *args):
    #xx_train, yy_train, xx_valid, yy_valid, X_result, gamma, C

    xx_train, yy_train, xx_valid, yy_valid, dresult, id_test,X_result =args

    parameters={} #'gamma':lstG,'C':[.5,1,2,4,7,10,100]}
    rmsle_scorer = make_scorer(rmsle, greater_is_better=False)
 
    pEta=x[0]
    pMax_depth=int(x[1])
    pSubsample =x[2]
    pColsample_bytree=x[3]

    
    xgb_params = {
        'eta': pEta,
        'max_depth': pMax_depth,
        'subsample': pSubsample,
        'colsample_bytree': pColsample_bytree,
        'objective': 'reg:linear',
        'silent': 1
    }
    
    print ('hard coded params, take out!')

    #xgb_params= {'colsample_bytree': 0.63730212420557653, 'silent': 1,
    #'subsample': 0.3268173105481037, 'eta': 0.001784428546056302, 
    #'objective': 'reg:linear', 'max_depth': 7}
    #xgb_params= {'colsample_bytree': 0.63730212420557653, 'silent': 1,
    #'subsample': 0.3268173105481037, 'eta': 0.001784428546056302,
    #'objective': 'reg:linear', 'max_depth': 7}

    #('xgb_params', {'colsample_bytree': 0.20390689004431892, 
    #'silent': 1, 'subsample': 0.42982414502167221,
   # 'eta': 0.0065191961329121029, 'objective': 'reg:linear', 'max_depth': 13})

    print ('xgb_params',xgb_params)

    dtrain = xgb.DMatrix(xx_train, yy_train)
    dtest = xgb.DMatrix(xx_valid, yy_valid)

    #clfSVR.fit(xx_train, yy_train)
    model = xgb.train(dict(xgb_params), dtrain, num_boost_round=1500,feval=rmsle,
                     verbose_eval=20, early_stopping_rounds=280, evals=[(dtest,'test')])
    if False:
        df=pd.DataFrame(model.get_fscore().items(), columns=['feature','importance']).sort_values('importance', ascending=False)

        #print feature importances
        for index, row in df.iterrows():
            print  (row['feature'], row['importance'])

    # Uncomment to tune XGB `num_boost_rounds`
    #model = xgb.cv(xgb_params, dtrain, num_boost_round=200,
                      #early_stopping_rounds=30, verbose_eval=10)

    
    if False:
        #model=clfSVR

        #y_pred=model.predict(dresult) # using XGBoost
        y_pred=model.predict(dresult) #using SVR
    
        #must convert log price into original price
        #y_pred =np.expm1(y_pred)

        output=pd.DataFrame(data={'price_doc':y_pred},index=id_test)
        #In [21]:
        #output.head()

        output.to_csv('output.csv',header=True)

    dValid= xgb.DMatrix(xx_valid)
    pred =model.predict(dValid)
    #def rmsle(predicted, actual):
    score =rmsle(pred,yy_valid, False)
    #print (C,gamma, "C, gamma")
    print('validation rmsle ',score)

    return score


if __name__=='__main__':

    # 'eta': 0.05,
    #'max_depth': 10,
    #'subsample': .7,
    #'colsample_bytree': 0.7,
    
    #lBounds=[(.001,.2),(2,30),(.1,.8),(.1,.8)]
    xx_train, yy_train, xx_valid, yy_valid, dresult,id_test,X_result=setupData()
    lBounds=[(.000000000001,.3),(50000,150000000)]
    result=differential_evolution(func=runSVR,bounds=lBounds,
    args=(xx_train, yy_train, xx_valid, yy_valid, dresult,id_test,X_result),disp=1)

    print(result)

   #best svr: (1254668.3131053746, 0.17444006241096333, 'C, gamma')
    #
   #('xgb_params', {'colsample_bytree': 0.63730212420557653, 'silent': 1,
   # 'subsample': 0.3268173105481037, 'eta': 0.001784428546056302, 'objective': 'reg:linear', 'max_depth': 7})