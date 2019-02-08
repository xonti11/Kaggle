#Reduction memory. For big date
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
	
#Bayes optimization for lgb
def optim_lgb(n_estimator,max_depth,num_leaves,lambda_l1,lambda_l2,min_split_gain,min_child_weight,bagging_fraction,early_stopping_rounds):
    def run_lgb(train_data,val_data,val_x,n_estimator,max_depth,num_leaves,lambda_l1,lambda_l2,min_split_gain,min_child_weight,bagging_fraction,early_stopping_rounds):
        num_round = 10000
        params = {
                    "objective" : "regression",
                    "metric" : "rmse", 
                    "boosting": "gbdt",
                    "learning_rate" : 0.01,
                    'n_estimators': int(n_estimator),
                     'max_depth':int(round(max_depth)),
                     'num_leaves':int(num_leaves),
                     'lambda_l1':max(lambda_l1, 0),
                     'lambda_l2':max(lambda_l2, 0),
                     'min_split_gain':min_split_gain,
                     'min_child_weight':min_child_weight,
                     'bagging_fraction':max(min(bagging_fraction, 1), 0)
                }
        model=lgb.train(params,train_data,num_round,valid_sets=[train_data,val_data],verbose_eval=False,early_stopping_rounds=int(early_stopping_rounds))

        pred_val = model.predict(val_x, num_iteration=model.best_iteration)

    #    return pred_val
        return  pred_val
    rskf=StratifiedKFold(5,shuffle=True,random_state=315)
    val_pr=np.zeros(len(df_train))
    for train_index,val_index in rskf.split(df_train,df_train['outliers'].values):
        train_data=lgb.Dataset(df_train[df_train_columns].loc[train_index],label=df_train['target'].loc[train_index])
        val_data=lgb.Dataset(df_train[df_train_columns].loc[val_index],label=df_train['target'].loc[val_index])
        val_pr[val_index]=run_lgb(train_data,val_data,df_train[df_train_columns].loc[val_index],n_estimator,max_depth,num_leaves,lambda_l1,lambda_l2,min_split_gain,min_child_weight,bagging_fraction,early_stopping_rounds)
    return 1-np.sqrt(mean_squared_error(val_pr,df_train['target']))
    
from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(optim_lgb,{'n_estimator':(20,700),'max_depth':(-1,12),'num_leaves':(20,50),'lambda_l1':(0,5),'lambda_l2':(0,3),
'min_split_gain':(0.001, 0.1),'min_child_weight':(5,50),'bagging_fraction':(0.8,1),'early_stopping_rounds':(50,400)}
    
)
optimizer.maximize(init_points=3,
    n_iter=30)
print(optimizer.max)
