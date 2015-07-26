#!/usr/bin/python2.7

import datetime,time
import simplejson as json
import pickle
import numpy as np
from pyspark import SparkConf, SparkContext
import sys,os
import ConfigParser
import logging
from sklearn.metrics import roc_auc_score, roc_curve, auc

from pyspark.sql import *
sc = SparkContext()
sqlContext = SQLContext(sc)
# 's3n://kiip-hive/scripts/ctr/fitbox'
############################
time_format = "%Y%m%d_%H-%M-%S"
timeStampForLog = time.strftime( time_format, time.localtime())
logName = "log." + timeStampForLog
logging.basicConfig(level=logging.INFO,
              format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
              #datefmt='%a, %d %b %Y %H:%M:%S',
              filename=logName,
              filemode='a')
############################
def ConfRead(strConf):
    try:
        if not os.path.isfile(strConf):
            logging.info( "[ERROR] ConfRead: no such conf file %s"%strConf)
            sys.exit(1)
        cf = ConfigParser.ConfigParser()
        cf.read(strConf)
        return cf

    except Exception, e:
        logging.info( "Fail to read conf file %s: %s" % (strConf, e))
        sys.exit(1)

def getPara(paraName,section="PARA"):
    try:
        return cf.get(section,paraName)
    except Exception, e:
        logging.error("Fail to get para[%s]: %s" % (paraName, e))
        return None
        # sys.exit(1)

conf_path = sys.argv[1]
logging.info( conf_path )
cf = ConfRead(conf_path)

def get_quads(s):
    arr1 = s.strip().split(':')[0].split(',')
    arr2= s.strip().split(':')[1].split(',')
    return [map(lambda x: x.strip(),arr1) , map(lambda x: x.strip(),arr2)]   

def write_lines(fpath,elem):
    file_object = open(fpath, 'a+')
    file_object.write(str(elem) + " ")
    file_object.close()

################################
code_path = getPara("code_path")
sc.addPyFile(code_path+'/preprocess/FeatureManager.py')
sc.addPyFile(code_path+'/preprocess/warm_start.py')
sc.addPyFile(code_path +'/preprocess/__init__.py')
sc.addPyFile(code_path+'/optimize/olbfgs.py')
sc.addPyFile(code_path+'/optimize/__init__.py')
sc.addPyFile(code_path+'/trainer.py')
sc.addPyFile(code_path+'/__init__.py')

from FeatureManager import *
from olbfgs import *
from warm_start import set_first_intercept, set_first_intercept_spark




data_path = cf.get("PARA","data_path")

pos_sample_num = int(cf.get("PARA","pos_sample_num") )
neg_sample_num = int(cf.get("PARA","neg_sample_num") )
max_iter = int(getPara('max_iter')) 
min_iter = int(getPara('min_iter'))
grad_norm2_threshold = float(cf.get("PARA","grad_norm2_threshold"))
m_1 = float(cf.get("PARA","m_1"))
c_1 = float(cf.get("PARA","c_1"))
lamb_const_1 = float(cf.get("PARA","lamb_const_1"))
work_path = getPara('work_path')

#================================prepare data=========================================#
startTimeForData = datetime.datetime.now()

label = 'target'
# Noe specify columns,then load all columns. 
# parquetFile is DataFrame, Each elem is Row
if getPara("columns") is None:
    parquetFile = sqlContext.load(
        path=data_path,
        source="parquet",
        mergeSchema="false")    
else: # load specified columns
    columns = map(lambda x: x.strip(), 
        getPara("columns").strip().split(",") + [label] )
    logging.info("columns:%s"% columns)     
    parquetFile = sqlContext.load(
        path=data_path,
        source="parquet",
        mergeSchema="false") \
        .select(*columns)

pos_rows_all = parquetFile.filter(parquetFile.target == 1.0)
neg_rows_all = parquetFile.filter(parquetFile.target == 0.0)

if  pos_sample_num == -1:
    pos_rows = pos_rows_all
else:
    pos_rows = pos_rows_all.takeSample(False, pos_sample_num)

r_subsample = float(cf.get("PARA","r_subsample") )

if neg_sample_num == -1: #take all neg subsamples 
    neg_subsample_rows = neg_rows_all.sample(False, r_subsample, 42)
else: # take some samples from neg subsamples
    neg_subsample_rows = neg_rows_all.sample(False, r_subsample, 42).takeSample(False, neg_sample_num) 
    
resampled_rows = pos_rows.unionAll( neg_subsample_rows ).repartition(sc.defaultParallelism)

# resampled_json_rows = resampled_rows.toJSON().repartition(sc.defaultParallelism)
# resampled_dict_rows = resampled_json_rows.map(json.loads)

####### resampled_rows_reduced is DataFrame, each elem is Row
# resampled_rows_reduced = resampled_dict_rows.coalesce(sc.defaultParallelism)
resampled_rows_reduced = resampled_rows    #.coalesce(sc.defaultParallelism)
# resampled_rows_reduced.cache()

#============================== join svd start=====#
# df_joined is DataFrame[Row]
if getPara('svd').lower() == "1": 
    logging.info("join svd feature...")
    df = resampled_rows_reduced   #.toDF()
    df_svd = sqlContext.jsonFile ( getPara('svd_feat_json') )
    df_joined = df.join(df_svd, df.deviceID == df_svd.device_id, 'left_outer')
    # df_joined.show()
else:
    df_joined = resampled_rows_reduced
# print rdd_json.first()

#============================= join svd end=========#
# Convert DataFrame[Row] to RDD[dict]
def c2(x):
    if x['device_id'] is None:
        x['device_id']=''
    if x['BKMaxIndex'] is None:
        x['BKMaxIndex']=''       
    return x
rdd_json = df_joined.rdd.map(lambda x : x.asDict()).map(lambda x : c2(x))  # asDict() , convert Row to dict
            
#========================convert to matrix and split ==================#   
num_part = rdd_json.getNumPartitions()
logging.info("num_part= %s" % num_part)
def splitData(pIdx, rows):
    if pIdx < num_part * 0.7 :
        for x in rows:
            yield ("training",x)
    else:
        for x in rows: 
            yield ("validation",x)
# convert json to matrix using the hash_feats instance 
all_data = rdd_json.map(hash_feats.parse_row)
#split data to train,cv,test data according to parition index 
rddpair = all_data.mapPartitionsWithIndex(splitData)

train_data = rddpair.filter(lambda x: x[0]=='training').values() # matrix rdd
train_data.cache() 
cv_data = rddpair.filter(lambda x: x[0]=='validation').values()  # matrix rdd
cv_data.cache()
# X_verify = map(hash_feats.parse_row, []) #matrix 
X_verify = cv_data.collect()  # a list of matrix, not rdd
# get number of training data
batch_size = train_data.count()
# print the count or not accord to conf
if getPara('comp_count') == "0":
    pass
else:
    logging.info( "counts:pos_rows_all,neg_rows_all,all_data,tr_data,cv_data=%s %s %s %s %s" % \
        (pos_rows_all.count(),neg_rows_all.count(),all_data.count(), batch_size, cv_data.count()))
        
#================== old model ====================================#
old_model_path = cf.get("PARA",'old_model_path')
logging.info( "old_model_path="+str(old_model_path) )

w_g_l_old = pickle.load(open(old_model_path, 'rb'))
w_old = w_g_l_old[0]
w_tmp = set_first_intercept_spark(w_old.copy(), train_data)
w_init = sp.sparse.csc_matrix(w_tmp)

#=======================================================================================#
def trainOneFeatFile(feat_file):
    # get singe_features 
    t_lines = open(feat_path+'/'+feat_file,'rb').read().splitlines()
    single_features = map( lambda x: x.strip(),t_lines[0].split(',') )
    logging.info("single_features:%s" % single_features)
    #get quad features
    quad_arr = t_lines[1:]
    quads = map( get_quads, quad_arr ) 
    # create an instance
    hash_feats = HashFeatureManager() \
        .set_k(21) \
        .set_label('target') \
        .set_single_features(single_features) \
        .set_quad_features(quads)


    #================================train====================================#

    def trainAndVerify(l2_r,w_init,subfix):
        lr_l2_obj_func = make_spark_lr_l2_obj_func(train_data, l2_r)
        lr_l2_obj_func_grad = make_spark_lr_l2_gradient(train_data,l2_r)

        m_1 = 20
        c_1 = 1.
        lamb_const_1 = 1.
        # Result: w, gradient_estimates, losses, B_t 
        w_g_l_new = olbfgs_batch(w_init, lr_l2_obj_func, lr_l2_obj_func_grad, m_1, c_1, lamb_const_1,
                               batch_size=batch_size, grad_norm2_threshold=1.*10**-5, max_iter=max_iter)

        pickle.dump(w_g_l_new, open(cf.get("PARA","new_model_path") + "ctr_model.p" + subfix, 'wb'))

        w_new = w_g_l_new[0]

        tr_obj_func = make_spark_lr_l2_obj_func(train_data, 0.0)
        tr_loss = tr_obj_func(w_new)

        cv_obj_func = make_spark_lr_l2_obj_func(cv_data, 0.0)
        cv_loss = cv_obj_func(w_new)

        return (tr_loss,cv_loss,w_new)

    tr_losses = []
    cv_losses = []
    roc_arr = [] 
    l2_r_arr = [ float(x) for x in cf.get("PARA","l2_r").split(',') ]
    # times 2 until reach the maxR
    if getPara('maxR'): # set maxR means use this method to construct para list
        l2_r = l2_r_arr[0]
        l2_r_arr = []
        while l2_r <= float(getPara('maxR')):
            l2_r_arr.append(l2_r)
            l2_r = l2_r * 2.0
    logging.info("l2_r_arr:%s"% l2_r_arr)
    timeStampForRst = time.strftime( time_format, time.localtime())
    subfix = '.'+feat_file+'.'+timeStampForRst
    # iterate through l2_r_arr, try each l2_r
    for l2_r in l2_r_arr:
        logging.info("l2_r= %s"% l2_r)
        # train
        starttime = datetime.datetime.now()

        (tr_loss,cv_loss, w_new) = trainAndVerify(l2_r,w_init,subfix)

        endtime =  datetime.datetime.now()
        logging.info( "time for l2_r[%s] %s" % (l2_r, str(endtime - starttime)) )
        #verify
        verify_pred = map(
            lambda x: (x[0], lr_predict(w_new, x[1])),
            X_verify
        )
        roc = roc_auc_score(
            [x[0] for x in verify_pred],
            [x[1] for x in verify_pred])

        logging.info( "l2_r,roc,tr_loss,cv_loss=%s,%s,%s,%s" %(l2_r,roc,tr_loss,cv_loss) )

        tr_losses.append(tr_loss)
        cv_losses.append(cv_loss)
        roc_arr.append(roc)
        # store the losses and roc
        
        
        write_lines(work_path +'/tr_losses' + subfix, str(tr_loss) )
        write_lines(work_path +'/cv_losses' + subfix, str(cv_loss) )
        write_lines(work_path +'/roc_arr' + subfix, str(roc) )


    pickle.dump( tr_losses,open(work_path +'/tr_losses.p' + subfix, 'wb') )
    pickle.dump( cv_losses,open(work_path +'/cv_losses.p' + subfix, 'wb') )
    pickle.dump( roc_arr,open(work_path +'/roc_arr.p' + subfix, 'wb') )

        # write_lines(work_path +'/tr_losses' + subfix, str(tr_losses))
        # write_lines(work_path +'/cv_losses' + subfix, str(cv_losses))
        # write_lines(work_path +'/roc_arr' + subfix, str(roc_arr))
        # pickle.dump( verify_pred , open(work_path + '/verify_pred' + subfix,'wb'))
        # pickle.dump(tr_losses, open(work_path +'/tr_losses' + "." + str(l2_r) + timeStamp, 'wb') )
        # pickle.dump(cv_losses, open(work_path + '/cv_losses' + "." + str(l2_r) + timeStamp, 'wb') )
        # pickle.dump(cv_losses, open(work_path + '/roc_arr' + "." + str(l2_r) + timeStamp, 'wb') )
        #================================train====================================#
feat_path = getPara('feat_path')
# feat_files = getPara("feat_files").strip().split(',')

filesDealed=[]
while 1:
    files = os.listdir(feat_path)
    feat_files = filter(lambda x:x[0]!='.',files)
    if set(feat_files) == set(filesDealed) : continue
    for f in set(feat_files) - set(filesDealed):
        logging.info("find new file to deal with: %s " % f )
        try:
            trainOneFeatFile(f)
        except Exception, e:
            logging.info("Failed to deal with %s,%s" % (f,e) )
        logging.info( "filesDealed: %s " % filesDealed)
        if f not in filesDealed:  filesDealed.append(f)
        

"""
MASTER=yarn-client /home/hadoop/spark/bin/spark-submit --driver-memory 10G --executor-memory 10G --num-executors 128 --executor-cores 4 /home/hadoop/t_para_parq.py /home/hadoop/model.conf

"""

    # # log the time spent on preparing data
    # endTimeForData = datetime.datetime.now()
    # logging.info("time for preparing data:" + str(startTimeForData - endTimeForData) )

    #================================prepare data end=========================================#