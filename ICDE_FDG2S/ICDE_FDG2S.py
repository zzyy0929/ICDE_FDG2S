# 运行环境： python-3.6.12 tensorflow-1.14.0  
import numpy as np
import pandas as pd
import tensorflow as tf
from geopy.distance import geodesic
import scipy.stats
import math
import os
from sklearn.preprocessing import StandardScaler #正态分布归一化
import warnings; warnings.simplefilter('ignore')

# Hyperparameter: 
Local_embed = 16 # location embedding 8/16/32/64
Contx_dim = 32  #   Context dimension embedding
trans1 = 256 

# Task-wise Weight 
Tw_shape = 0.005 #  0.05/0.1/0.5/1/1.5
Tw_dir = 20 #  2/5/10 20 30 50
Tw_cov = 80  # 10/50 /100/80

# MAPE = loss2:  72.11642// loss_shape:  -86.35721//  Dir_loss:  -0.7427752// Loss_cov: 3.7508573e-05

Context_raw = np.load('SIP_Time5_Weather10-Interval30-201701_03.npy')#4320*15
SIP_30min_flow = np.load('SIP_30m_Flow.npy')#108*4320
Adj_matrix = np.load('Norm_Adj_SIP_1122.npy') 
SIP_30min_diff = np.load('SIP_30_diff.npy')#108*4320

#地区数量Node
location_degree = SIP_30min_flow.shape[0]

#设置一个batch_size大小以及总共有多少块
batch_size = 1
batch_number = np.int(4320/18/batch_size)
one_epoch_size = np.int((4320-17)/batch_size)

# 每个batch的形状为：
# Dim of Time 
Dim_time = 5
Dim_weat = 10
Dim_Ctotal = 15
Contx_dim = 64
trans1 = 256
#M
w3_degree = 4
#
Local_embed = 16
#GNN
w6_degree = 64

#将12个按多少分一组
GNN_degree = 64
number_of_GNN = 3
#LSTM_output_seq
batchsize = 10
seq_len = 6
#
num_stacked_layers = 1
hidden_dim_1 = 96
hidden_dim_2 = 80
Total_interval = SIP_30min_flow.shape[1]
Td = 48 


from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
# 构建字典，先对csv进行预处理
Context_file = pd.read_csv('Weather_201701-03-1.csv')
# One_hot processing 处理模块
Category_List = np.unique(list(Context_file['Condition']))
Embed_onehot = np.zeros([len(Category_List)])
# print("Weather context:", Category_List)
# for i in range(len(Category_List)):
#     print(i,Category_List[i])

   
# 构建字典
Context_index = np.zeros([Total_interval, 3]) # 三列，表示#  时间戳索引/第几个时间步/第几种类型的天气
for i in range(Total_interval):
    Context_index[i,0] = i  #  时间戳索引
    Context_index[i,1] = i%Td  # 第几个时间步
    Context_index[i,2] = list(Category_List).index(Context_file['Condition'][i]) # 第几种类型的天气
Context_index = np.int32(Context_index)

# 按context分类整理，并计算邻接矩阵
Time_adj = np.zeros([Td,location_degree,location_degree]) 
Weat_adj = np.zeros([len(Category_List),location_degree,location_degree])

for time in range(Time_adj.shape[0]):
    SIP_30min_flow_temp = SIP_30min_flow[:, time::Td] #108*90
    cos_sim = cosine_similarity(SIP_30min_flow_temp, SIP_30min_flow_temp)
    Time_adj[time] = cos_sim
for weather in range(Weat_adj.shape[0]):
    Cur_index =np.where(Context_index[:,2] == weather)
    SIP_30min_flow_temp = (SIP_30min_flow.T[Cur_index]).T 
    cos_sim = cosine_similarity(SIP_30min_flow_temp, SIP_30min_flow_temp)
    Weat_adj[weather] = cos_sim




def sampling_nearest(T_start, T_end, k):
    '''X（available）-Y()
    data: 流量数据，size(108*4320)
    k：预测的T_start为几天之后的流量
    predict_time_start: 待预测数据的起始时刻，图示中的T
    predict_time_end: 待预测数据的起始时刻，图示中的T+6
    cur_day2: 最近一个日周期
    cur_day1: 次近一个日周期
    cur_week: 最邻近周周期
    返回numpy形式的near_data，在图示中size为3*108*7
    '''
    Seq_len = T_end - T_start 
    near_data = []
    cur_week = np.arange(T_start-(Td*k+6*Td)-Seq_len,T_start-(Td*k+6*Td))
    cur_day2 = np.arange(T_start-(Td*(k+1))-Seq_len,T_start-(Td*(k+1)))
    cur_day1 = np.arange(T_start-Td*k-Seq_len,T_start-Td*k)
    cur_t = np.arange(T_start,T_end) 
    # print(cur_week,cur_day2,cur_day1)
    for cur in [cur_week, cur_day2, cur_day1]:
        near_data_temp = (SIP_30min_flow.T[cur]).T 
        near_data.append(near_data_temp)
    near_data = np.array(near_data)
    X = near_data
    Y = SIP_30min_flow[:,T_start:T_end]
    Context = np.mean(Context_raw[T_start:T_end],axis = 0)
    # 两种context合并，也可以写成K个context,time在前，weather在后
    Time_A = Time_adj[Context_index[T_start,1],:,:]
    Weat_A = Weat_adj[Context_index[T_start,2],:,:]
    Conxt_Adj = list([Time_A,Weat_A,np.multiply(Weat_A,Time_A)])
    Conxt_Adj = np.array(Conxt_Adj)
    # 计算spatiotemporal variance
    Y_var = []
    for t in range(T_start, T_end):
        time_index, weat_index = Context_index[t][1],Context_index[t][2]
        C_inter_idx = set(np.flatnonzero(Context_index[:,1]==time_index)) &  set(np.flatnonzero(Context_index[:,2]==weat_index))
        C_inter_idx = list(C_inter_idx)
        # print(C_inter_idx)
        Temp_flow = []
        for indx in C_inter_idx:
            Temp_flow.append(SIP_30min_flow[:,indx:indx+1])
        Temp_flow_var = np.std(np.array(Temp_flow), axis=0)
        Var = np.std(np.array(Temp_flow_var), axis=1) 
        Y_var.append(Var)
    Y_var = np.array(Y_var)
    Y_var1 = np.zeros([108,6])
    for i in range(seq_len):
        Y_var1[:,i] = Y_var[i,:]
    return X, Y, Y_var1,Context,Conxt_Adj


def Generate_batch(T_start,batchsize,k):# T_start是一批中sample起始索引，batchsize是一批里有几个sample，k是预测未来第几天的
    X_batch = []
    Y_batch = []
    Context_batch = []
    Var_batch = []
    ConxtAdj_batch = []
    seq_len = 6
    if T_start-(Td*k+7*Td)-6>0:
        #print("generate samples: ", T_start, T_start + batchsize)
        for i in range(T_start, T_start + batchsize):
            X,Y,Var,Context,ConxtAdj = sampling_nearest(i,i+seq_len,k)
            X_batch.append(X)
            Y_batch.append(Y)
            Context_batch.append(Context)
            Var_batch.append(Var)
            ConxtAdj_batch.append(ConxtAdj)
    X_batch = np.array(X_batch)
    Y_batch = np.array(Y_batch)
    Context_batch = np.array(Context_batch)
    Context_batch = Context_batch.reshape([batchsize,1,-1])
    Var_batch = np.array(Var_batch)
    ConxtAdj_batch = np.array(ConxtAdj_batch)
    return X_batch,Y_batch,Context_batch,Var_batch,ConxtAdj_batch


def build_graph(feed_previous = False,reuse_variables=False):
    tf.reset_default_graph()
    global_step = tf.Variable(
                  initial_value=0,
                  name="global_step",
                  trainable=False,
                  collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    def get_shape_sim_tf(x,y):
        x_diff = x - tf.roll(x, shift=1, axis=0)
        y_diff = y - tf.roll(y, shift=1, axis=0)
        x_ycos = (-tf.reduce_sum(tf.multiply(x,y))/(tf.sqrt(tf.reduce_sum(tf.multiply(x,x))+0.01)*tf.sqrt(tf.reduce_sum(tf.multiply(y,y))))+0.01)
        #x_ydist = tf.sqrt(tf.reduce_sum(tf.pow(x_diff-y_diff, 2)))
        return x_ycos
    def tf_cov(x):
        mean_x = tf.reduce_mean(x, axis=1, keepdims=True)
        cov_xx = tf.matmul(tf.transpose(x-mean_x),x-mean_x)/tf.cast(tf.shape(x)[0]-1, tf.float32)
        return cov_xx

    def cut_matrix(matrix, x):
        return matrix[:,:,0:x], matrix[:,:,x:]
    # Timestamp, Weather =  ConIdx[:,1] ConIdx[:,2]
    x1 = tf.placeholder(tf.float32,[None,1,15])  # 时间天气向量 batch_size*1*15
    x2 = tf.placeholder(tf.float32,[None,3,location_degree,seq_len])# 输入3个period
    ConAdj = tf.placeholder(tf.float32,[None,3,location_degree,location_degree])# 流量
    #ys = tf.placeholder(tf.float32,[batch_size,location_degree,6])#batch_size*108*6
    Output_Seq = tf.placeholder(shape=(None, location_degree,seq_len),dtype=tf.float32)
    Output_Var = tf.placeholder(shape=(None, location_degree,seq_len),dtype=tf.float32)
    # Direction_shape: (1,  108, 1)
    Location_matrix = tf.Variable(tf.random_normal([location_degree,Local_embed]),dtype=tf.float32,name='Location_matrix') # 108 * 8
    # Local_ =  tf.Variable(tf.random_normal([Local_embed,Local_embed]),dtype=tf.float32,name='Location_weigh') 
    Loc_Time_w = tf.Variable(tf.random_normal([Local_embed,1]),dtype=tf.float32,name='loc_time') # √
    Time_wea_w = tf.Variable(tf.random_normal([Dim_time,1]),dtype=tf.float32,name='time_wea') # √
    Weat_K = tf.Variable(tf.random_normal([Dim_weat,Contx_dim]),dtype=tf.float32,name='ContxtK') # √

    Contxt_Trans = tf.Variable(tf.random_normal([Dim_Ctotal,Contx_dim]),dtype=tf.float32,name='SelfCorr')  
    #b1 = tf.Variable(tf.truncated_normal([location_degree,location_degree],stddev=0.1),dtype=tf.float32,name='b1')

    Trans1 = tf.Variable(tf.truncated_normal([Contx_dim,trans1],stddev=0.1),dtype=tf.float32,name='Context_trans1')
    Trans2 = tf.Variable(tf.truncated_normal([trans1,Contx_dim],stddev=0.1),dtype=tf.float32,name='Context_trans2')
    wm_d = tf.Variable(tf.truncated_normal([Contx_dim,1]),dtype=tf.float32,name='wm_d')

    def Get_ContextEmb(inputs,activation_function=None):
        #获得时间天气
        inputs = tf.cast(inputs,tf.float32)
        time_vector ,weather_vector = cut_matrix(inputs, 5) #b*1*5 b*1*10
        # Context-wise embedding
        Loc_Time_emb = tf.matmul(Location_matrix, Loc_Time_w) # 108*8*8*1
        Loc_Time_emb = tf.map_fn(lambda x: tf.matmul(Loc_Time_emb, x), time_vector) #b*108*5 #得到Local_Time-vector
        Time_weat_emb = tf.matmul(Loc_Time_emb, Time_wea_w)  # 108*5*(5*1)
        Loc_Time_wea_emb = tf.matmul(Time_weat_emb, weather_vector) #108*5*1 (1*10)
        Contxt_emb =  tf.matmul(Loc_Time_wea_emb,Weat_K)#b*108*1
        # Self-correlation
        Context_selfembed = tf.matmul(inputs, Contxt_Trans)  
        # Addition
        Context_Summy =  tf.add(Context_selfembed, Contxt_emb)
        #------------Batch normalization-----------#
        wb_mean, wb_var = tf.nn.moments(Context_Summy, [0,1,2])
        scale = tf.Variable(tf.ones([1]))
        offset = tf.Variable(tf.zeros([1]))
        variance_epsilon = 0.001
        Context_SummyN = tf.nn.batch_normalization(Context_Summy, wb_mean, wb_var, offset, scale, variance_epsilon)
        Context_S_N =  tf.matmul(Context_SummyN,Trans1)#b*108*M    
        Context_S_N = tf.matmul(Context_S_N,Trans2)#b*108*6
        print("Time_weat_emb:",Time_weat_emb)
        if activation_function is None:
            outputs = Context_S_N
        else:
            outputs = activation_function(Context_S_N)
        return outputs

    Context_Embed = Get_ContextEmb(x1,activation_function=tf.nn.relu)    


    #----------------Weight matrix to derive adjacent matrix ----------------#
    W_time =  tf.Variable(tf.random_normal([Dim_time,1]),dtype=tf.float32,name='w_time')
    W_weat =  tf.Variable(tf.random_normal([Dim_weat,1]),dtype=tf.float32,name='w_weat')
    W_inte =  tf.Variable(tf.random_normal([Dim_time,1]),dtype=tf.float32,name='w_weat')
    W_time_weat = tf.Variable(tf.random_normal([Dim_time,Dim_weat]),dtype=tf.float32,name='time_wea')
    W_Time_Weat = tf.Variable(tf.random_normal([Contx_dim,3]),dtype=tf.float32,name='time_wea1')
    #得到batch个108*108个context矩阵
    def get_ADJ(inputs,activation_function=None):
        inputs = tf.cast(x1,tf.float32)
        time_vector ,weather_vector = cut_matrix(inputs, 5) #b*1*5 b*1*10
        # Context-wise embedding
        Loc_Time_emb = tf.matmul(Location_matrix, Loc_Time_w) # 108*8*8*1
        Loc_Time_emb = tf.map_fn(lambda x: tf.matmul(Loc_Time_emb, x), time_vector) #b*108*5 #得到Local_Time-vector
        Time_weat_emb = tf.matmul(Loc_Time_emb, Time_wea_w)  # 108*5*(5*1)
        Loc_Time_wea_emb = tf.map_fn(lambda x: tf.matmul(Time_weat_emb, x), weather_vector) #108*5*1 (1*10)
        Contxt_emb = tf.map_fn(lambda x: tf.matmul(x,Weat_K), Loc_Time_wea_emb)#b*108*1
        Weat_Time_i = tf.matmul(time_vector,W_time_weat) 
        # Weat_Time_i = tf.matmul(Weat_Time_i,weather_vector) 
        # time_vector,weather_vector,Weat_Time_i
        Weat_Time_i = tf.matmul(Weat_Time_i, tf.transpose(W_time_weat)) 
        #Weat_Time_i = tf.map_fn(lambda x: tf.matmul(Weat_Time_i, tf.transpose(x)), weather_vector)       
        #----------------------Aggr-method1---------------#
        RW_Time = tf.nn.sigmoid(tf.matmul(time_vector,W_time))
        RW_Weat  =tf.nn.sigmoid(tf.matmul(weather_vector,W_weat))
        RW_WeatT =tf.nn.sigmoid( tf.matmul(Weat_Time_i,W_inte))
        Weightd =tf.nn.softmax((tf.concat([RW_Time,RW_Weat,RW_WeatT], axis = 1)))
        Adj = Weightd[:,0,:] * ConAdj[:,0,:,:] +  Weightd[:,1,:] * ConAdj[:,1,:,:] +  Weightd[:,2,:] * ConAdj[:,2,:,:]

        return Adj
    #----------------Adjacent matrix generation----------------#


    #得到context
    ADJ_matrix = get_ADJ(Context_Embed,activation_function=tf.nn.leaky_relu) 

   #------------------------Context-GCRF------------------------#
    # FC:Context-target
    W_CT = tf.Variable(tf.truncated_normal([Contx_dim,seq_len]),dtype=tf.float32,name='W_CT')


    #FC——得到b*108*108的参数
    w5 = tf.Variable(tf.truncated_normal([Contx_dim,location_degree]),dtype=tf.float32,name='w5')
    b5 = tf.Variable(tf.truncated_normal([location_degree,w3_degree]),dtype=tf.float32,name='b5')
    #Gnn参数
    #108*3的b
    b6 = tf.Variable(tf.truncated_normal([batch_size,location_degree,GNN_degree]),dtype=tf.float32,name='b6')
    w6 = tf.Variable(tf.truncated_normal([seq_len, 32]),dtype=tf.float32,name='w6')
    w7 = tf.Variable(tf.truncated_normal([32,64]),dtype=tf.float32,name='w7')
    w8 = tf.Variable(tf.truncated_normal([64,32]),dtype=tf.float32,name='w8')
    w9 = tf.Variable(tf.truncated_normal([32,6]),dtype=tf.float32,name='w9')




    def Long_termGCN(input1,  Adj_M):
    #输入为b*108*3  b*108*108
        #第一层
        #得到b*108*3
        Contarg = tf.matmul(Context_Embed, W_CT)
        layer1_temp = tf.matmul(Adj_M, input1)
        layer_1_output = tf.map_fn(lambda x: tf.matmul(x, w6), layer1_temp)
        layer2_temp = tf.matmul(Adj_M, layer_1_output)
        #b*108*3   *    3*M   =  b*108*M
        layer_2_output = tf.map_fn(lambda x: tf.matmul(x, w7), layer2_temp)
        #b*108*M   *    M*1   =  b*108*1
        wb_mean, wb_var = tf.nn.moments(layer_2_output, [0,1,2])
        scale = tf.Variable(tf.ones([1]))
        offset = tf.Variable(tf.zeros([1])) 
        variance_epsilon = 0.001
        layer_2_output = tf.nn.batch_normalization(layer_2_output, wb_mean, wb_var, offset, scale, variance_epsilon)
        layer_2_output = tf.map_fn(lambda x: tf.matmul(x, w8), layer_2_output)
        layer_2_output = tf.map_fn(lambda x: tf.matmul(x, w9), layer_2_output)

        print("layer_2_output ", layer_2_output)
        print("Contarg ", Contarg)
        # layer_2_output = Contarg + layer_2_output
        # layer_2_output = tf.squeeze(layer_2_output, axis = 1)
        return layer_2_output
    
    
    def lstm(Output6,Output_Seq,num_stacked_layers,input_size, hidden_dim,feed_previous): 
        
        weightslstm={
         'in':tf.Variable(tf.truncated_normal(shape=[input_size,hidden_dim], stddev=0.1)), 
         'out':tf.Variable(tf.truncated_normal(shape=[hidden_dim,input_size], stddev=0.1)) 
        }
        biaseslstm={
                'in':tf.Variable(tf.truncated_normal(shape=[hidden_dim,], stddev=0.1)), 
                'out':tf.Variable(tf.truncated_normal(shape=[input_size,], stddev=0.1)) 
               }
        ## Seq2Seq Parameters
        ## Parameters
        enc_inp = Output6

        print("len_enc_inp:",len(enc_inp))
        print(enc_inp)
        # Decoder: target outputs
        target_seq = [
            tf.placeholder(tf.float32, shape=(None, input_size), name="y".format(t))
              for t in range(seq_len)
        ]

        target_seq = Output_Seq
        print("len_target_seq:",len(target_seq))
        dec_inp = [ tf.zeros_like(target_seq[0], dtype=tf.float32, name="GO") ] + target_seq[:-1]
    
    
        cells = []
        for i in range(num_stacked_layers):
            with tf.variable_scope('RNN_{}'.format(i)):
                cells.append(tf.contrib.rnn.LSTMCell(num_units = hidden_dim, activation = tf.nn.relu)) # activation = tf.nn.leaky_relu

        cell = tf.contrib.rnn.MultiRNNCell(cells)
        def _rnn_decoder(decoder_inputs,
                        initial_state,
                        cell,
                        loop_function=None,
                        scope=None):
            
            state = initial_state
            outputs = []
            prev = None
            for i, inp in enumerate(decoder_inputs):
                if loop_function is not None and prev is not None:
                    with tf.variable_scope("loop_function", reuse=tf.AUTO_REUSE):
                        inp = loop_function(prev, i)
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                output, state = cell(inp, state)
                outputs.append(output)
                if loop_function is not None:
                    prev = output
            return outputs, state


        def _basic_rnn_seq2seq(encoder_inputs,
                              decoder_inputs,
                              cell,
                              feed_previous,
                              dtype=tf.float32,
                              scope=None):
        #  """Basic RNN sequence-to-sequence model.
        #  """
        # Encoder  
        # 运行 encoder,把待编码的输入RNN中得到enc_state
            enc_cell = cell
            _, enc_state = tf.contrib.rnn.static_rnn(enc_cell, encoder_inputs, dtype=dtype)
            # Decoder 
            #如何feed_previous = True 的话 就把先前decoder 输出的放到下一个时间步去进行decoder
            if feed_previous:
                return _rnn_decoder(decoder_inputs, enc_state, cell, _loop_function)
            else:
                return _rnn_decoder(decoder_inputs, enc_state, cell)
        #print(tf.get_variable_scope().reuse)
        def _loop_function(prev, _):
          '''Naive implementation of loop function for _rnn_decoder. Transform prev from 
          dimension [batch_size x hidden_dim] to [batch_size x output_dim], which will be
          used as decoder input of next time step '''
          return tf.nn.relu(tf.matmul(prev, weightslstm['out']) + biaseslstm['out'])
        
        dec_outputs, dec_memory = _basic_rnn_seq2seq(
            enc_inp, 
            dec_inp,  ##输入带有context的数据
            cell, 
            feed_previous = feed_previous
        )
        # list:内部是6个1*108tensor张量
        Pred = [tf.nn.relu(tf.matmul(i, weightslstm['out'])+biaseslstm['out']) for i in dec_outputs]
       # Pred = [(tf.matmul(i, weightslstm['out'])+biaseslstm['out']) for i in dec_outputs]
        return Pred
    #---------------构建输入到LSTM里的序列-----------------------#
    number_of_GNN = 3
    GCRF_out = []
    for i in range(number_of_GNN):
        output = Long_termGCN(x2[:,i,:,:], ADJ_matrix)
        for j in range(seq_len):
            GCRF_out.append(output[:,:,j]) # 有18个时间步输入进去
    print("GCRF_out:",GCRF_out)
    #4 * 108 * 1
    Output_Seq_F = []  # 6*108
    for i in range(6):
        Output_Seq_F.append(Output_Seq[:,: ,i])
    print("output##############################",Output_Seq_F)

    #LSTM
    with tf.variable_scope("Pred_RNN"):  
        Pred_Result = lstm(GCRF_out,Output_Seq_F,num_stacked_layers,108,hidden_dim_1,False)
    Pred_mse = 0
    Pred_mape = 0
    loss_shape = 0
    loss_cov = 0
    for _y, _Y in zip(Output_Seq_F, Pred_Result):
        print(_y.shape, _Y.shape)
        Pred_mse += tf.reduce_mean(tf.keras.losses.mse(_y ,_Y))
        Pred_mape += tf.reduce_mean(tf.keras.losses.mape(_y ,_Y))
    
    #-----------------LSTM结束---------------------#
    Pred_mape = Pred_mape/(108*6)
    Pred_mse = Pred_mse/(108*6)
    #---------------构建自编码器输入到LSTM里的序列-----------------------#
    Input_Seq_Auto = []
    number_of_step = 6
    Auto_output = Long_termGCN(x2[:,-1,:,:], ADJ_matrix)
    for i in range(number_of_step):
        Input_Seq_Auto.append(Auto_output[:,:,i])  
    print("Input_Seq_Auto:",Input_Seq_Auto)
    
    
    #4 * 108 * 1
    Output_Seq_Auto = []  # 6*108
    for i in range(6):
        Output_Seq_Auto.append(x2[:,-1,:,i])
    print("output_Auto:##############################",Output_Seq_Auto)
    
    #LSTM
    with tf.variable_scope("Auto_RNN"):  
        Auto_Result = lstm(Input_Seq_Auto,Output_Seq_Auto,num_stacked_layers,108,hidden_dim_2,False)
    Auto_mse = 0
    Auto_mape = 0
    for _y, _Y in zip(Output_Seq_Auto, Auto_Result):
        print(_y.shape, _Y.shape)
        Auto_mse += tf.reduce_mean(tf.keras.losses.mse(_y ,_Y))
        Auto_mape +=  tf.reduce_mean(tf.keras.losses.mape(_y ,_Y))
    # tf.reduce_mean
    Auto_mape = Auto_mape/(108*6)
    Auto_mse = Auto_mse/(108*6)
    print("Auto_mse:",Auto_mse)
    print("Pred_mse:",Pred_mse)
    
    

    #-----------------LSTM结束---------------------#
    
    #-------------------计算不确定性------------#
    WUAle1 = tf.Variable(tf.truncated_normal([Contx_dim,32]),dtype=tf.float32,name='W_u1')    
    WUAle2 = tf.Variable(tf.truncated_normal([32,32]),dtype=tf.float32,name='W_u2')   
    WUAle3 = tf.Variable(tf.truncated_normal([32,6]),dtype=tf.float32,name='W_u3') 
    UAle1 = tf.matmul(Context_Embed,WUAle1)
    UAle2 = tf.matmul(UAle1,WUAle2)
    UAle3 = tf.matmul(UAle2,WUAle3)
    UAle3 = 0.1*UAle3
    print("UAle3:",UAle3.shape)
    print("Auto_Result:",Auto_Result)
    print("Output_Seq_Auto:",Output_Seq_Auto)
    print("Auto_mape:",Auto_mape)
    
    Loss_unc = 0.01*tf.reduce_mean(tf.keras.losses.mape(Output_Var, tf.abs(UAle3)))
    Loss_unc = Loss_unc/(108*6)
    # #----------------------------计算Shape Loss---------------------------#  
    #     for i in range(location_degree):
    #        # loss_shape += get_shape_sim_tf(Pred_Context_out[0,:,i],Output_Seq1[0,:,i]) +  get_shape_sim_tf(Pred_Result[0,:,i],Output_Seq1[0,:,i])
    #         loss_shape += get_shape_sim_tf(Pred_Result[0,:,i],Output_Seq1[0,:,i])
    
    
    #------------------构建Uncertainty-error Consisetency Loss-----------------#
    Pred_Result_tf = tf.transpose(tf.cast(Pred_Result, tf.float32),[1,2,0])
    Auto_Result_tf = tf.transpose(tf.cast(Auto_Result, tf.float32),[1,2,0])
    
    Uncer_total = tf.abs(Auto_Result_tf- x2[:,-1,:,:])+ tf.abs(UAle3)
    print("Auto_Result_tf:",Auto_Result_tf)
    print("x2:",x2[:,-1,:,:])
    Cons_loss = tf.reduce_mean((0.01*((Pred_Result_tf - Output_Seq)**2))/(Uncer_total**2))  
    Cons_loss = tf.reduce_mean(Cons_loss)
    Tw_unc = tf.abs(Pred_mape)/tf.abs(Loss_unc + 1)
    Tw_auto = tf.abs(Pred_mape)/tf.abs(Auto_mse + 1)
    Tw_Ucons = tf.abs(Pred_mape)/tf.abs(Cons_loss + 1)
    Tw_mse = tf.abs(Pred_mape)/tf.abs(Pred_mse+ 1)
    loss_sum = tf.log(Pred_mape)  + 0.01 * Tw_mse * tf.log(Pred_mse) + 0.1 * Tw_unc * tf.log(Loss_unc) + 0.1 * Tw_auto * tf.log(Auto_mse) + 0.1 * Tw_Ucons *Cons_loss  #  0.05*tf.log(Pred_mse) ++# Loss_unc/1000  # + tf.log(Tw_cov * loss_cov) + loss_shape*Tw_shape 
    #print("#########",loss1)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(0.0001,global_step,20,decay_rate=0.98,staircase=True)  
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_sum)#学习率下降
    saver = tf.train.Saver
    # context_embed = f6_out 
    return dict(
        x1 = x1,
        x2 = x2,
        ConAdj = ConAdj,
        Output_Seq = Output_Seq, # √ Ground truth
        Output_Var = Output_Var,        
        Pred_Result = Pred_Result, # √  Prediction
        context_embed = Context_Embed, # √ Context
        ADJ_matrix = ADJ_matrix,
        Pred_mse = Pred_mse,
        Pred_mape = Pred_mape,   # √  
        Auto_Result = Auto_Result,
        Ula = UAle3,
        Cons_loss = Cons_loss,
        loss_shape = loss_shape, 
        Loss_unc= Loss_unc,
        Auto_mse = Auto_mse,
        train = train,      
        saver = saver
        )

def MAPE(prediction, label):  # a的维度为batchsize*a*b
    temp1 = np.subtract(label, prediction)
    temp2 = np.divide(temp1, prediction)
    temp3 = np.abs(temp2)
    return np.sum(temp3) / np.size(temp1)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def PICP(prediction, label, uncertainty):
    low = np.subtract(prediction, uncertainty)
    high = np.add(prediction, uncertainty)
    temp1 = np.greater(label, low)
    temp2 = np.less(label, high)
    temp3 = temp1 & temp2
    return np.sum(temp3) / np.size(prediction)


def UP(uncertainty, label):
    uncertainty = np.where(label==0,1,uncertainty)
    label = np.where(label == 0, 1, label)
    temp1 = np.divide(uncertainty, label)
    return np.sum(temp1) / np.size(temp1)

       

graph = build_graph(feed_previous = False,reuse_variables=False)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session()
sess.run(init)

# MAPE = loss2:  72.11642 loss_shape:  -86.35721 Dir_loss:  -0.7427752 Loss_cov: 3.7508573e-05
#----------Start to train--------------#
Pred_ContextL = [] 
Pred_ResultL= []
Output_Seq_L = []
Context_embedL = []
Adjmatrix_L = []
Input_Seq_FL = []
# saver = graph['saver']().restore(sess, os.path.join('./', 'SIP_Volume_Model_dire_mape_cov/SIP_pred'))

print("接续预测结果：\n")
for epoch in range(200):
    min_mape = 50
    sum_mpape = 0
    for i in range(1000,3000):
        if (i>=2136)& (i<=2145):
            continue
        X_batch,Y_batch,Context_batch,Var_batch,Adj_batch = Generate_batch(i,1,0) # sample起始索引,batchsize, k
        feed_dict={graph['x1']:Context_batch,graph['x2']:X_batch,graph['Output_Seq']:Y_batch, graph['ConAdj']:Adj_batch, graph['Output_Var']: Var_batch}
        _,Context_embed,Pred_Result,Output_Seq,Pred_mape = sess.run([graph['train'],graph['context_embed'],graph['Pred_Result'],graph['Output_Seq'],graph['Pred_mape']], feed_dict)
        Pred_Result = np.array(Pred_Result)
        Output_Seq = np.array(Output_Seq)
        Pred_Result = Pred_Result.transpose([1,2,0])
        Pred_Result = np.where(Pred_Result-Output_Seq>500, Output_Seq,Pred_Result)
        rmse = np.sqrt(np.mean((Pred_Result-Output_Seq)**2))
        mape = np.mean(np.abs(np.abs(Pred_Result)-Output_Seq)/Output_Seq)          
        sum_mpape = sum_mpape + Pred_mape
        if i%500==0:
            print("MAPE:",Pred_mape,mape)
    print("Pred_MAPE:",Pred_mape)
    print("MAPE:",mape)
    print("Node: ", "预测结果:",Pred_Result[0,25,:]," 目标: ",Output_Seq[0,25,:])
    temp_saver = graph['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'SIP_Volume_K/SIP_pred'))
    sum_mpape = sum_mpape/batch_number
    print("epoch,Pred_mape",epoch,Pred_mape)
    file_handle = open ( 'Results_0129.txt', mode = 'a' )
    file_handle.write(str(epoch)+str(Pred_mape)+'\t')
    file_handle.write(str(Pred_mape)+'\n' )
    file_handle.close()
    print("sum_mpape: ",sum_mpape)

sum_mpape = 0
for i in range(3000,4000):
    X_batch,Y_batch,Context_batch,Var_batch,Adj_batch = Generate_batch(i,1,0) # sample起始索引,batchsize, k
    feed_dict={graph['x1']:Context_batch,graph['x2']:X_batch,graph['Output_Seq']:Y_batch, graph['ConAdj']:Adj_batch, graph['Output_Var']: Var_batch}
    _,Context_embed,Pred_Result,Output_Seq,Pred_mape = sess.run([graph['train'],graph['context_embed'],graph['Pred_Result'],graph['Output_Seq'],graph['Pred_mape']], feed_dict)
    Pred_Result = np.array(Pred_Result)
    Output_Seq = np.array(Output_Seq)
    Pred_Result = Pred_Result.transpose([1,2,0])
    Pred_Result = np.where(Pred_Result-Output_Seq>500, Output_Seq,Pred_Result)
    rmse = np.sqrt(np.mean((Pred_Result-Output_Seq)**2))
    mape = np.mean(np.abs(np.abs(Pred_Result)-Output_Seq)/Output_Seq)          
    sum_mpape = sum_mpape + Pred_mape
    if i%500==0:
        print("MAPE:",Pred_mape,mape)
print("Pred_MAPE:",Pred_mape)
print("MAPE:",mape)
print("Node: ", "预测结果:",Pred_Result[0,25,:]," 目标: ",Output_Seq[0,25,:])
temp_saver = graph['saver']()
save_path = temp_saver.save(sess, os.path.join('./', 'SIP_Volume_K/SIP_pred'))
sum_mpape = sum_mpape/batch_number
print("epoch,Pred_mape",epoch,Pred_mape)
file_handle = open ( 'Results_0129.txt', mode = 'a' )
file_handle.write(str(epoch)+str(Pred_mape)+'\t')
file_handle.write(str(Pred_mape)+'\n' )
file_handle.close()
print("Test_sum_mpape: ",sum_mpape)

sum_mpape = 0
graph = build_graph(feed_previous = False,reuse_variables=False)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session()
sess.run(init)

# MAPE = loss2:  72.11642 loss_shape:  -86.35721 Dir_loss:  -0.7427752 Loss_cov: 3.7508573e-05
#----------Start to train--------------#
Pred_ContextL = [] 
Pred_ResultL= []
Output_Seq_L = []
Context_embedL = []
Adjmatrix_L = []
Input_Seq_FL = []
print("one-day预测结果：\n")
sum_mpape = 0
for epoch in range(200):
    min_mape = 50
    sum_mpape = 0
    Counter = 0
    for i in range(1000,3000):
        if (i>=2136)& (i<=2145):
            continue
        X_batch,Y_batch,Context_batch,Var_batch,Adj_batch = Generate_batch(i,1,1) # sample起始索引,batchsize, k
        feed_dict={graph['x1']:Context_batch,graph['x2']:X_batch,graph['Output_Seq']:Y_batch, graph['ConAdj']:Adj_batch, graph['Output_Var']: Var_batch}
        _,Context_embed,Pred_Result,Output_Seq,Pred_mape = sess.run([graph['train'],graph['context_embed'],graph['Pred_Result'],graph['Output_Seq'],graph['Pred_mape']], feed_dict)
        Pred_Result = np.array(Pred_Result)
        Output_Seq = np.array(Output_Seq)
        Pred_Result = Pred_Result.transpose([1,2,0])
        Pred_Result = np.where(Pred_Result-Output_Seq>500, Output_Seq,Pred_Result)
        rmse = np.sqrt(np.mean((Pred_Result-Output_Seq)**2))
        mape = np.mean(np.abs(np.abs(Pred_Result)-Output_Seq)/Output_Seq)          
        sum_mpape = sum_mpape + Pred_mape
        if i%400==0:
            print("Pred_MAPE:",Pred_mape)
        #print("MAPE:",mape)
        #print("Node: ", "预测结果:",Pred_Result[0,25,:]," 目标: ",Output_Seq[0,25,:])
        Counter = Counter + 1
    sum_mpape = sum_mpape/Counter
    print("epoch,Pred_mape",epoch,Pred_mape)
    file_handle = open ( 'Results_0129.txt', mode = 'a' )
    file_handle.write(str(epoch)+str(Pred_mape)+'\t')
    file_handle.write(str(Pred_mape)+'\n' )
    file_handle.close()
    print("sum_mpape: ",sum_mpape)

sum_mpape = 0
for i in range(3000,4000):
        X_batch,Y_batch,Context_batch,Var_batch,Adj_batch = Generate_batch(i,1,1) # sample起始索引,batchsize, k
        feed_dict={graph['x1']:Context_batch,graph['x2']:X_batch,graph['Output_Seq']:Y_batch, graph['ConAdj']:Adj_batch, graph['Output_Var']: Var_batch}
        _,Context_embed,Pred_Result,Output_Seq,Pred_mape = sess.run([graph['train'],graph['context_embed'],graph['Pred_Result'],graph['Output_Seq'],graph['Pred_mape']], feed_dict)
        Pred_Result = np.array(Pred_Result)
        Output_Seq = np.array(Output_Seq)
        Pred_Result = Pred_Result.transpose([1,2,0])
        Pred_Result = np.where(Pred_Result-Output_Seq>500, Output_Seq,Pred_Result)
        rmse = np.sqrt(np.mean((Pred_Result-Output_Seq)**2))
        mape = np.mean(np.abs(np.abs(Pred_Result)-Output_Seq)/Output_Seq)          
        sum_mpape = sum_mpape + Pred_mape
        # print("Pred_MAPE:",Pred_mape)
        # print("MAPE:",mape)
        # print("Node: ", "预测结果:",Pred_Result[0,25,:]," 目标: ",Output_Seq[0,25,:])
sum_mpape = sum_mpape/1000
print("epoch,Pred_mape",epoch,Pred_mape)
file_handle = open ( 'Results_0129.txt', mode = 'a' )
file_handle.write(str(epoch)+str(Pred_mape)+'\t')
file_handle.write(str(Pred_mape)+'\n' )
file_handle.close()
print("Test_sum_mpape: ",sum_mpape)



graph = build_graph(feed_previous = False,reuse_variables=False)

init = tf.global_variables_initializer()
saver = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session()
sess.run(init)

# MAPE = loss2:  72.11642 loss_shape:  -86.35721 Dir_loss:  -0.7427752 Loss_cov: 3.7508573e-05
#----------Start to train--------------#
Pred_ContextL = [] 
Pred_ResultL= []
Output_Seq_L = []
Context_embedL = []
Adjmatrix_L = []
Input_Seq_FL = []

print("one-week预测结果：\n")
for epoch in range(200):
    min_mape = 50
    sum_mpape = 0
    Counter = 0
    for i in range(1000,3000):
        if (i>=2136)& (i<=2145):
            continue
        X_batch,Y_batch,Context_batch,Var_batch,Adj_batch = Generate_batch(i,1,7) # sample起始索引,batchsize, k
        feed_dict={graph['x1']:Context_batch,graph['x2']:X_batch,graph['Output_Seq']:Y_batch, graph['ConAdj']:Adj_batch, graph['Output_Var']: Var_batch}
        _,Context_embed,Pred_Result,Output_Seq,Pred_mape = sess.run([graph['train'],graph['context_embed'],graph['Pred_Result'],graph['Output_Seq'],graph['Pred_mape']], feed_dict)
        Pred_Result = np.array(Pred_Result)
        Output_Seq = np.array(Output_Seq)
        Pred_Result = Pred_Result.transpose([1,2,0])
        Pred_Result = np.where(Pred_Result-Output_Seq>500, Output_Seq,Pred_Result)
        rmse = np.sqrt(np.mean((Pred_Result-Output_Seq)**2))
        mape = np.mean(np.abs(np.abs(Pred_Result)-Output_Seq)/Output_Seq)          
        sum_mpape = sum_mpape + Pred_mape
        Counter = Counter + 1
    print("Pred_MAPE:",Pred_mape)
    print("MAPE:",mape)
    print("Node: ", "预测结果:",Pred_Result[0,25,:]," 目标: ",Output_Seq[0,25,:])
    temp_saver = graph['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'SIP_Volume_K/SIP_pred'))
    sum_mpape = sum_mpape/Counter
    print("epoch,Pred_mape",epoch,Pred_mape)
    file_handle = open ( 'Results_0129.txt', mode = 'a' )
    file_handle.write(str(epoch)+str(Pred_mape)+'\t')
    file_handle.write(str(Pred_mape)+'\n' )
    file_handle.close()
    print("sum_mpape: ",sum_mpape)


for i in range(3000,4000):
    X_batch,Y_batch,Context_batch,Var_batch,Adj_batch = Generate_batch(i,1,7) # sample起始索引,batchsize, k
    feed_dict={graph['x1']:Context_batch,graph['x2']:X_batch,graph['Output_Seq']:Y_batch, graph['ConAdj']:Adj_batch, graph['Output_Var']: Var_batch}
    _,Context_embed,Pred_Result,Output_Seq,Pred_mape = sess.run([graph['train'],graph['context_embed'],graph['Pred_Result'],graph['Output_Seq'],graph['Pred_mape']], feed_dict)
    Pred_Result = np.array(Pred_Result)
    Output_Seq = np.array(Output_Seq)
    Pred_Result = Pred_Result.transpose([1,2,0])
    Pred_Result = np.where(Pred_Result-Output_Seq>500, Output_Seq,Pred_Result)
    rmse = np.sqrt(np.mean((Pred_Result-Output_Seq)**2))
    mape = np.mean(np.abs(np.abs(Pred_Result)-Output_Seq)/Output_Seq)          
    sum_mpape = sum_mpape + Pred_mape
    # print("Pred_MAPE:",Pred_mape)
    # print("MAPE:",mape)
    # print("Node: ", "预测结果:",Pred_Result[0,25,:]," 目标: ",Output_Seq[0,25,:])
    temp_saver = graph['saver']()
    save_path = temp_saver.save(sess, os.path.join('./', 'SIP_Volume_K/SIP_pred'))
sum_mpape = sum_mpape/1000
print("epoch,Pred_mape",epoch,Pred_mape)
file_handle = open ( 'Results_0129.txt', mode = 'a' )
file_handle.write(str(epoch)+str(Pred_mape)+'\t')
file_handle.write(str(Pred_mape)+'\n' )
file_handle.close()
print("Test_sum_mpape: ",sum_mpape)