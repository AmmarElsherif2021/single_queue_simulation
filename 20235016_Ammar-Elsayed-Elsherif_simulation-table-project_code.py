# -*- coding: utf-8 -*-
'''
-student name: Ammar Elsayed Elsherif

-Assignment: Simulation of a grocery store M/M/1 Queue





'''
#libraries used 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

def generate_ts(seed):
    #variables

    max_inter_arr = 8
    min_inter_arr = 1

    max_service_time= 6
    min_service_time= 1

    n_events = 200

    #initial state 
    np.random.seed(seed)
    event = 0
    time_ = 0
    #counters for arrived and served customers
    arrived_customers = 0
    served_customers = 0
    departed_customers = 0

    #generate random variables for next events
    interarrival_time = np.random.uniform(min_inter_arr , max_inter_arr)
    next_arrival_time = time_ + interarrival_time
    server_status = 'idle'
    queue = 0
    arrived_customers += 1

    ''' -----------------------End of event 0 ----------------------------
    -----------------------------------------------------------------------'''

    event += 1  
    #create timeseries and populate with event 1 details
    ts_columns = ['event', 'time', 'type', 
                  'queue', 'arr cust', 'served cust', 'depar cust']

    time_series =  pd.DataFrame([[1, float(next_arrival_time), 'arrival', 
                                  queue, arrived_customers, 0, 0]],
                                 columns = ts_columns) 


    while event <= n_events:
        #event starts
        #parameters at event t
        event_type = time_series['type'].iloc[event-1]
        time_ = time_series['time'].iloc[event-1]
        
        
        
        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  if event = arrival !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        if event_type == 'arrival':
           
            #counter of arrived customers increases by 1
            arrived_customers += 1
            
            #generate next arrival time
            interarrival_time = np.random.uniform(min_inter_arr, max_inter_arr)
            next_arrival_time = time_ + interarrival_time  

            #if server status is idle customer is served immediatly 
            #and generates service time
            if server_status == 'idle':
                #customer is served and counter of served customer increases by 1
                served_customers += 1
                #this customer number is added to the 'served customer' column at event n
                time_series['served cust'].iloc[event-1] =  served_customers

                #generate next events (service and departure time)
                service_time = np.random.uniform(min_service_time, max_service_time)
                departure_time = time_ + service_time
                departed_customers += 1 

                #add generated events to existing time series
                generated_events =  pd.DataFrame([
                              [event, float(departure_time), 'departure', 0, 0,0,  departed_customers],
                              [event, float(next_arrival_time), 'arrival', 0, arrived_customers, 0, 0]
                              ], columns = ts_columns) 
                              #Order doesnt matter because it's sorted next

                time_series =  pd.concat([time_series, generated_events])
                #events are sorted by time 
                time_series = time_series.sort_values(['time'])
                time_series.reset_index(drop=True, inplace=True)
                #event number is assigned by time order
                time_series['event'] = list(range(1, time_series.shape[0]+1))

                #event is finished and event counter increases
                event += 1 

            #if server status is busy increase queue and only generates arrival activity
            if server_status == 'busy':
                queue += 1
                #add generated events to existing time series
                generated_events =  pd.DataFrame([
                                    [event, float(next_arrival_time), 'arrival', 
                                     0, arrived_customers,0, 0]]
                                    , columns = ts_columns) 

                time_series =  pd.concat([time_series, generated_events])
                time_series = time_series.sort_values(['time'])
                time_series.reset_index(drop=True, inplace=True)
                time_series['event'] = list(range(1, time_series.shape[0]+1))
                time_series['queue'].iloc[event-1] = queue
                #event is finished and event counter increases
                event += 1 

        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  if event = arrival !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        if event_type == 'departure':
            
            #if queue is zero and customer departs, server status remains idle and next event is an arrival
            if queue == 0 :
                server_status = 'idle'
                #event is finished and event counter increases
                #nothing else happens untill next arrival
                event += 1

            #if there are customers in queue (>0), server changes to busy and queue decreases by one   
            if queue != 0 :
                #customer is served and counter of served customer increases by 1
                served_customers += 1
                #this customer number is added to the 'served customer' column at event n
                time_series['served cust'].iloc[event-1] =  served_customers           
                
                #queue decreases by one
                queue -= 1
                server_status = 'busy'
                
                #generate next events (service and departure time)
                service_time = np.random.uniform(min_service_time, max_service_time)
                departure_time = time_ + service_time
                departed_customers += 1 #same customer that is served at arrival time departs are departure time

                #add generated events to existing time series
                generated_events =  pd.DataFrame([
                                        [event, float(departure_time), 'departure', 0, 0, 0, departed_customers]
                                        ], columns = ts_columns) 

                time_series =  pd.concat([time_series, generated_events])
                time_series = time_series.sort_values(['time'])
                time_series.reset_index(drop=True, inplace=True)
                time_series['event'] = list(range(1, time_series.shape[0]+1)) 
                time_series['queue'].iloc[event-1] = queue

                #event is finished and event counter increases
                event += 1 

        
   
        
         
        
        #if the next arrival before the departure of current customer, server will be busy at arrival
        if next_arrival_time < departure_time:
            server_status = 'busy' 
            
        else: 
            server_status = 'idle'
            
        
    return time_series


ts1=generate_ts(25)

'''
.....................................................................................................................
.......................................................EVENTs TIME SERIES END.................................................
......................................................................................................................
'''








'''-------------------------------   Customer arrival/departure time dataframe -------------------------------'''


def get_customer_tf(time_series):
    arrivals = time_series.loc[time_series['type'] == 'arrival', ['time', 'arr cust' ]]
    arrivals.columns = ['time', 'customer']
    #get departing customers
    depature = time_series.loc[time_series['type'] == 'departure', ['time', 'depar cust' ]]
    depature.columns = ['time', 'customer']
    #get customers being served
    serving = time_series.loc[time_series['served cust'] != 0 , ['time', 'served cust' ]]
    serving.columns = ['time', 'customer']

    #merge 
    customer_df = arrivals.merge(depature, on='customer')
    customer_df = customer_df.merge(serving, on='customer')
    customer_df.columns = ['arrival time', 'customer', 'departure time', 'serving time']
    customer_df = customer_df[['customer', 'arrival time', 'serving time', 'departure time']] 

    #get time in queue
    customer_df['time in queue'] = customer_df['serving time'] - customer_df['arrival time'] 
    #get time in system
    customer_df['time in system'] = customer_df['departure time'] - customer_df['arrival time'] 
    #get time in server
    customer_df['time in server'] = customer_df['departure time'] - customer_df['serving time'] 
    #round all floats to 2 digits
    customer_df = customer_df.round(2)
    
    
    
    
    '''add recorded idle time for each customer waiting for 0 time units -------------------------------'''
    def add_idle(row):
        if row.name == 0:
            return row['arrival time']
        elif row['time in queue'] == 0:
            prev_row = customer_df.loc[row.name - 1]
            return row['arrival time']- prev_row['departure time'] 
        else: return 0
    
    '''add recorded time intervals between arrivals -------------------------------'''
    def add_intervals(row):
        if row.name == 0:
            return row['arrival time']
        else:
            prev_row = customer_df.loc[row.name - 1]
            return row['arrival time']- prev_row['arrival time'] 

        
    '''add states of customer waiting -------------------------------'''
    def add_waiting_states(row):
         if row['time in queue']!= 0:
             return 1
         else:
             return 0   
        
    customer_df['idle time']=customer_df.apply(add_idle, axis=1)
    customer_df['intervals']=customer_df.apply(add_intervals, axis=1)
    customer_df['wait state']=customer_df.apply(add_waiting_states, axis=1)
    
    return customer_df

customer_tf1=get_customer_tf(ts1)
#plt.hist(customer_tf1['time in queue'])
fig1, ax1 = plt.subplots()
customer_tf1['time in queue'].hist(ax=ax1)
ax1.set(title='Frequency of individual customer waiting time')









'''--------------get a single queue data averages:'time in queue','time in server','time in system' and idle time--------------------------------------------'''
def get_Q_avgs(customer_df):
    sum_idle = customer_df['idle time'].sum()
    #sum_await= customer_df['wait state'].sum()
    total_time= customer_df['departure time'].iloc[-1]
    results_df=customer_df[['time in queue','time in server','time in system','wait state','intervals']].mean()
    results_df['idle prob']=sum_idle/total_time
    new_df = pd.DataFrame(columns=results_df.index)
    new_df.loc[0] = results_df.values
    print(new_df)
    return new_df





'''############################################## run a single queue ####################################################'''
def run_queue(seed=31):
    time_series=generate_ts(seed)
    print(time_series)
    
    customer_df=get_customer_tf(time_series)
    print(customer_df)
    
    q_avgs=get_Q_avgs(customer_df)
    print(q_avgs)
    
    return q_avgs


avgs_sample=run_queue(31)

#print(test_single_q['time in queue'])

''''---------------------------------------simulation of experiments------------------------------------'''
def run_experiments(n_runs=50):
    df = pd.DataFrame(columns = ['time in queue', 'time in server', 'time in system','intervals','idle prob'])
    for i in range(n_runs):
        run_result = pd.DataFrame(run_queue(i))
        run_result.fillna(0, inplace=True)
        #new_row = {'time in queue':run_result['time in queue'],'time in server':run_result['time in server'],'time in system':run_result['time in system'],'idle time':run_result['idle time']}
        df=  df.append(run_result)
    df = df.head(50) # Resize df to have 50 rows
    df.reset_index(inplace=True, drop=True) 
    #df['run number'] = range(1, n_runs+1)
    #df = df[['time in queue', 'time in server', 'time in system','idle_time']] #rearrange columns
    return df





experiments = run_experiments()
#plt.hist(experiments['time in queue'])
fig2, ax2 = plt.subplots()
experiments['time in queue'].hist(ax=ax2)
ax2.set(title='Histogram for average customer waiting')
experiments_avg=experiments.mean()

'''-----------------------Get experiments values avgs------------------------'''


