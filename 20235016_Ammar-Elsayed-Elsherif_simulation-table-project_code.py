# -*- coding: utf-8 -*-
'''
-Author: Ammar Elsayed Elsherif

-Task: Simulation of a grocery store single server queue
 (uniform inter-arrival times U(1, 8), uniform service times U(1, 6))

-Pipeline:
 1- generate_ts       : event time series of one run (simulation table)
 2- get_customer_tf   : per customer arrival / serving / departure table
 3- get_Q_avgs        : averages of one run
 4- run_experiments   : repeat the run with seeds 0..49 and collect the averages

-Libraries: numpy, pandas, matplotlib only
-Tested with pandas 1.5.x and 2.x (no df.append, no chained assignment)
'''
# libraries used

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


'''-------------------------------- simulation parameters --------------------------------'''
MIN_INTER_ARR = 1
MAX_INTER_ARR = 8

MIN_SERVICE_TIME = 1
MAX_SERVICE_TIME = 6

N_EVENTS = 200

# columns of the event time series
TS_COLUMNS = ['event', 'time', 'type',
              'queue', 'arr cust', 'served cust', 'depar cust']

# column order of the experiments table (same order as the reference output)
EXPERIMENT_COLUMNS = ['time in queue', 'time in server', 'time in system',
                      'intervals', 'idle prob', 'wait state']

# set to False to skip writing the histograms to disk
SAVE_FIGURES = True


'''------------------------------------ events time series ------------------------------------'''
def add_events(time_series, new_rows):
    # append the generated events to the existing time series
    new_events = pd.DataFrame(new_rows, columns=TS_COLUMNS)
    time_series = pd.concat([time_series, new_events])
    # events are sorted by time (order of the new rows doesnt matter)
    time_series = time_series.sort_values(['time'])
    time_series.reset_index(drop=True, inplace=True)
    # event number is assigned by time order
    time_series['event'] = list(range(1, time_series.shape[0] + 1))
    return time_series


def generate_ts(seed):
    # initial state
    np.random.seed(seed)
    event = 0
    time_ = 0

    # counters for arrived, served and departed customers
    arrived_customers = 0
    served_customers = 0
    departed_customers = 0

    # generate random variables for next events
    interarrival_time = np.random.uniform(MIN_INTER_ARR, MAX_INTER_ARR)
    next_arrival_time = time_ + interarrival_time
    departure_time = 0
    # server status is the status expected at the NEXT arrival
    server_status = 'idle'
    queue = 0
    arrived_customers += 1

    '''----------------------- end of event 0 -----------------------'''

    event += 1
    # create time series and populate with event 1 details
    time_series = pd.DataFrame([[1, float(next_arrival_time), 'arrival',
                                 queue, arrived_customers, 0, 0]],
                               columns=TS_COLUMNS)

    while event <= N_EVENTS:
        # event starts, parameters at event t
        event_type = time_series['type'].iloc[event - 1]
        time_ = time_series['time'].iloc[event - 1]

        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! if event = arrival !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        if event_type == 'arrival':

            # counter of arrived customers increases by 1
            arrived_customers += 1

            # generate next arrival time
            interarrival_time = np.random.uniform(MIN_INTER_ARR, MAX_INTER_ARR)
            next_arrival_time = time_ + interarrival_time

            # if server status is idle customer is served immediately
            # and generates service time
            if server_status == 'idle':
                # customer is served and counter of served customers increases by 1
                served_customers += 1
                # this customer number is added to the 'served cust' column at event n
                time_series.at[event - 1, 'served cust'] = served_customers

                # generate next events (service and departure time)
                service_time = np.random.uniform(MIN_SERVICE_TIME, MAX_SERVICE_TIME)
                departure_time = time_ + service_time
                departed_customers += 1

                # add generated events to existing time series
                time_series = add_events(time_series, [
                    [event, float(departure_time), 'departure', 0, 0, 0, departed_customers],
                    [event, float(next_arrival_time), 'arrival', 0, arrived_customers, 0, 0]
                ])

            # if server status is busy increase queue and only generates arrival activity
            else:
                queue += 1
                # add generated events to existing time series
                time_series = add_events(time_series, [
                    [event, float(next_arrival_time), 'arrival', 0, arrived_customers, 0, 0]
                ])
                time_series.at[event - 1, 'queue'] = queue

            # event is finished and event counter increases
            event += 1

        '''!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! if event = departure !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'''
        if event_type == 'departure':

            # if queue is zero and customer departs, server status remains idle
            # and nothing else happens until next arrival
            if queue == 0:
                server_status = 'idle'

            # if there are customers in queue (>0), server changes to busy
            # and queue decreases by one
            else:
                # customer is served and counter of served customers increases by 1
                served_customers += 1
                # this customer number is added to the 'served cust' column at event n
                time_series.at[event - 1, 'served cust'] = served_customers

                queue -= 1
                server_status = 'busy'

                # generate next events (service and departure time)
                service_time = np.random.uniform(MIN_SERVICE_TIME, MAX_SERVICE_TIME)
                departure_time = time_ + service_time
                # same customer that is served departs at departure time
                departed_customers += 1

                # add generated events to existing time series
                time_series = add_events(time_series, [
                    [event, float(departure_time), 'departure', 0, 0, 0, departed_customers]
                ])
                time_series.at[event - 1, 'queue'] = queue

            # event is finished and event counter increases
            event += 1

        # if the next arrival is before the departure of current customer,
        # server will be busy at arrival
        if next_arrival_time < departure_time:
            server_status = 'busy'
        else:
            server_status = 'idle'

    return time_series


'''------------------------------- customer arrival/departure time dataframe -------------------------------'''
def get_customer_tf(time_series):
    # get arriving customers
    arrivals = time_series.loc[time_series['type'] == 'arrival', ['time', 'arr cust']]
    arrivals.columns = ['arrival time', 'customer']
    # get departing customers
    departures = time_series.loc[time_series['type'] == 'departure', ['time', 'depar cust']]
    departures.columns = ['departure time', 'customer']
    # get customers being served
    serving = time_series.loc[time_series['served cust'] != 0, ['time', 'served cust']]
    serving.columns = ['serving time', 'customer']

    # merge on customer number
    customer_df = arrivals.merge(departures, on='customer')
    customer_df = customer_df.merge(serving, on='customer')
    customer_df = customer_df[['customer', 'arrival time', 'serving time', 'departure time']]

    # get time in queue
    customer_df['time in queue'] = customer_df['serving time'] - customer_df['arrival time']
    # get time in system
    customer_df['time in system'] = customer_df['departure time'] - customer_df['arrival time']
    # get time in server
    customer_df['time in server'] = customer_df['departure time'] - customer_df['serving time']
    # round all floats to 2 digits
    customer_df = customer_df.round(2)

    # add recorded idle time for each customer waiting for 0 time units
    # (first customer: idle since time 0, others: gap since previous departure)
    prev_departure = customer_df['departure time'].shift(1)
    customer_df['idle time'] = np.where(customer_df['time in queue'] == 0,
                                        customer_df['arrival time'] - prev_departure,
                                        0.0)
    customer_df.loc[customer_df.index[0], 'idle time'] = customer_df['arrival time'].iloc[0]

    # add recorded time intervals between arrivals (first interval starts at time 0)
    customer_df['intervals'] = customer_df['arrival time'].diff().fillna(customer_df['arrival time'])

    # add states of customer waiting (1 = waited, 0 = served immediately)
    customer_df['wait state'] = (customer_df['time in queue'] != 0).astype(int)

    return customer_df


'''-------------- get a single queue data averages: 'time in queue', 'time in server', 'time in system' and idle prob --------------'''
def get_Q_avgs(customer_df, verbose=True):
    sum_idle = customer_df['idle time'].sum()
    total_time = customer_df['departure time'].iloc[-1]
    results = customer_df[['time in queue', 'time in server', 'time in system',
                           'wait state', 'intervals']].mean()
    results['idle prob'] = sum_idle / total_time

    # one row dataframe so runs can be stacked later
    q_avgs = pd.DataFrame([results.values], columns=results.index)
    if verbose:
        print(q_avgs)
    return q_avgs


'''############################################## run a single queue ####################################################'''
def run_queue(seed=31, verbose=True):
    time_series = generate_ts(seed)
    customer_df = get_customer_tf(time_series)
    q_avgs = get_Q_avgs(customer_df, verbose=False)
    if verbose:
        print(time_series)
        print(customer_df)
        print(q_avgs)
    return q_avgs


'''--------------------------------------- simulation of experiments ---------------------------------------'''
def run_experiments(n_runs=50):
    # one run per seed (0 .. n_runs-1), results are stacked into a single dataframe
    runs = [run_queue(i, verbose=False) for i in range(n_runs)]
    df = pd.concat(runs, ignore_index=True)
    df.fillna(0, inplace=True)
    # fix column order
    return df[EXPERIMENT_COLUMNS]


'''---------------------------------------------- main ----------------------------------------------'''
if __name__ == '__main__':

    # single run: time series and customer table (seed 25)
    ts1 = generate_ts(25)
    customer_tf1 = get_customer_tf(ts1)

    fig1, ax1 = plt.subplots()
    customer_tf1['time in queue'].hist(ax=ax1)
    ax1.set(title='Frequency of individual customer waiting time')

    # single run averages (seed 31)
    avgs_sample = run_queue(31)

    # 50 runs experiment
    experiments = run_experiments()

    fig2, ax2 = plt.subplots()
    experiments['time in queue'].hist(ax=ax2)
    ax2.set(title='Histogram for average customer waiting')

    '''----------------------- get experiments values avgs ------------------------'''
    experiments_avg = experiments.mean()
    print(experiments_avg)

    if SAVE_FIGURES:
        fig1.savefig('histogram-1.png')
        fig2.savefig('histogram-2.png')

    plt.show()
