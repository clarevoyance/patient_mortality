import time
import utils
import pandas as pd
import numpy as np
import codecs
import filecmp 


def read_csv(filepath):
    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')
    return events, mortality, feature_map

def calculate_index_date(events, mortality, deliverables_path):
    # Get date for dead patients
    dead_date = mortality.loc[:, ('patient_id', 'timestamp')]
    dead_date.loc[:, ('timestamp')] = pd.to_datetime(dead_date['timestamp'])
    dead_date.loc[:, ('timestamp')] = dead_date['timestamp'] - pd.to_timedelta(30, unit='d')

    # Get date for alive patients, we will first get all the dates and the last event
    all_date = events.loc[:,('patient_id' , 'timestamp')].groupby(['patient_id']).agg({'timestamp': 'max'}).reset_index()
    alive_date = all_date.loc[~all_date['patient_id'].isin(mortality.loc[:,'patient_id'])]
    alive_date.loc[:,('timestamp')] = pd.to_datetime(alive_date.loc[:,('timestamp')])

    # Unit testing to ensure no patient_id is left out
    assert len(dead_date) + len(alive_date) == len(all_date), 'length of dates to not tally up'

    indx_date = pd.concat([dead_date, alive_date])
    indx_date = indx_date.rename(columns = {'timestamp' : 'indx_date'})
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    return indx_date

def filter_events(events, indx_date, deliverables_path):
    df = pd.merge(events, indx_date, how="left", on='patient_id')
    df.indx_date = pd.to_datetime(df.loc[:,'indx_date'])
    df.timestamp = pd.to_datetime(df.loc[:,'timestamp'])

    time_diff = (df.loc[:,'indx_date'] - df.loc[:,'timestamp'])
    time_diff = (time_diff / np.timedelta64(1, 'D')).astype(int)
    time_diff_mask = np.logical_and(time_diff >= 0, time_diff <= 2000)

    # original code
    filtered_events = df[time_diff_mask]
    filtered_events = filtered_events[['patient_id', 'event_id', 'value']]

    filtered_events = filtered_events.loc[:,('patient_id', 'event_id', 'value')]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', index=False)

    return filtered_events    

def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    # Replace event_id with index
    idx_events = pd.merge(filtered_events_df, feature_map_df, how='left', on='event_id')
    idx_events = idx_events.dropna(subset=['value'])
    idx_events = idx_events.loc[:,('patient_id','idx','value')]

    # Separate diagnostics+drug events from lab events
    d_events = idx_events[idx_events.loc[:,'idx']<2680]
    l_events = idx_events[idx_events.loc[:,'idx']>=2680]
    assert len(d_events) + len(l_events) == len(idx_events), 'events do not tally up'

    # sum for d_events and count for l_events
    d_events = d_events.groupby(['patient_id', 'idx']).agg('sum')
    l_events = l_events.groupby(['patient_id', 'idx']).agg('count')
    d_events.reset_index(inplace = True) 
    l_events.reset_index(inplace = True)
    
    aggregated_events = pd.concat([d_events, l_events])

    # min-max normalization
    min_max = aggregated_events[['idx', 'value']].groupby(['idx']).agg({'value': ['min', 'max']})
    min_max.columns = ['_'.join(col).strip() for col in min_max.columns.values]
    min_max.reset_index(inplace=True)


    aggregated_events = aggregated_events.merge(min_max, how='left', on=['idx'])

    aggregated_events['value_norm'] = (aggregated_events.value) / (aggregated_events.value_max)
    aggregated_events = aggregated_events[['patient_id',
                                           'idx',
                                           'value_norm']].rename(columns={'idx':'feature_id', 'value_norm':'feature_value'})
    
    # export and output
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    deliverables_path = '../deliverables/'

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

    # Create dictionary for patient_features
    patients = aggregated_events.patient_id.drop_duplicates().tolist()
    patient_features = {k:[] for k in patients}

    for index, row in aggregated_events.iterrows():
        patient_features[row.patient_id].append((row.feature_id, row.feature_value))

    # Create dictionary for mortality
    mortality_dict = pd.Series(mortality.loc[:,'label'].values, index = mortality.loc[:,'patient_id']).to_dict()
    assert type(mortality_dict) == dict, 'mortality is not dict type'
    return patient_features, mortality_dict

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    deliverable1 = codecs.open(op_file, 'wb')
    deliverable2 = codecs.open(op_deliverable, 'wb')
    
    patients = list(patient_features.keys())
    patients.sort()

    for patient in patients:
        features = patient_features[patient]
        features = sorted(features, key=lambda x: x[0])
        patient_features[patient] = features

    for patient in patients:
        if patient in mortality:
            deliverable1_text = '1 '  
        else:
            deliverable1_text = '0 '    
        
        for features in patient_features[patient]:
            deliverable1_text += (str(int(features[0])) + ':' + str("%.6f" % features[1]) + ' ')

        deliverable1_text = deliverable1_text[:-1] 
        deliverable1_text += ' \n'
        deliverable2_text = str(int(patient)) + ' ' + deliverable1_text

        deliverable1.write(bytes(deliverable1_text, 'UTF-8'))
        deliverable2.write(bytes(deliverable2_text, 'UTF-8'))

    deliverable1.close()
    deliverable2.close()

def main():
    train_path = '../data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')

if __name__ == "__main__":
    main()