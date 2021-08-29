import time
import pandas as pd
import numpy as np

def read_csv(filepath):
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')
    return events, mortality

def event_count_metrics(events, mortality):
    # Calculate total events for each patient_id
    total_event_count = events['patient_id'].value_counts()
        
    # Segment total events for dead and alive patients
    dead_event_count = total_event_count[mortality['patient_id']]    
    alive_event_count = total_event_count.drop(mortality['patient_id'])

    # Unit test to ensure no patient_id is left out
    assert len(dead_event_count + alive_event_count) == len(total_event_count), 'events for dead and alive patients do not tally with total patients'

    avg_dead_event_count = dead_event_count.mean()
    max_dead_event_count = dead_event_count.max()
    min_dead_event_count = dead_event_count.min()
    avg_alive_event_count = alive_event_count.mean()
    max_alive_event_count = alive_event_count.max()
    min_alive_event_count = alive_event_count.min()
    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    # Calculate total encounters for each patient_id
    total_encounter_count = events[['patient_id', 'timestamp']].groupby(['patient_id'])
    total_encounter_count = total_encounter_count['timestamp'].nunique()

    # Segment total encounters for dead and alive patients
    dead_encounter_count = total_encounter_count[mortality['patient_id']]    
    alive_encounter_count = total_encounter_count.drop(mortality['patient_id'])

    # Unit test to ensure no patient_id is left out
    assert len(dead_encounter_count + alive_encounter_count) == len(total_encounter_count), 'Dead and alive patients do not tally with total patients'

    avg_dead_encounter_count = dead_encounter_count.mean()
    max_dead_encounter_count = dead_encounter_count.max()
    min_dead_encounter_count = dead_encounter_count.min()
    avg_alive_encounter_count = alive_encounter_count.mean()
    max_alive_encounter_count = alive_encounter_count.max()
    min_alive_encounter_count = alive_encounter_count.min()
    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    events['timestamp'] = pd.to_datetime(events['timestamp'])

    rec_len = events[['patient_id','timestamp']].groupby('patient_id').agg([max, min])
    rec_len['duration'] = rec_len['timestamp']['max'] - rec_len['timestamp']['min']
    total_rec_len = rec_len['duration']

    # Segment rec length ofr dead and alive patients
    dead_rec_len = total_rec_len[mortality['patient_id']].dt.days
    alive_rec_len = total_rec_len.drop(mortality['patient_id']).dt.days

    # Unit test to ensure no patient_id is left out
    assert len(dead_rec_len + alive_rec_len) == len(total_rec_len), 'Dead and alive patients do not tally with total patients'

    avg_dead_rec_len = dead_rec_len.mean()
    max_dead_rec_len = dead_rec_len.max()
    min_dead_rec_len = dead_rec_len.min()
    avg_alive_rec_len = alive_rec_len.mean()
    max_alive_rec_len = alive_rec_len.max()
    min_alive_rec_len = alive_rec_len.min()
    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    train_path = '../data/train/'
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()
