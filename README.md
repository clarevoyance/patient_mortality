# ML Pipeline for Predicting Mortality Rate

## Overview
We will build an ML pipeline to predict the mortality of a patient given their clinical history. This may include their diagnosis, drug consumption, and laboratory test results. This is a retrospective records-basedstudy on patients from their past electronic health records (EHRs). Care has been taken to remove all personal identifiable information (PII) and/or protected health information (PHI) in compliance with HIPAA research-related rules has been de-identified. 

## Data
We will be using the following datasets found in the `/data` folder which are further divided between the training and test data set:

`/train`
- `event_feature_map.csv`
- `events.csv`
- `mortality_events.csv`

`/test`
- `event_feature_map.csv`
- `events.csv`

### `events.csv`
- `'patient_id'`
- `'event_id'`
- `'event_description'`
- `'timestamp'`
- `'value'`

This dataset describes the clinical events `['event_id']` that have occurred for each respective patient `['patient_id']` and the date `[timestamp']` of when the event occurred. There is additional meta information in `['event_description']` describing the respective event. There will be `740,066` labeled records in the training dataset and `488,347` unlabelled records in the test dataset. 

### `event_feature_map.csv`
- `'idx'`
- `'event_id'`
This csv file is for feature mapping the events that have occurred in `events.csv` to a respective `['idx']` as a feature label during the model training process. This is to optimize the model training time given that this will be a sparse matrix (i.e. most patients will only have a small subset of all the possible events occurring).

### `mortality_events.csv`
- `'patient_id'`
- `'timestamp'`
- `'label'`
This spreadsheet indicates the date `['timestamp']` of death for each patient `['patient_id']` and the label is set at the value of `1` by default to indicate it is a positive label. We will assume all patients that are in `events_csv` but not in `mortality_events.csv` to be alive and will have a label of `0` during the ETL process.

