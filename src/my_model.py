import utils
import etl
import models_partc

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


def my_features(events, feature_map):
	#TODO: complete this

	idx_events = pd.merge(events, feature_map, how='left', on='event_id')
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
	aggregated_events = aggregated_events[['patient_id', 'idx',
											'value_norm']].rename(columns={'idx':'feature_id', 'value_norm':'feature_value'})	

	patients = aggregated_events.patient_id.drop_duplicates().tolist()
	patient_features = {k:[] for k in patients}

	for index, row in aggregated_events.iterrows():
		patient_features[row.patient_id].append((row.feature_id, row.feature_value))
	
	deliverable = open('../deliverables/test_features.txt', 'wb')


	patients = list(patient_features.keys())
	patients.sort()

	for patient in patients:
		features = patient_features[patient]
		features = sorted(features, key=lambda x: x[0])
		patient_features[patient] = features
		
	for patient in patients:
		deliverable_text = str(patient) + ' '
        
		for features in patient_features[patient]:
			deliverable_text += (str(int(features[0])) + ':' + str("%.6f" % features[1]) + ' ')
		
		deliverable.write(bytes(deliverable_text, 'UTF-8'))
		deliverable.write(bytes('\n', 'UTF-8'))

	deliverable.close()


def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
	
	clf = RandomForestClassifier(random_state=545510477, warm_start=True)
	clf.fit(X_train, Y_train)

	Y_pred = clf.predict_proba(X_test)[:, 1]

	return Y_pred


def main():
	events = pd.read_csv('../data/test/events.csv')
	feature_map = pd.read_csv('../data/test/event_feature_map.csv')

	my_features(events, feature_map)

	X_train, Y_train = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	X_test, Y_test = utils.get_data_from_svmlight("../deliverables/test_features.txt")

	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	print(Y_pred)

	#Generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.
	utils.generate_submission("../deliverables/test_features.txt", Y_pred)
	
if __name__ == "__main__":
    main()

	