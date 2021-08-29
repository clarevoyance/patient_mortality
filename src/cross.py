import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import numpy as np
import utils


RANDOM_STATE = 545510477

def get_acc_auc_kfold(X,Y,k=5):
	kf = KFold(n_splits=k, random_state=RANDOM_STATE, shuffle=True)
	
	acc_ = np.array([])
	auc_ = np.array([])

	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		Y_pred = models_partc.logistic_regression_pred(X_train, Y_train, X_test)
		acc = accuracy_score(Y_pred, Y_test)
		auc = roc_auc_score(Y_pred, Y_test)
		acc_ = np.append(acc_, acc)
		auc_ = np.append(auc_, auc)

	accuracy = np.mean(acc_)
	auc = np.mean(auc_)
	return accuracy, auc

def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	kf = ShuffleSplit(n_splits=iterNo, random_state=RANDOM_STATE, test_size=test_percent)
	
	acc_ = np.array([])
	auc_ = np.array([])

	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		Y_pred = models_partc.logistic_regression_pred(X_train, Y_train, X_test)
		acc = accuracy_score(Y_pred, Y_test)
		auc = roc_auc_score(Y_pred, Y_test)
		acc_ = np.append(acc_, acc)
		auc_ = np.append(auc_, auc)

	accuracy = np.mean(acc_)
	auc = np.mean(auc_)
	return accuracy, auc

def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

