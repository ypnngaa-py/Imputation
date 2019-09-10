import pandas as pd
import scipy
import numpy as np
import numpy as np
import random
import time
import os, sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
def transform(pre_value):
	xdim, ydim = pre_value.shape
	res_pred = np.zeros((xdim,ydim))
	for i in xrange(xdim):
		for j in xrange(ydim):
			if pre_value[i][j] >= 1.5:
				res_pred[i][j] = 2
			elif pre_value[i][j] >= 0.5:
				res_pred[i][j] = 1
			else:
				res_pred[i][j] = 0
	return res_pred
def Training(df,seed, yratio, xratio, index = 1):
	snp_matrix = np.array(df.values)
	xdim, ydim = snp_matrix.shape

	ydimlist = range(0,ydim)
	xdimlist = range(0,xdim)

	random.seed(seed)
	random.shuffle(ydimlist) # shuffle the individuals
	random.shuffle(xdimlist) # shuffle the SNPs	
	accuracy = 0

	snp_matrix_shuffle = np.copy(snp_matrix[:,ydimlist])
	snp_matrix_shuffle = np.copy(snp_matrix[xdimlist,:])
	snp_matrix_train = snp_matrix_shuffle[:,0:int(ydim*yratio)]
	snp_matrix_test = snp_matrix_shuffle[:,int(ydim*yratio):]

	snp_matrix_train_x = snp_matrix_train[0:int(xdim*xratio),:]
	snp_matrix_test_x = snp_matrix_test[0:int(xdim*xratio),:]

	for i in range(int(xdim*xratio), xdim):
		snp_matrix_train_y = snp_matrix_train[i,:]
		snp_matrix_test_y = snp_matrix_test[i,:]
		if index != 7:
			if index == 1:
				clf = AdaBoostClassifier(n_estimators= 100)
			elif index == 2:
				clf = RandomForestClassifier(n_estimators=100)
			elif index == 3:
				clf = linear_model.LogisticRegression(C=1e5)
			elif index == 4:
				clf = svm.SVC(kernel = 'rbf')
			elif index == 5:
				clf = svm.SVC(kernel = 'poly')
			else:
				clf = svm.SVC(kernel = 'linear')
			clf = clf.fit(snp_matrix_train_x.T, snp_matrix_train_y)
			Y_pred = clf.predict(snp_matrix_test_x.T)
			prediction = snp_matrix_test_y - Y_pred
			wrong = np.count_nonzero(prediction)
			tmp = 1 - (wrong + 0.0) / len(prediction)
			print tmp
			accuracy += tmp

	accuracy = accuracy / (xdim - int(xdim*xratio))

	if index == 7:
		pls2 = PLSRegression(n_components = 50, scale=False, max_iter=1000)
		snp_matrix_train_y = snp_matrix_train[int(xdim*xratio):,:]
		pls2.fit(snp_matrix_train_x.T,snp_matrix_train_y.T)
		snp_matrix_test_x = snp_matrix_test[0:int(xdim*xratio),:]
		snp_matrix_test_y = snp_matrix_test[int(xdim*xratio):,:]		
		Y_pred = transform(pls2.predict(snp_matrix_test_x.T))
		prediction = snp_matrix_test_y - Y_pred.T
		xdim, ydim = prediction.shape
		wrong = np.count_nonzero(prediction)
		accuracy = 1 - wrong / (xdim * ydim + 0.0)
	return accuracy
def read_file(filename):
	with open(filename) as fp:
		List = [x.strip() for x in fp if len(x.strip())>0]
		return List
def baseline_method(seed, yratio, xratio):
	df = pd.read_csv('./data/imputation_training.txt',sep=' ')
	snp_matrix = np.array(df.values)
	xdim, ydim = snp_matrix.shape

	ydimlist = range(0,ydim)
	xdimlist = range(0,xdim)

	random.seed(seed)
	random.shuffle(ydimlist) # shuffle the individuals
	random.shuffle(xdimlist) # shuffle the SNPs	

	snp_matrix_shuffle = np.copy(snp_matrix[:,ydimlist])
	snp_matrix_shuffle = np.copy(snp_matrix[xdimlist,:])
	snp_matrix_train = snp_matrix_shuffle[:,0:int(ydim*yratio)]
	snp_matrix_test = snp_matrix_shuffle[:,int(ydim*yratio):]

	snp_matrix_train_x = snp_matrix_train[0:int(xdim*xratio),:]
	snp_matrix_test_x = snp_matrix_test[0:int(xdim*xratio),:]
	snp_matrix_test_y = snp_matrix_test[int(xdim*xratio):,:]
	snp_matrix_train_y = snp_matrix_train[int(xdim*xratio):,:]

	snp_matrix_train_whole = np.copy(snp_matrix_shuffle[int(xdim*xratio):,:])
	snp_matrix_train_impute = np.copy(snp_matrix_train_whole)
	snp_matrix_train_impute[:,int(ydim*yratio):] = -9999
	imp = Imputer(missing_values=-9999, strategy='most_frequent', axis=1)
	snp_matrix_train_impute = imp.fit_transform(snp_matrix_train_impute)
	wrong = np.count_nonzero(snp_matrix_train_whole - snp_matrix_train_impute)
	accuracy = 1 - (wrong + 0.0) / (ydim * (1-yratio) * xdim * (1-xratio))
	return accuracy
def draw_ggplot(path):
	algorithm = ['AdaBoost','RandomForest','LogisticRegression','SVM-Kernel','SVM-Poly','SVM-Linear','PLS','Baseline']
	fw = open('ggplot.txt','w')
	fw.write('Method\tAccuracy\tMissingRatio\n')
	for missing_ratio in [0.05,0.1,0.2,0.3,0.4,0.5]:
		accuracy = baseline_method(1,0.75,1 - missing_ratio)
		print accuracy
		for i in range(1,8):
			acc = float(read_file(path + '1_0.75_{0}_{1}'.format(1 - missing_ratio,i))[0].strip())
			fw.write(algorithm[i-1] + '\t' + str(acc) + '\t' + str(missing_ratio) + '\n')
		fw.write(algorithm[-1] + '\t' + str(accuracy) + '\t' + str(missing_ratio) + '\n')
	fw.close()
	print 'setwd("/Users/panzhicheng/GoogleDrive/homework/M224/Final_Projet/")'
	print 'library("ggplot2")'
	print 'data <- read.table("ggplot.txt",header=T);'
	print 'p <- ggplot(data,aes(x=MissingRatio,y=Accuracy,group=Method,colour=Method)) + geom_line() + theme(text = element_text(size=20),legend.background = element_rect(fill=alpha("blue", 0)),panel.margin = unit(0.1, "lines"),panel.background = element_rect(fill = "transparent",colour = NA),panel.border = element_rect(colour = "grey", fill=NA)) + ylab("Accuracy") + xlab("Missing ratio");'
if __name__=="__main__":
	args = sys.argv
	if len(args) != 4:
		print '''
Usage:
	python imp_adaboost.py seed yratio x_missing_ratio LearningAlgorithm
		'''
	else:
		seed, yratio, x_missing_ratio ,learningAlgorithm = int(args[1]), float(args[2]), 1 - float(args[3]), int(args[4])
		df = pd.read_csv('./data/imputation_training.txt',sep=' ')
		accuracy = Training(df,seed, yratio, x_missing_ratio,learningAlgorithm)
		fw = open('./test/{0}_{1}_{2}_{3}'.format(seed,yratio,x_missing_ratio,learningAlgorithm),'w')
		print accuracy
		fw.write(str(accuracy) + '\n')
		fw.close()