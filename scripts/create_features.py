"""
TODO:
- create test and train data
- test output on subclasses
- use labelencoder sklearn?
"""
import glob
import pickle
import re
import itertools	# for flattening list
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
import operator
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
import sklearn
from sklearn import cross_validation
from pprint import pformat 
import hashlib
import time
import random
import csv
from sklearn.externals.six import StringIO
import math

import pandas as pd

class Create_features(object):

	# COMMENT FOR NFI
	#STARTPATH = '/images/'
	#UNCOMMENT FOR NFI
	STARTPATH = '/home/sharon/Documents/test/'
	FILEPATTERN = STARTPATH + '*.output.djpeg-dqt'
	PICKLEFILE = STARTPATH + 'merge.pickletotal'
	PICKLE_PATTERN = STARTPATH + '*.pickle'
	ITEMS = []
	camera_dict = {}
	feature_dict = {}
	class_to_int_dict = {}

	def __init__(self, load=False, dump=False, multiload=False):
		""" Load all files to dictionary. Can be loaded and/or dumped from/to pickle files"""
		if multiload:
			load = True
		self.load_files(load, dump, multiload)

	def load_files(self, load, dump, multiload):
		"""
		Load in all files with specified file pattern. Append infotuple to ITEMS array
		"""
		# load camera_dict from file
		if load:
			print "Loading dictionary from file.."
			if multiload:
				pickle_files = glob.glob(self.PICKLE_PATTERN)
				for filename in pickle_files:
					loadeddict = pickle.load( open( filename, "rb" ) )
					self.camera_dict = self.merge_dictionaries(self.camera_dict, loadeddict)
			else:
				self.camera_dict = pickle.load( open( self.PICKLEFILE, "rb" ) )
		else:
			files = glob.glob(self.FILEPATTERN)
			counter = 0
			print "Creating dictionary.."
			for name in files:
				try:
					infotuple = pickle.load( open( name, "rb" ) )
					self.create_dictionary(infotuple)
					counter += 1
					if counter % 1000 == 0:
						print counter
				except:
					print "problem with file %s" %(name)
			print "finished dictionary"

		# dump to file
		if dump:
			print "Dumping dictionary to file.."
			pickle.dump( self.camera_dict, open( self.PICKLEFILE, "wb" ) )
		total_files = 0
		manufacturers = {}
		for key,value in self.camera_dict.iteritems():
			print "%s : %i" %(key, len(value))
			total_files += len(value)
			make = key[0]
			manufacturers[make] = manufacturers.get(make,0)+1

		for key,value in manufacturers.iteritems():
			print "%s : %i" %(key, value)
		print "TOTAL: %i" %(total_files)

	def create_dictionary(self, infotuple):
		""" Extract camera make, model and the dqts """
		camerainfo = infotuple[0]
		try:
			# UNCOMMENT FOR NFI	
			identifier = self.get_identifier(camerainfo)
			
			# identifier is camera make and model
			# COMMENT FOR NFI
			#cameramake = re.sub(self.STARTPATH, '', camerainfo[0])
			#identifier = (cameramake, camerainfo[1])
			
			#UNCOMMENT FOR NFI
			#identifier = (camerainfo[0], camerainfo[1])
			# check dqt for correctness
			for j in range(1,3):
				for i in infotuple[j]:
					if len(i) != 8:
						raise Exception("problem for %s" %(infotuple[0]))
			dqts = [infotuple[1], infotuple[2]]
			# known make & model, append
			if identifier in self.camera_dict:
				value = self.camera_dict[identifier]
				self.camera_dict[identifier] = value + (dqts,)

			# new make & model
			else:
				self.camera_dict[identifier] = (dqts,)
		except:
			print "problem! %s, %s, %s" %(infotuple[0], infotuple[1], infotuple[2])

	def get_identifier(self, camerainfo):
		""" Extract camera make and model. Needed due to other subfolders in database"""
		sub1 = 'Dresden_Image_Database-'
		sub2 = 'fotos-'
		sub3 = 'Scanner'
		if any(sub2 in x for x in camerainfo):
			camerainfolist = self.retrieve_identifier_dresdenfolder(camerainfo, sub2)
		elif any(sub1 in x for x in camerainfo):
			camerainfolist = self.retrieve_identifier_dresdenfolder(camerainfo, sub1)

		elif any(sub3 in x for x in camerainfo):
			info = camerainfo[2]
			camerainfolist = info.split('-')[0].split('_')
		else:
			camerainfolist = camerainfo[0:2]		

		manufacturer = camerainfolist[0]
		cameramodel = ''.join(camerainfolist[1:])
		if '--camera-1-' in cameramodel:
			cameramodel = re.sub('--camera-1-', '', cameramodel)
		return (manufacturer, cameramodel)
			

	def retrieve_identifier_dresdenfolder(self, camerainfo, substring):
		""" Retrieve camera make and model from dresden style folders"""
		# get correct tuple and extract everythin after substring
		info = [s for s in camerainfo if substring in s][0]
		camerainfo_string = info.split(substring,1)[1]
		# split to get correct list
		info_split = camerainfo_string.split('-')
		camerainfolist = info_split[0].split('_')
		return camerainfolist


	def merge_dictionaries(self, dict1, dict2):
		""" Merge values of dictionaries """
		keys = set(dict1).union(dict2)
		default = ()
		mergedict = dict((k, dict1.get(k, default) + dict2.get(k, default)) for k in keys)
		return mergedict

	def run(self):
		"""
		Convert dqt to feature or hash and their class list. Create train and test sets. Run training and predictions
		"""
		# Convert dictionary to feature sets
		# do for every camera make & model in dictionary
		classlist = []
		h_featurelist = []
		dt_featurelist = []
		#dqtdict = {}
		print "Creating feature and class lists.."
		print "KEY VALUE = make"
		for key, value in self.camera_dict.iteritems():
			# for every different dqt for this camera make & model
			make = key[0]
			cameramodel = key
			keyvalue = make
			for dqtset in value:
				classlist.append(self.get_camera_id(keyvalue))
				h_featurelist.append(self.get_hash(dqtset))
				dt_featurelist.append(self.create_dt_feature_set(dqtset))
		print "DT> old shape: (%i, %i) " %(len(h_featurelist),len(h_featurelist[0]))
		print "number of unique hashes: %i" %(len(set(h_featurelist)))

		# feature selection dt set
		dt_featurelist_small = self.feature_selection(dt_featurelist, classlist)
		#dt_featurelist_small = dt_featurelist
		print "DT> new shape:"
		print dt_featurelist_small.shape

		print "Creating train and test sets.."
		# create indices for train and test sets

		h_precision_list = []
		h_recall_list = []
		h_f2_list = []
		dumb_h_precision_list =[]
		dumb_h_recall_list = []
		dumb_h_f2_list = []
		dt_precision_list = []
		dt_recall_list = []
		dt_f2_list = []


		stratifiedkfoldsplit = cross_validation.StratifiedKFold(classlist, n_folds=5)
		counter  = 0
		for i in stratifiedkfoldsplit:
			counter += 1
		#split = cross_validation.ShuffleSplit(len(classlist), test_size=.1, random_state=13)
		#for i in split:
			train, test = i
		# create train and test sets
			h_X_train = []
			dt_X_train = []
			y_train = []
			for index in train:
				h_X_train.append(h_featurelist[index])
				dt_X_train.append(dt_featurelist_small[index])
				y_train.append(classlist[index])	

			h_X_test = []
			dt_X_test = []
			y_test = []
			for index in test:
				h_X_test.append(h_featurelist[index])
				dt_X_test.append(dt_featurelist_small[index])
				y_test.append(classlist[index])	

			filename_csv= 'COMPARE%i.csv' %(counter)

			metrics_info = self.run_onefold_traintest(h_X_train, h_X_test, dt_X_train, dt_X_test, y_train, y_test, filename_csv)
			hash_precision, hash_recall, hash_f2 = metrics_info[0]
			hash_precision_dumb, hash_recall_dumb, hash_f2_dumb = metrics_info[1]
			dt_precision, dt_recall, dt_f2 = metrics_info[2]

			h_precision_list.append(hash_precision)
			h_recall_list.append(hash_recall)
			h_f2_list.append(hash_f2)

			dumb_h_precision_list.append(hash_precision_dumb)
			dumb_h_recall_list.append(hash_recall_dumb)
			dumb_h_f2_list.append(hash_f2_dumb)

			dt_precision_list.append(dt_precision)
			dt_recall_list.append(dt_recall)
			dt_f2_list.append(dt_f2)

		h_average_precision = sum(h_precision_list)/float(len(h_precision_list))
		h_average_recall = sum(h_recall_list)/float(len(h_recall_list))
		h_average_f2 = sum(h_f2_list)/float(len(h_f2_list))

		dumb_h_average_precision = sum(dumb_h_precision_list)/float(len(dumb_h_precision_list))
		dumb_h_average_recall = sum(dumb_h_recall_list)/float(len(dumb_h_recall_list))
		dumb_h_average_f2 = sum(dumb_h_f2_list)/float(len(dumb_h_f2_list))

		dt_average_precision = sum(dt_precision_list)/float(len(dt_precision_list))
		dt_average_recall = sum(dt_recall_list)/float(len(dt_recall_list))
		dt_average_f2 = sum(dt_f2_list)/float(len(dt_f2_list))

		print "\nHASH ALGORITHM:"
		for index in range(0,len(h_precision_list)):
			print "%i:\t%.4f\t%.4f\t%.4f" %(index, h_precision_list[index], h_recall_list[index], h_f2_list[index])
		print "Average:\t %.4f \t %.4f\t%.4f" %(h_average_precision, h_average_recall, h_average_f2)

		print "\nDUMB HASH ALGORITHM:"
		for index in range(0,len(dumb_h_precision_list)):
			print "%i:\t%.4f\t%.4f\t%.4f" %(index, dumb_h_precision_list[index], dumb_h_recall_list[index], dumb_h_f2_list[index])
		print "Average:\t %.4f \t %.4f\t%.4f" %(dumb_h_average_precision, dumb_h_average_recall, dumb_h_average_f2)

		print "\n\nDT ALGORITHM:"
		for index in range(0,len(dt_precision_list)):
			print "%i:\t%.4f\t%.4f\t%.4f" %(index, dt_precision_list[index], dt_recall_list[index], dt_f2_list[index])
		print "Average:\t %.4f \t %.4f\t%.4f" %(dt_average_precision, dt_average_recall, dt_average_f2)
		
		# create final classifier
		#clf_final = tree.DecisionTreeClassifier()
		#clf_final.fit(dt_featurelist_small, classlist)
		#with open("graphics.dot", 'w') as f:
		#	f = tree.export_graphviz(clf_final, out_file=f)

		#self.get_lineage(clf_final)

		

	def write_to_csv(self, predictions_hash, predictions_dt, y, dqts, filename_csv):
		with open(filename_csv, 'wb') as csvfile:
			c = csv.writer(csvfile, delimiter=',')
			array = ["ACTUAL","HASH","DT", "DQT"]
			c.writerow(array)
			for index in range(0, len(predictions_hash)):
				row = []
				if predictions_hash[index] != y[index] or predictions_dt[index] != y[index]:
					correctid = [k for k, v in self.class_to_int_dict.iteritems() if v == y[index]]
					hash_id = [k for k, v in self.class_to_int_dict.iteritems() if v == predictions_hash[index]]
					dt_id = [k for k, v in self.class_to_int_dict.iteritems() if v == predictions_dt[index]]
					row.append(correctid)
					row.append(hash_id)
					row.append(dt_id)
					# get dqts
					row.extend(dqts[index][:64])
					row.extend(dqts[index][169:233])
					c.writerow(row)

			

	def run_onefold_traintest(self, h_X_train, h_X_test, dt_X_train, dt_X_test, y_train, y_test, filename_csv):
		""" run one fold of training and testing for kfold """
	
		
		#h_X_train, h_X_test, h_y_train, h_y_test = cross_validation.train_test_split(h_featurelist, classlist, test_size=0.3, random_state=42)
		#dt_X_train, dt_X_test, dt_y_train, dt_y_test = cross_validation.train_test_split(dt_featurelist_small, classlist, test_size=0.3, random_state=42)
		#print len(h_X_test)

		# training
		print "Start training.."
		hashdict = self.train_hashfunction(h_X_train, y_train)
		hashdict_dumb = self.train_hashfunction_dumb(h_X_train, y_train)
		dt_clf = self.train_decisiontree(dt_X_train, y_train)

		# fit
		print "Start prediction.."
		predictions_hash = self.test_hashfunction(h_X_test, y_test, hashdict)
		predictions_hash_dumb = self.test_hashfunction_dumb(h_X_test, y_test, hashdict_dumb)
		predictions_dt = self.test_decisiontree(dt_X_test, y_test, dt_clf)


		self.write_to_csv(y_test, predictions_hash, predictions_dt, dt_X_test, filename_csv)

		# calculate statistics
		#(hash_precision, hash_recall) = self.get_recallprecision(y_test, predictions_hash)
		#(dt_precision, dt_recall) = self.get_recallprecision(y_test, predictions_dt)
		possible_labels = self.class_to_int_dict.values()
		print self.class_to_int_dict
		#print "METRICS FSCORE HASH"
		metrics_hash_dumb = sklearn.metrics.precision_recall_fscore_support(y_test, predictions_hash_dumb, beta=2, average='weighted')

		metrics_hash = sklearn.metrics.precision_recall_fscore_support(y_test, predictions_hash, beta=2, average='weighted')
		average_recall_hash = metrics_hash[1]
		average_precision_hash = self.get_precision_hash(y_test)
		average_f2_hash = self.calculate_f2_hash(average_precision_hash, metrics_hash[1], 2)
		metrics_hash = [average_precision_hash, average_recall_hash, average_f2_hash]
		#print "METRICS FSCORE DECISION TREE"
		metrics_dt = sklearn.metrics.precision_recall_fscore_support(y_test, predictions_dt, beta=2, labels=possible_labels, average='weighted')

		metrics_overview_dt = sklearn.metrics.precision_recall_fscore_support(y_test, predictions_dt, beta=2)
		print "overview"
		print metrics_overview_dt

		#print "HASH precision: %.4f \t recall: %.4f" %(hash_precision, hash_recall)
		#return (hash_precision, hash_recall, dt_precision, dt_recall)
		return (metrics_hash[:3], metrics_hash_dumb[:3], metrics_dt[:3])

	def calculate_f2_hash(self, precision, recall, beta):
		""" Calculate and return fbeta_score"""
		fbeta_score = (1 + math.pow(beta,2)) * ( (precision * recall)/( (math.pow(beta,2)*precision)+recall))
		return fbeta_score
			
	def get_precision_hash(self, y_test):
		""" calculate precision by extracting false positives"""
		total_precision = []
		weighted_precision = []
		counter = []
		for value in self.class_to_int_dict.values():
			correctcounter = y_test.count(value)
			counter.append(correctcounter)
			wrongcounter = self.false_positives.get(value,0)
			precision = correctcounter / float(correctcounter+wrongcounter)
			total_precision.append(precision)
			#weighted_precision.append(precision / float(correctcounter))	
		average = sum(total_precision) / float(len(total_precision))
		total = sum(counter)
		weighted_precision = 0
		for index in range(0,len(total_precision)):
			relevant = counter[index]
			weighted_precision += total_precision[index] *( relevant/float(total) )
		print "WEIGHTED: %f" %(weighted_precision)
		return weighted_precision

	def get_recallprecision(self, class_testset, predictions):
		""" retrieve precision & recall for tests """
		precision = sklearn.metrics.precision_score(class_testset, predictions)
		recall = sklearn.metrics.recall_score(class_testset, predictions)
		return (precision, recall)
		

	def get_hash(self, dqtset):
		"""
		For all items in camera dictionary convert quantizationtable to hashes. Return hashes and their class
		"""
		return hashlib.sha256(pformat(dqtset)).hexdigest()

	def create_dt_feature_set(self,dqtset):
		"""
		For all items in camera dictionary convert quantizationtable to features. Return feature array and their class
		"""	
		return self.convert_one(dqtset)

	def feature_selection(self, X, y):
		""" Perform feature selection on feature set. Return modified feature set """
		clf = ExtraTreesClassifier()
		X_new = clf.fit(X, y).transform(X) 
		print ">>>> shape feature selection:"
		print X_new.shape
		#important = clf.feature_importances_
		#self.print_importantfeatures(important)
		return X_new


	def print_importantfeatures(self, important):
		""" Print importance of features in readable format"""
		important = [i * 100 for i in important] 
		print max(important)
		
		print "IMPORTANT FEATURES"
		dqt1 = important[:64]
		rest1 = important[64:len(important)/2]
		dqt2 = important[len(important)/2:len(important)/2+64]
		rest2 = important[len(important)/2+64:]
		print "DQT1:"
		for i in range(0,8):
			myList = dqt1[i*8:i*8+8]
			print [ '%.2f' % elem for elem in myList ]

		print "rest DQT1:"
		print "\t Totalsum: %.2f, diagonalsum: %.2f, diagonalsum: %.2f, max: %.2f, min: %.2f" %(rest1[0], rest1[1], rest1[2], rest1[3], rest1[4] )
		print "\t Mean: %.2f, median: %.2f, var: %.2f, std: %.2f" %(rest1[5], rest1[6], rest1[7], rest1[8])
		for i in range(0,2):
			print "axis: %i" %(i)
			minlist = [ '%.2f' % elem for elem in rest1[9+56*i:17+56*i] ]
			maxlist = [ '%.2f' % elem for elem in rest1[17+56*i:25+56*i] ]
			meanlist = [ '%.2f' % elem for elem in rest1[25+56*i:32+56*i] ]
			medianlist = [ '%.2f' % elem for elem in rest1[32+56*i:40+56*i] ] 
			varlist = [ '%.2f' % elem for elem in rest1[40+56*i:48+56*i] ]
			stdlist =  [ '%.2f' % elem for elem in rest1[48+56*i:56+56*i] ]
			print "\tMin: %s \n\t Max: %s \n\t Mean: %s \n\t Median: %s \n\t Var: %s \n\t Std: %s" %(minlist, maxlist, meanlist, medianlist, varlist, stdlist)
		print "DQT2:"
		for i in range(0,8):
			myList = dqt2[i*8:i*8+8]
			print [ '%.2f' % elem for elem in myList ]

		print "rest DQT2:"
		print "\t Totalsum: %.2f, diagonalsum: %.2f, diagonalsum: %.2f, max: %.2f, min: %.2f" %(rest2[0], rest2[1], rest2[2], rest2[3], rest2[4] )
		print "\t Mean: %.2f, median: %.2f, var: %.2f, std: %.2f" %(rest2[5], rest2[6], rest2[7], rest2[8])
		for i in range(0,2):
			print "axis: %i" %(i)
			minlist = [ '%.2f' % elem for elem in rest2[9+56*i:17+56*i] ]
			maxlist = [ '%.2f' % elem for elem in rest2[17+56*i:25+56*i] ]
			meanlist = [ '%.2f' % elem for elem in rest2[25+56*i:32+56*i] ]
			medianlist = [ '%.2f' % elem for elem in rest2[32+56*i:40+56*i] ] 
			varlist = [ '%.2f' % elem for elem in rest2[40+56*i:48+56*i] ]
			stdlist =  [ '%.2f' % elem for elem in rest2[48+56*i:56+56*i] ]
			print "\tMin: %s \n\t Max: %s \n\t Mean: %s \n\t Median: %s \n\t Var: %s \n\t Std: %s" %(minlist, maxlist, meanlist, medianlist, varlist, stdlist)

	def train_hashfunction(self, hash_trainingset, class_trainingset):
		""" Create dictionary of hash functions. Return dictionary"""
		print "> Training hash function.."
		now = time.time()
		hashdict = {}
		for index in range(0,len(hash_trainingset)):
			hashvalue = hash_trainingset[index]

			if hashdict.get(hashvalue):
				values = hashdict.get(hashvalue)
				class_value = class_trainingset[index]	
				if class_value in values:
					continue
				else:
					values.append(class_value)
					hashdict[hashvalue] = values
			else:
				hashdict[hashvalue] = [class_trainingset[index]]
			"""
			if hashdict.get(hashvalue):		
				continue
			else:
				hashdict[hashvalue] = class_trainingset[index]
			"""
		print "H> old length hash dict: %i \n H> new length hash dict: %i" %(len(hash_trainingset), len(hashdict))
		print "..... Elapsed time HASH train: %.5f " %(time.time() - now)
		#counter = 0
		#for item in hashdict:
		#	counter += len(hashdict[item])
		#print "number of hashes with multiple options: %i" %(counter)
		return hashdict 

	def train_hashfunction_dumb(self, hash_trainingset, class_trainingset):
		""" Create dictionary of hash functions. Return dictionary"""
		print "> Training DUMB hash function.."
		now = time.time()
		hashdict = {}
		for index in range(0,len(hash_trainingset)):
			hashvalue = hash_trainingset[index]

			if hashdict.get(hashvalue):		
				continue
			else:
				hashdict[hashvalue] = class_trainingset[index]

		print "H> old length hash dict: %i \n H> new length hash dict: %i" %(len(hash_trainingset), len(hashdict))
		print "..... Elapsed time DUMB HASH train: %.5f " %(time.time() - now)
		return hashdict 


	def test_hashfunction_dumb(self, hash_testset, class_testset, hashdict):
		""" Test hash functionwith test set. Calculate precision, recall"""
		now = time.time()
		print "H> Testing DUMB hash function.."
		predictions = []
		for item in hash_testset:
			# return -1 when unknown
			predict = hashdict.get(item, -1)
			# choose something random
			if predict == -1:
				possiblevalues = hashdict.values()
				predict = random.choice(possiblevalues)
			predictions.append(predict)
		
		# number of differences:
		print "..... Elapsed time DUMBHASH test: %.5f " %(time.time() - now)
		return predictions

	def test_hashfunction(self, hash_testset, class_testset, hashdict):
		""" Test hash functionwith test set. Calculate precision, recall"""
		now = time.time()
		print "H> Testing hash function.."
		predictions = []
		"""
		for item in hash_testset:
			# return -1 when unknown
			predict = hashdict.get(item, -1)
			# choose something random
			if predict == -1:
				possiblevalues = hashdict.values()
				predict = random.choice(possiblevalues)
			predictions.append(predict)
		"""
		self.predictions_wrong = {}
		self.false_positives = {}
		for index in range(0,len(hash_testset)):
			predict = hashdict.get(hash_testset[index], -1)
			if predict == -1:
				possiblevalues = range(0,len(self.class_to_int_dict))
				predicted = random.choice(possiblevalues)
				predictions.append(predicted)
			else:
				true = class_testset[index]
				#choose right possibility
				if true in predict:
					predictions.append(true)
					self.false_positives[true] = self.false_positives.get(true,0) +1
				else:
					predicted = random.choice(predict)
					predictions.append(predicted)
				for item in predict:
					if item != true:
						self.predictions_wrong[item] = self.predictions_wrong.get(item,0) +1
		# number of differences:
		print "..... Elapsed time HASH test: %.5f " %(time.time() - now)
		return predictions	

	def train_decisiontree(self, feature_trainingset, class_trainingset):	
		""" Train decision tree with training set. Return classifier"""
		now = time.time()
		print "DT> Training decision tree.."
		# fit classifier
		clf3 = tree.DecisionTreeClassifier()
		clf3.fit(feature_trainingset, class_trainingset)
		print "..... Elapsed time DT train: %.5f " %(time.time() - now)
		return clf3

		#print "Accuracy: %0.2f " % clf3.score(X_train, y_train)
		#scores = cross_validation.cross_val_score(clf3, train, target, cv=15)
		#print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)   


	def test_decisiontree(self, dt_testset, dt_class_testset, clf):
		""" Test decision tree with test set. Calculate precision, recall"""
		now = time.time()
		print "DT> Testing decision tree function.."
		predictions = clf.predict(dt_testset)
		print "..... Elapsed time DT test: %.5f " %(time.time() - now)
		return predictions
		

	def get_camera_id(self, key):
		"""
		Returns ID for camera model in class list.
		Looks up in table and if not present, creates one by adding 1 to max.
		"""
		if self.class_to_int_dict.get(key):
			return self.class_to_int_dict[key]
		else:
			if len(self.class_to_int_dict) == 0:
				self.class_to_int_dict[key] = 0
				return 0
			else:
				# return max number + 1 and add to dictionary
				camera_id = max(self.class_to_int_dict.iteritems(), key=operator.itemgetter(1))[1] +1
				self.class_to_int_dict[key] = camera_id
				return camera_id
			

	def convert_one(self, dqts):
		"""
		Convert quantizationtable to features. A feature is an array of values
		"""
		dqt_features = []
		for index in range(0,len(dqts)):
			# flatten dqt
			dqt = list(itertools.chain.from_iterable(dqts[index]))
			dqt_features.extend(dqt)
			"""
			dqt_np = np.array(dqt)
			# extra features
			totalsum = sum(dqt)
			dqt_features.append(totalsum)	# total sum
			dqt_features.append(sum([r[i] for i, r in enumerate(dqts[index])]))	#diagonal sum L-> R 
			dqt_features.append(sum([r[-i-1] for i, r in enumerate(dqts[index])])) #diagonal sum R -> L
			dqt_features.append(max(dqt))	# max of all values
			dqt_features.append(min(dqt))	# min of all values

			mean = float('%.3f' %(np.mean(dqt)))
			median = float('%.3f' %(np.median(dqt)))
			var = float('%.3f' %(np.var(dqt)))
			std = float('%.3f' %(np.std(dqt)))

			dqt_features.append(mean)	# mean of all values
			dqt_features.append(median)	# median
			dqt_features.append(var)	# variance
			dqt_features.append(std)	# standard deviation

			# axis=0 : over columns, axis=1 : over rows
			for i in range(0,2):
				dqt_features.extend(np.amin(dqts[index], axis=i).tolist()) 
				dqt_features.extend(np.amax(dqts[index], axis=i).tolist()) 
				dqt_features.extend(np.mean(dqts[index], axis=i).tolist())
				dqt_features.extend(np.median(dqts[index], axis=i).tolist())
				dqt_features.extend(np.var(dqts[index], axis=i).tolist())
				dqt_features.extend(np.std(dqts[index], axis=i).tolist())

			"""
		return dqt_features


	def get_lineage(self, tree):
	     left      = tree.tree_.children_left
	     right     = tree.tree_.children_right
	     threshold = tree.tree_.threshold


	     # get ids of child nodes
	     idx = np.argwhere(left == -1)[:,0]     

	     def recurse(left, right, child, lineage=None):          
		  if lineage is None:
		       lineage = [child]
		  if child in left:
		       parent = np.where(left == child)[0].item()
		       split = 'l'
		  else:
		       parent = np.where(right == child)[0].item()
		       split = 'r'

		  lineage.append((parent, split, threshold[parent]))

		  if parent == 0:
		       lineage.reverse()
		       return lineage
		  else:
		       return recurse(left, right, parent, lineage)

	     for child in idx:
		  for node in recurse(left, right, child):
		       print node

test = Create_features(multiload=True, dump=False, load=False)
test.run()


