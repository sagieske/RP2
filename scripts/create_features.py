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
from sklearn.ensemble import ExtraTreesClassifier
import operator
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
from sklearn import cross_validation

import hashlib


class Create_features(object):

	FILEPATTERN = '/images/*.output.djpeg-dqt'
	#FILEPATTERN = '/tmp/*'
	ITEMS = []
	camera_dict = {}
	feature_dict = {}
	class_to_int_dict = {}

	def __init__(self):
		self.load_files()

	def load_files(self):
		"""
		Load in all files with specified file pattern. Append infotuple to ITEMS array
		"""
		files = glob.glob(self.FILEPATTERN)
		counter = 0
		for name in files:
			try:
				infotuple = pickle.load( open( name, "rb" ) )
				self.create_dictionary(infotuple)
				counter += 1
				if counter % 1000 == 0:
					print counter
				#if counter == 1000:
				#	break
			except:
				print "problem with file %s" %(name)
		print "finished dictionary"

	def create_dictionary(self, infotuple):
		""" Extract camera make, model and the dqts """
		camerainfo = infotuple[0]
		try:
			camera = re.sub('/images/','',camerainfo[0])
			identifier = (camera, camerainfo[1])
			for j in range(1,3):
				for i in infotuple[j]:
					if len(i) != 8:
						raise Exception("problem for %s" %(infotuple[0]))
			dqts = [infotuple[1], infotuple[2]]
			# known make & model
			if identifier in self.camera_dict:
				# value already in dictionary
				#if dqts in self.camera_dict[identifier]:
				#	pass
				# new value, same identifier
				#else:
				value = self.camera_dict[identifier]
				self.camera_dict[identifier] = value + (dqts,)

			# new make & model
			else:
				self.camera_dict[identifier] = (dqts,)
		except:
			print "problem! %s, %s, %s" %(infotuple[0], infotuple[1], infotuple[2])


	def run(self):
		"""
		Convert dqt to feature or hash and their class list. Create train and test sets. Run training and predictions
		"""
		# Convert dictionary to feature sets
		h_featurelist, h_classlist = self.create_hash_set()
		dt_featurelist, dt_classlist = self.create_dt_feature_set()

		# create train and test sets
		h_X_train, h_X_test, h_y_train, h_y_test = cross_validation.train_test_split(h_featurelist, h_classlist, test_size=0.3, random_state=42)
		dt_X_train, dt_X_test, dt_y_train, dt_y_test = cross_validation.train_test_split(dt_featurelist, dt_classlist, test_size=0.3, random_state=42)
		hashdict = self.train_hashfunction(h_X_train, h_X_test)
		dt_clf = self.train_decisiontree(dt_X_train, dt_X_test)


	def create_hash_set(self):
		"""
		For all items in camera dictionary convert quantizationtable to hashes. Return hashes and their class
		"""
		featurelist = []
		classlist = []
		# do for every camera make & model in dictionary
		for key, value in self.camera_dict.iteritems():
			# for every different dqt for this camera make & model
			for dqt in value:
				hash_object = hashlib.sha256(pformat(dqt))
				featurelist.append(hash_object)
				classlist.append(self.get_camera_id(key))
		print "HASH> Length featurelist: %i \n > Length classlist: %i"  %(len(featurelist),len(classlist))
		return featurelist, classlist

	def create_dt_feature_set(self):
		"""
		For all items in camera dictionary convert quantizationtable to features. Return feature array and their class
		"""	
		featurelist = []
		classlist = []
		# do for every camera make & model in dictionary
		counter = 0
		for key, value in self.camera_dict.iteritems():
			# for every different dqt for this camera make & model
			for dqt in value:
				featurelist.append(self.convert_one(dqt))
				classlist.append(self.get_camera_id(key))
				counter += 1
				#if counter % 100 == 0:
				#	print "COUNTER AT: %i" %(counter)
		print "DT> Length featurelist: %i \n > Length classlist: %i"  %(len(featurelist),len(classlist))

		# feature selection
		smaller_featurelist = self.feature_selection(featurelist, classlist)

		return smaller_featurelis, classlist

	def feature_selection(self, X, y):
		""" Perform feature selection on feature set. Return modified feature set """
		clf = ExtraTreesClassifier()
		X_new = clf.fit(X, y).transform(X) 
		return X_new
		#print clf.feature_importances_ 
		#print X_new.shape
		#print y
		#for i in range(0,10):
		#	print X[i]
		#	print y[i]

	def train_hashfunction(self, hash_trainingset, class_trainingset):
		""" Create dictionary of hash functions. Return dictionary"""
		hashdict = {}
		for index in range(0,len(hash_trainingset)):
			hashdict[hash_trainingset[index]] = class_trainingset[index]
		print "H> length hash dict: %i" %(len(hashdict))
		return hashdict 

	def train_decisiontree(self, feature_trainingset, class_trainingset):	
		""" Train decision tree with training set. Return classifier"""
		# fit classifier
		clf3 = tree.DecisionTreeClassifier()
		clf3.fit(feature_trainingset, class_trainingset)
		return clf3

		#print "Accuracy: %0.2f " % clf3.score(X_train, y_train)
		#scores = cross_validation.cross_val_score(clf3, train, target, cv=15)
		#print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)   



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
			dqt_np = np.array(dqt)
			# extra features
			totalsum = sum(dqt)
			dqt_features.append(totalsum)	# total sum
			dqt_features.append(sum([r[i] for i, r in enumerate(dqts[index])]))	#diagonal sum L-> R 
			dqt_features.append(sum([r[-i-1] for i, r in enumerate(dqts[index])])) #diagonal sum R -> L
			dqt_features.append(max(dqt))	# max of all values
			dqt_features.append(min(dqt))	# min of all values
                        dqt_features.extend(np.amin(dqts[index], axis=1).tolist())
                        dqt_features.extend(np.amin(dqts[index], axis=0).tolist())
                        dqt_features.extend(np.amax(dqts[index], axis=1).tolist())
                        dqt_features.extend(np.amax(dqts[index], axis=0).tolist())

			mean = float('%.3f' %(np.mean(dqt)))
			median = float('%.3f' %(np.median(dqt)))
                        var = float('%.3f' %(np.var(dqt)))
                        std = float('%.3f' %(np.std(dqt)))

			dqt_features.append(mean)	# mean of all values
			dqt_features.append(median)	# median
			dqt_features.append(var)	# variance
			dqt_features.append(std)	# standard deviation
			dqt_features.extend(np.mean(dqts[index], axis=1).tolist())
			dqt_features.extend(np.mean(dqts[index], axis=0).tolist())
                        dqt_features.extend(np.median(dqts[index], axis=1).tolist())
                        dqt_features.extend(np.median(dqts[index], axis=0).tolist())
                        dqt_features.extend(np.var(dqts[index], axis=1).tolist())
                        dqt_features.extend(np.var(dqts[index], axis=0).tolist())
                        dqt_features.extend(np.std(dqts[index], axis=1).tolist())
                        dqt_features.extend(np.std(dqts[index], axis=0).tolist())

		return dqt_features

test = Create_features()
#test.create_dt_feature_set()
test.run()
