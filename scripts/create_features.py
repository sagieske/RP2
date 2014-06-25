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



class Create_features(object):

	# COMMENT FOR NFI
	STARTPATH = '/images/'
	#UNCOMMENT FOR NFI
	#STARTPATH = '/home/sharon/Documents/test/'
	FILEPATTERN = STARTPATH + '*.output.djpeg-dqt'
	PICKLEFILE = STARTPATH + 'server-dictionary.pickle'
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
		for key,value in self.camera_dict.iteritems():
			print "%s : %i" %(key, len(value))


	def create_dictionary(self, infotuple):
		""" Extract camera make, model and the dqts """
		camerainfo = infotuple[0]
		try:
			# UNCOMMENT FOR NFI	
			#identifier = self.get_identifier(camerainfo)
			
			# identifier is camera make and model
			# COMMENT FOR NFI
			cameramake = re.sub(self.STARTPATH, '', camerainfo[0])
			identifier = (cameramake, camerainfo[1])
			
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
		for key, value in self.camera_dict.iteritems():
			# for every different dqt for this camera make & model
			for dqtset in value:
				classlist.append(self.get_camera_id(key))
				h_featurelist.append(self.get_hash(dqtset))
				dt_featurelist.append(self.create_dt_feature_set(dqtset))
		print len(h_featurelist)

		# feature selection dt set
		dt_featurelist_small = self.feature_selection(dt_featurelist, classlist)
		print "DT> new shape:"
		print dt_featurelist_small.shape


		print "Creating train and test sets.."
		# create indices for train and test sets
		split = cross_validation.ShuffleSplit(len(classlist), test_size=.3, random_state=13)
		for i in split:
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
		
		#h_X_train, h_X_test, h_y_train, h_y_test = cross_validation.train_test_split(h_featurelist, classlist, test_size=0.3, random_state=42)
		#dt_X_train, dt_X_test, dt_y_train, dt_y_test = cross_validation.train_test_split(dt_featurelist_small, classlist, test_size=0.3, random_state=42)
		#print len(h_X_test)

		# training
		print "Start training.."
		hashdict = self.train_hashfunction(h_X_train, y_train)
		dt_clf = self.train_decisiontree(dt_X_train, y_train)

		# fit
		print "Start prediction.."
		predictions_hash = self.test_hashfunction(h_X_test, y_test, hashdict)
		predictions_dt = self.test_decisiontree(dt_X_test, y_test, dt_clf)

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
		return X_new

	def train_hashfunction(self, hash_trainingset, class_trainingset):
		""" Create dictionary of hash functions. Return dictionary"""
		print "> Training hash function.."
		now = time.time()
		hashdict = {}
		for index in range(0,len(hash_trainingset)):
			hashvalue = hash_trainingset[index]
			hashdict[hashvalue] = class_trainingset[index]
		print "H> old length hash dict: %i \n H> new length hash dict: %i" %(len(hash_trainingset), len(hashdict))
		print "..... Elapsed time: %.5f " %(time.time() - now)
		return hashdict 

	def test_hashfunction(self, hash_testset, class_testset, hashdict):
		""" Test hash functionwith test set. Calculate precision, recall"""
		now = time.time()
		print "> Testing hash function.."
		predictions = []
		for item in hash_testset:
			# return -1 when unknown
			predict = hashdict.get(item, -1)
			predictions.append(predict)
		# number of differences:
		diff = sum(1 for i, j in zip(predictions, class_testset) if i != j)
		print "H> Precision"
		print sklearn.metrics.precision_score(class_testset, predictions)
		print "H> Recall"
		print sklearn.metrics.recall_score(class_testset, predictions)
		print "..... Elapsed time: %.5f " %(time.time() - now)
		return predictions	

	def train_decisiontree(self, feature_trainingset, class_trainingset):	
		""" Train decision tree with training set. Return classifier"""
		now = time.time()
		print "> Training decision tree.."
		# fit classifier
		clf3 = tree.DecisionTreeClassifier()
		clf3.fit(feature_trainingset, class_trainingset)
		print "..... Elapsed time: %.5f " %(time.time() - now)
		return clf3

		#print "Accuracy: %0.2f " % clf3.score(X_train, y_train)
		#scores = cross_validation.cross_val_score(clf3, train, target, cv=15)
		#print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)   


	def test_decisiontree(self, dt_testset, dt_class_testset, clf):
		""" Test decision tree with test set. Calculate precision, recall"""
		now = time.time()
		print "> Testing decision tree function.."
		predictions = clf.predict(dt_testset)
		print "DT> Precision:"
		print sklearn.metrics.precision_score(dt_class_testset, predictions)
		print "DT> Recall:"
		print sklearn.metrics.recall_score(dt_class_testset, predictions)
		print "..... Elapsed time: %.5f " %(time.time() - now)
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

		return dqt_features

test = Create_features(multiload=False, dump=True, load=False)
test.run()

