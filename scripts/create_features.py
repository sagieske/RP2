"""
TODO:
- for every item perform feature creation
- save
"""
import glob
import pickle
import re
import itertools	# for flattening list
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
import operator
from sklearn import tree

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
				#try:
				#	camera = re.sub('/images/','',infotuple[0][0])
				#except:
				#	print "nooooo %s" %name
				#self.ITEMS.append(infotuple)
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
				if dqts in self.camera_dict[identifier]:
					pass
				# new value, same identifier
				else:
					value = self.camera_dict[identifier]
					self.camera_dict[identifier] = value + (dqts,)

			# new make & model
			else:
				self.camera_dict[identifier] = (dqts,)
		except:
			print "problem! %s, %s, %s" %(infotuple[0], infotuple[1], infotuple[2])

	def convert_to_features(self):
		"""
		For all items in camera dictionary convert quantizationtable to features
		"""	
		featurelist = []
		classlist = []
		# do for every camera make & model in dictionary
		counter = 0
		#counter_total = 0
		# DEBUG calculate length for each input:
		#for item in self.camera_dict.values():
		#	for a in item:
		#		counter_total +=1
		#print "> Length camera_dict: %i" %(counter_total) 
		#%(len(self.camera_dict.values()))
		for key, value in self.camera_dict.iteritems():
			# for every different dqt for this camera make & model
			for dqt in value:
				featurelist.append(self.convert_one(dqt))
				classlist.append(self.get_camera_id(key[0]))
				counter += 1
				if counter % 100 == 0:
					print "COUNTER AT: %i" %(counter)
		print "> Length featurelist: %i \n > Length classlist: %i"  %(len(featurelist),len(classlist))
		self.feature_selection(featurelist, classlist)
		print self.class_to_int_dict

	def feature_selection(self, X, y):
		clf = ExtraTreesClassifier()
		X_new = clf.fit(X, y).transform(X) 
		print clf.feature_importances_ 
		print X_new.shape
		print y
		#for i in range(0,10):
		#	print X[i]
		#	print y[i]
		clf2 = tree.DecisionTreeClassifier()
		clf2 = clf2.fit(X_new, y)

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
			
			average = float('%.3f' %(np.average(dqt)))
			median = float('%.3f' %(np.median(dqt)))
                        var = float('%.3f' %(np.var(dqt)))
                        std = float('%.3f' %(np.std(dqt)))

			dqt_features.append(average)	# mean of all values
			dqt_features.append(median)	# median
			dqt_features.append(var)	# variance
			dqt_features.append(std)	# standard deviation

		return dqt_features

test = Create_features()
test.convert_to_features()
