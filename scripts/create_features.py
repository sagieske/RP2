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
		for name in files:
			infotuple = pickle.load( open( name, "rb" ) )
			#try:
			#	camera = re.sub('/images/','',infotuple[0][0])
			#except:
			#	print "nooooo %s" %name
			#self.ITEMS.append(infotuple)
			self.create_dictionary(infotuple)

	def create_dictionary(self, infotuple):
		""" Extract camera make, model and the dqts """
		camerainfo = infotuple[0]
		try:
			camera = re.sub('/images/','',camerainfo[0])
			identifier = (camera, camerainfo[1])
			dqts = [infotuple[1], infotuple[2]]
			# known make & model
			if identifier in self.camera_dict:
				# value already in dictionary
				if dqts in self.camera_dict[identifier]:
					pass
				# new value, same identifier
				else:
					value = self.camera_dict[identifier]
					self.camera_dict[identifier] = value.append((dqts,))

			# new make & model
			else:
				self.camera_dict[identifier] = (dqts,)
		except:
			print "problem!"

	def convert_to_features(self):
		"""
		For all items in camera dictionary convert quantizationtable to features
		"""	
		featurelist = []
		classlist = []
		# do for every camera make & model in dictionary
		for key, value in self.camera_dict.iteritems():
			# for every different dqt for this camera make & model
			for dqt in value:
				featurelist.append(self.convert_one(dqt))
				classlist.append(self.get_camera_id(key[0]))
		print len(featurelist)
		print len(classlist)


	def get_camera_id(self, key):
		"""
		Returns ID for camera model in class list.
		Looks up in table and if not present, creates one by adding 1 to max.
		"""
		if self.class_to_int_dict[key]:
			return self.class_to_int_dict[key]
		else:
			# return max number + 1 and add to dictionary
			camera_id = max(self.class_to_int_dict.iteritems(), key=operator.itemgetter(1))[0] +1
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
			dqt_features.append(np.average(dqt))	# mean of all values
			dqt_features.append(np.median(dqt))	# median
			dqt_features.append(np.var(dqt))	# variance
			dqt_features.append(np.std(dqt))	# standard deviation

		return dqt_features

test = Create_features()
test.convert_to_features()
