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

class Create_features(object):

	FILEPATTERN = '/images/*.output.djpeg-dqt'
	#FILEPATTERN = '/tmp/*'
	ITEMS = []
	camera_dict = {}
	feature_dict = {}

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
		"""
		Extract camera make, model and the dqt	
		"""
		camerainfo = infotuple[0]
		try:
			camera = re.sub('/images/','',camerainfo[0])
			identifier = (camera, camerainfo[1])
			if identifier in self.camera_dict:
				if self.camera_dict[identifier] == [infotuple[1], infotuple[2]]:
					print "TRUE"
				else:
					print "FALSE"
			else:
				print "ADD"
				self.camera_dict[identifier] = [infotuple[1], infotuple[2]]
		except:
			print "problem!"

	def convert_to_features(self):
		"""
		For all items in camera dictionary convert quantizationtable to features
		"""	
		featurelist = []
		classlist = []
		# do loop
		featurelist.append()
		pass

	def convert_one(self, dqts):
		"""
		Convert quantizationtable to features. A feature is an array of values
		"""
		dqt_features = []
		# perform
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


test = Create_features()
test.create_patterns()