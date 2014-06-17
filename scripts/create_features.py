"""
TODO:
- for every item perform feature creation
- save
"""
import glob
import pickle
import re

class Create_features(object):

	FILEPATTERN = '/images/*.output.djpeg-dqt'
	#FILEPATTERN = '/tmp/*'
	ITEMS = []

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
			self.ITEMS.append(infotuple)

	def create_patterns(self):
		"""
		Extract camera make, model and the dqt	
		"""
		camera_dict = {}
		for item in self.ITEMS:
			camerainfo = item[0]
			print camerainfo
			try:
				camera = re.sub('/images/','',camerainfo[0])
				identifier = (camera, camerainfo[1])
				if identifier in camera_dict:
					if camera_dict[identifier] == [item[1], item[2]]:
						print "TRUE"
					else:
						print "FALSE"
				else:
					camera_dict[identifier] = [item[1], item[2]]
			except:
				print "problem!"

test = Create_features()
test.create_patterns()
