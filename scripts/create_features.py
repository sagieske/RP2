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
	camera_dict = {}

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
			self.create_patterns(infotuple)

	def create_patterns(self, infotuple):
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

test = Create_features()
test.create_patterns()
