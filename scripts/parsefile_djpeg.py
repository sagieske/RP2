import re
import pickle
import sys
from sys import argv
import os.path

INPUTFILE = argv[1]
OUTPUTFILE = INPUTFILE + '.djpeg-dqt'

# File already exists, stop! (should already be handled by shell script)
if os.path.isfile(OUTPUTFILE):
	sys.exit()

# extract camera information from title
filename_camera = INPUTFILE.split('/')[-1]
camerainfo = re.sub('.JPG.output', '',filename_camera).split('_')
manufacturer = camerainfo[0]
cameramodel = camerainfo[1]
deviceid = camerainfo[2]
uniqueID = camerainfo[3]
camerainfo = (manufacturer, cameramodel, deviceid, uniqueID)

# read file
try:
	with open(INPUTFILE) as f:
	    data=f.read(1600)

	# get all dqt rows
	#dqt_values = re.findall(r'\s{2,4}\d{1,3}\s', data)
	# so ugly, but can't find another way
	dqt_values = re.findall(r'\d{1,3}\s{1,4}\d{1,3}\s{1,4}\d{1,3}\s{1,4}\d{1,3}\s{1,4}\d{1,3}\s{1,4}\d{1,3}\s{1,4}\d{1,3}\s{1,4}\d{1,3}', data)
	# only extract & dump if values are found
	if dqt_values:
		# extract tables to array
		dqt1 = []
		dqt2 = []
		# extract only first 16 rows
		for j in range(0,16):
			#print dqt_values[j]
			# strip all spaces and sub for comma
			stripped_string = re.sub(' +', ',', dqt_values[j].lstrip())
			# map to ints
			row = map(int, stripped_string.split(","))
			#row = map(int, dqt_values[j*8:j*8+8])
			if j < 8:
				dqt1.append(row)
			else:
				dqt2.append(row)
		# pickle dump camera data and quantization tables
		pickle.dump( (camerainfo, dqt1, dqt2), open( OUTPUTFILE, "wb" ) )
	else:
		print "no DQT found for file %s" %(INPUTFILE)
except: 
	print "no such file"



