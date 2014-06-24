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

### EXAMPLE /home/sharon/Documents/test/media.sharon.My_Book.PRNU_Compare_Image_Database.Database.Samsung_ST30.Special_images.Facebook_(standard_compression).Reference.02.04_526234_103259413159326_707335452_n.jpg.output
# extract camera information from title
total_filename_camera = re.sub('.jpg.output', '',INPUTFILE).split('/')
sub_filename_camera = total_filename_camera[5].split('.')
camera_type = sub_filename_camera[5]
manufacturer = camera_type.split('_')[0]
cameramodel = '-'.join(camera_type.split('_')[1:])
folderid = '-'.join(sub_filename_camera [6:])
camerainfo = (manufacturer, cameramodel, folderid)

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



