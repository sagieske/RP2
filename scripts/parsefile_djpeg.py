import re
import pickle
import sys
from sys import argv

INPUTFILE = argv[1]
OUTPUTFILE = INPUTFILE + '.djpeg-dqt'

# extract camera information from title
camerainfo = re.sub('.JPG', '',INPUTFILE).split('_')
manufacturer = camerainfo[0]
cameramodel = camerainfo[1]
deviceid = camerainfo[2]
uniqueID = camerainfo[3]

# read file
with open(INPUTFILE) as f:
    data=f.read(1500)

# get all dqt rows
dqt_values = re.findall(r'\s{3,4}\d{1,2}', data)

# only extract & dump if values are found
if dqt_values:
	# extract tables to array
	dqt1 = []
	dqt2 = []
	for j in range(0,16):
		row = map(int, dqt_values[j*8:j*8+8])
		if j < 9:
			dqt1.append(row)
		else:
			dqt2.append(row)
	# pickle dump camera data and quantization tables
	pickle.dump( (camerainfo, dqt1, dqt2), open( OUTPUTFILE, "wb" ) )




