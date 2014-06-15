import re
import pickle
from sys import argv

INPUTFILE = argv[1]
OUTPUTFILE = INPUTFILE + '.dqt'
# read file
with open(INPUTFILE) as f:
    data=f.read()

# get all dqt rows
dqt_values = re.findall(r'DQT, Row.*\n', data)

if dqt_values:
	# extract tables to array
	for i in range(0,8):
		dqt1 = map(int, dqt_values[i].split()[3:])
	for i in range(8,16):
		dqt2 = map(int,dqt_values[i].split()[3:])	
	pickle.dump( (dqt1, dqt2), open( OUTPUTFILE, "wb" ) )




