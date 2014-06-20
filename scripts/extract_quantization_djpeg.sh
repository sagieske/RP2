#!/bin/bash

# Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
T="$(date +%s)"
COUNTER=0
#for image in /images/Nikon_D70_0_19761.JPG /images/Nikon_D70s_1_22980.JPG /images/Nikon_D70_1_20761.JPG /images/Praktica_DCZ5.9_2_34586.JPG /images/Nikon_D70s_1_23088.JPG /images/Nikon_D70_1_20891.JPG /images/Nikon_D70_1_20889.JPG /images/Nikon_D70s_0_22085.JPG /images/Nikon_D70_1_20771.JPG /images/Nikon_D70s_0_21983.JPG /images/Nikon_D70s_0_22011.JPG /images/Olympus_mju_1050SW_4_25619.JPG /images/Praktica_DCZ5.9_2_34587.JPG /images/Nikon_D70_0_19665.JPG /images/Nikon_D70s_1_22972.JPG /images/Nikon_D70s_0_21999.JPG /images/Olympus_mju_1050SW_4_25618.JPG /images/Nikon_D70s_0_22083.JPG /images/Nikon_D70_1_20759.JPG /images/Nikon_D70s_0_22005.JPG /images/Nikon_D70_0_19759.JPG /images/Nikon_D70s_1_23090.JPG /images/Nikon_D70_0_19671.JPG /images/Nikon_D70s_1_22986.JPG /images/Nikon_D70s_0_21987.JPG /images/Nikon_D70s_0_22087.JPG /images/Nikon_D70_1_20777.JPG /images/Nikon_D70s_1_22974.JPG

for image in /images/*.JPG
do
(( COUNTER++ ))
outputfilename=$image.output
finalname=$outputfilename.djpeg-dqt
if [ ! -f $outputfilename ]; then
	djpeg -verbose -verbose $image 2>&1 | cat > $outputfilename
	# wait until finished
	wait
	python parsefile_djpeg.py $outputfilename
	wait
	rm $outputfilename
fi

if [ $(( $COUNTER % 500 )) -eq 0 ] ; then
	echo $COUNTER
fi
done

# Print elapsed time
T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"

printf "Pretty format: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"

