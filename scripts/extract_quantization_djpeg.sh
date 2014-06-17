#!/bin/bash

# Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
T="$(date +%s)"
COUNTER=0

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

