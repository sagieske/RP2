#!/bin/bash
# stupid spaces
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

# Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
T="$(date +%s)"
COUNTER=0

# get all filenames with jpg extension
cd /media/sharon/My\ Book/PRNU\ Compare\ Image\ Database/Database/
for image in $(find . -name '*.jpg' -exec echo {} \;);
do
#echo $image
(( COUNTER++ ))
# rename file to delete preceeding dot and space for output files
filename_nospaces=${image// /_}
filename_withoutdot=`echo $filename_nospaces | cut -c 3-`
echo $filename_withoutdot
outputfilename=/home/sharon/Documents/test/$filename_withoutdot.output
finalname=$outputfilename.djpeg-dqt
#echo $outputfilename
#if [ ! -f $finalname ]; then
	#cd /media/sharon/My\ Book/PRNU\ Compare\ Image\ Database/Database/
#	djpeg -verbose -verbose $filename_withoutdot 2>&1 | cat > $outputfilename
	# wait until finished
#	wait
#	python /home/sharon/Documents/SNE/RP2/scripts/parsefile_djpeg_nfi.py $outputfilename
#	wait
#	rm $outputfilename
#fi

if [ $(( $COUNTER % 500 )) -eq 0 ] ; then
	echo $COUNTER
fi
done

# Print elapsed time
T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"

printf "Pretty format: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))"
IFS=$SAVEIFS

