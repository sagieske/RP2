#!/bin/bash
# stupid spaces
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

# Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
T="$(date +%s)"
COUNTER=0

# get all filenames with jpg extension
imagedatabase=/media/sharon/_MEDIA_NFI-/PRNU_Compare_Image_Database/Database/
for image in $(find $imagedatabase -name '*.jpg'  -exec echo {} \;);
do
(( COUNTER++ ))

# rename file to process brackets
filename_brackets1=${image//\(/}
filename_brackets2=${filename_brackets1//\)/}

# delete / and preceeding dot, append to output folder to create output file.
filename_withoutfolders=${filename_brackets2//\//.}
outputname_ready=`echo $filename_withoutfolders | cut -c 2-`
outputfilename=/home/sharon/Documents/test/$outputname_ready.output

# if already processed with python:
finalname=$outputfilename.djpeg-dqt

if [ ! -f $finalname ]; then
	djpeg -verbose -verbose $filename_brackets2 2>&1 | cat > $outputfilename
	# wait until finished
	wait
	python /home/sharon/Documents/SNE/RP2/scripts/parsefile_djpeg_nfi.py $outputfilename
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
IFS=$SAVEIFS

