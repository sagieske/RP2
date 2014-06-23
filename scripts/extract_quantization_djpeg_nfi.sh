#!/bin/bash
# stupid spaces
SAVEIFS=$IFS
IFS=$(echo -en "\n\b")

# Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
T="$(date +%s)"
COUNTER=0

# get all filenames with jpg extension
imagedatabase=/media/sharon/My\ Book/PRNU\ Compare\ Image\ Database/Database/
for image in $(find $imagedatabase -name '*.jpg'  -exec echo {} \;);
do
(( COUNTER++ ))
#real_path=$(readlink -e $image)
#echo $real_path
# rename file to delete preceeding dot and space for output files
filename_nospaces=${image// /_}
filename_withoutfolders=${filename_nospaces//\//.}
filename_brackets1=${filename_withoutfolders//\(/}
filename_brackets2=${filename_brackets1//\)/}
outputname_ready=`echo $filename_brackets2 | cut -c 2-`

# delete prepending dot for djpeg process
filename_spaceslinuxstyle=${image// /\\ }
filename_spacesbrackets1linuxstyle=${filename_spaceslinuxstyle//\(/\\(}
filename_spacesbrackets2linuxstyle=${filename_spacesbrackets1linuxstyle//\)/\\)}

#echo $filename_spacesbrackets2linuxstyle

outputfilename=/home/sharon/Documents/test/$outputname_ready.output
#echo $outputfilename
finalname=$outputfilename.djpeg-dqt

if [ ! -f $finalname ]; then
	echo djpeg -verbose -verbose $filename_spacesbrackets2linuxstyle 2>&1 | cat > $outputfilename
	# wait until finished
	#wait
	#python /home/sharon/Documents/SNE/RP2/scripts/parsefile_djpeg_nfi.py $outputfilename
	#wait
	#rm $outputfilename
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

