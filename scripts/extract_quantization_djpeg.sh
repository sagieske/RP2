#!/bin/bash

# Get time as a UNIX timestamp (seconds elapsed since Jan 1, 1970 0:00 UTC)
T="$(date +%s)"

for image in /images/*.JPG
do
outputfilename=$image.output
djpeg -verbose -verbose $image 2>&1 | cat > $outputfilename
python parsefile_djpeg.py $outputfilename
done

# Print elapsed time
T="$(($(date +%s)-T))"
echo "Time in seconds: ${T}"

printf "Pretty format: %02d:%02d:%02d:%02d\n" "$((T/86400))" "$((T/3600%24))" "$((T/60%60))" "$((T%60))""

