for image in *.JPG
do
outputfilename=$image.output
djpeg -verbose -verbose $image 2>&1 | cat > $outputfilename
python parsefile_djpeg.py $outputfilename
done

