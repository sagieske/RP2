for image in *.JPG
do
outputfilename=$image.output
wine JPEGsnoop.exe -i $image -o $outputfilename -nogui
python parsefile_jpegsnoop.py $outputfilename
done

