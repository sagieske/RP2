for image in *.JPG
do
outputfilename=$image.output
wine JPEGsnoop.exe -i $image -o $outputfilename -nogui
python parsefile.py $outputfilename
done

