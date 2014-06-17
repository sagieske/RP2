for image in /images/*.JPG
do
outputfilename=$image.output
if [ ! -f $outputfilename ]; then
	djpeg -verbose -verbose $image 2>&1 | cat > $outputfilename
	# wait until finished
	wait
	python parsefile_djpeg.py $outputfilename &
fi
done

