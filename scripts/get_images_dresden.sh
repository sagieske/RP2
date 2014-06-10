sudo mkdir /images/
for file in ../imagedb/*;
do
for url in $(cat $file);
do 
wget -P /images/ $url
done
done
