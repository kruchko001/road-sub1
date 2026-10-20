# This script allows downloading ALL the raw images one by one.
# To avoid flooding our server, we ensure that images are downloaded only once (not downloading alreading existing files)
# The following variable needs to be adjusted according to your settings:
# First the variable "ALASKA_list_path" specifies the path of the file which contains the list of of images' URL (default, this file is in the present script's directory)
ALASKA_list_path="./list_RAWs_ALASKA_v2.txt"

# More important, the variable "Image_local_path" tells where the image will be downloaded; beware that this path will also be used in the script "ALASKA_conversion.sh", which generate developement of RAW files into JPEG images.
# Default setting is to put the RAW images in the ALASKA_conversion.sh script default directory for RAW.
Image_local_path="../Conversion_script/devHome/ALASKA_v2_RAWs/"

# Begining of the script; timestamp and creation of non-existing directory.
timeStart=$(date +%s)
if [ ! -d $Image_local_path ]; then echo "Directory $Image_local_path where RAW images will be downloaded does not exist. It will be created" ; mkdir -p $Image_local_path; fi
if [ ! -d ./tmp ]; then mkdir -p ./tmp ; fi

# Begining of the download, reading each line of the file "ALASKA_list_RAWs" and calling the software wget to download the raw image from its URL.
image_index=0
NB_image_to_DL=$(wc -c $ALASKA_list_path)
while read imageName 
do
    imageURL="http://alaska.utt.fr/DATASETS/ALASKA_v2_RAWs/$imageName"
    #[ -f ./tmp/$imageName ] && rm ./tmp/$imageName
    if [ ! -f  $Image_local_path$imageName ]
    then
        ( wget -c -P ./tmp/ $imageURL && mv ./tmp/$imageName $Image_local_path$imageName ) &>> ./log_ALASKA_v2_downloads
        # This is the download command, all output is no printed out in the terminal but rather in the file "./ALASKA_log_downloads"
	# note that the -c option is for "--continue: Continue getting a partially-downloaded file." To avoid restarting partial download one can instead use  the command "[ -f ./tmp/$imageName ] && echo rm ./tmp/$imageName" above(comment) which clean/remove the file ahead
    fi
    # For progress display purpose, we print out every 10 images the number of image download with the elapsed time.
    let image_index=$image_index+1
    if [ $(expr $image_index % 10) -eq 0 ];
    then
    currentTime=$(date +%s)
        echo "RAW image number $image_index / $NB_image_to_DL downloaded ! Time Elapsed = $(($currentTime - timeStart)) sec. "
    fi

done < $ALASKA_list_path
