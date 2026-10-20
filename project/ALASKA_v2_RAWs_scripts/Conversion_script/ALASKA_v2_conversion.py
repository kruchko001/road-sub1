#ALASKA_convertion
# Conversion script main function
# Created by Quentin GIBOULOT (UTT), Remi COGRANNE (UTT), and Patrick BAS (CRIStAL)
# This scipt converts RAW images from a specified directory to jpg image using rawtherapee (version 5.4)
# The conversion includes a wide range of processing algorithm and, more specifically randomize the processing pipeline from raw to jpg.
# All the required parameters (directories and development / processing parameters) are defined within the present script. Therefore simply launch it by calling it like :
# python3 ./ALASKA_conversion.py
# OR python2 ./ALASKA_conversion.py > output_conv.txt (if you'd rather familiar with python2 AND want to get the output into a log file rather than in the terminal)
#
# The random development parameters are the followings:
#   1) the demosaicing algorithm
#   2) a randomized subsampling algorithm (smart crop and/or resizing)
#   3) directional pyramid denoising settings
#   4) sharpening (with Unsharpening mask) settings
#   6) JPEG compression quality factor
#
#
# There is two sets of paramters to specify at the begining of present script.
# First, the variable "config_path" specifies the directories used (from which raw images are read, where jpeg images are written and where settings applied on each images mainly)
# Second, the variable "config_process" specifies which process is picked with which probability during the development process. Note that the default setting exactly matches the configuration used to create JPEG images for the ALASKA challenge.
# The current script is lightly commented, more details can be found in the pdf documentation.
import numpy as np
import os
from subprocess import call
import shutil
from PIL import Image
import multiprocessing
from joblib import Parallel, delayed
from hashlib import md5

import image_conversion_fun as imProc
from random_dev import devRandomGenerator

#IMPORTANT NOTE : paths should be changed to your system, especially the "root"


# Note that this first variable is used to allows several development, with possibly various settings, with names that can be easily identified
# Switch indicating whether an uncompressed version (tiff) of the developed images should be kept
# Usefull for steganography / steganalysis / forensics in spatial domain
keepUncompressed=True; 

# Name of the directory that will contain developed JPEG images:
baseName="ALASKA_v2"

# Here we start defining the most important variable "config_path" which defines ALL directoty used
# we start with the main root direcroty, in which everything will be output:
config_path = dict(root="devHome")
# where RAW images should be
config_path["raw_dir"] = config_path["root"]+"/ALASKA_v2_RAWs" 
# where JPEG images will be stored:
config_path["out_dir"]=config_path["root"]+"/"+baseName+"_JPG_VariousSize_QFVarious"
# where TIFF (uncompressed) images will be stored:
config_path["out_dir_tif"]=config_path["root"]+"/"+baseName+"_TIFF_VariousSize"

# where itermediate (temporary) images will be stored:
config_path["tmp_dir"]=config_path["root"]+"/TIFF_tmp"
# important, direcroty in which "profiles" that defines the developpement parameters will be written for each and every image:
config_path["profile_used_dir"]=config_path["root"]+"/profiles_applied"
# initial profiles, for demosaicing only:
config_path["dem_profile_dir"]=config_path["root"]+"/demProfiles" 

# File in which the randomly generated developement parameters are output for loging purpose:
backup_file_path= config_path["root"]+"/list_img_profiles.txt"  

#Second main variable, the "config_process", that defines, for ALL developpement parameters, the range in which those are picked.
#This configuration of the developpement process is quite "coarse grain"; More specification on the distribution of each parameters are to be found in the companion script "random_dev.py" 
config_process=dict()
# Number of JPG image to create; if larger than number of RAW images automatically reset to the latest ; otherwise, randomly select a subset of images
config_process["number_of_output_images"]=100000
# Probability of using unsharpening mask
config_process["prob_usm"]=0.8
# Probability of using directional pyramid denoising algorithm 
config_process["prob_denoise"]=0.6
# Probability of using unsharpening mask if denoising is used first
config_process["prob_usm_if_denoise"]=0.4
# Probability of using denoising if unsharpening mask is used first
config_process["prob_denoise_if_usm"]=0.4

# Possible final image size (a very last cropping step is applied)
config_process["crop_size"]=[512,640,720,1024]          
#To match this final size, we can either crop / resize or do both; those are used with the following probabilities:
# Probability of resizing images:
config_process["prob_resize_only"]=0.25
# Probability of croping the image:
config_process["prob_crop_only"]=0.25
# Probability of doing both resising and then croping:
config_process["prob_resize_and_crop"]= 1 - config_process["prob_resize_only"] - config_process["prob_crop_only"]
# Definition of the set of prossible resampling kernels (for resizing)....
config_process["resize_kernel"] = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
#Along with the probabability of each
config_process["resize_kernel_prob"]= [0.1,0.15,0.25,0.5]
#Maximal resizing factor (here, upsampling by 25%)
config_process["resize_factor_upperBound"] = 1.25
# Min and max possible QF, beware that this range should match with the number of values is the associated probabilities vector.
config_process["jpeg_qf"]=np.arange(60,100+1)
# Probabilities associated with each possible JPEG QF compression; values estimated from 2.75+ Milions of images from FlickR.
config_process["jpeg_qf_probabilities"]=[ 0.0030 , 0 , 0 , 0 , 0 , 0.0010 , 0 , 0 , 0 , 0.0010 , 0.0070 , 0.0020 , 0.0010 , 0 , 0.0010 , 0.1080 , 0.0010 , 0.0010 , 0.0010 , 0.0020 , 0.1730 , 0.0020 , 0.0020 , 0.0020 , 0.0040 , 0.0840 , 0.0050 , 0.0040 , 0.0100 , 0.0180 , 0.1190 , 0.0090 , 0.0160 , 0.0220 , 0.0240 , 0.0510 , 0.0470 , 0.0350 , 0.0560 , 0.0300 , 0.1580 ]
# Thing are slighly different for demosaicing since each demosaicing algorithm needs to be associated with a pp3 file. Hence, we first set the directory that contain the files for different demosaicing algorithms ....
config_process["demosaicing"]=os.listdir(config_path["dem_profile_dir"])
#then we need to set the probability of each of those (that, of course, needs to be a vector with number of component equals the number of demosaicing files);
dem_probs=[]
for dem_name in config_process["demosaicing"]:
	if dem_name=="dem_igv.pp3":
		dem_probs.append(0.1)
	if dem_name=="dem_amaze.pp3":
		dem_probs.append(0.4)
	if dem_name=="dem_fast.pp3":
		dem_probs.append(0.15)
	if dem_name=="dem_dcb_2_amel.pp3":
		dem_probs.append(0.35)
config_process["demosaicing_probabilities"]=dem_probs


# **************************#
# MAIN conversion function #
# **************************#
def From_RAW_to_JPG( RAWimageName ):
# Here we start 1) splitting image path by filename and extension
	imageBaseName = os.path.splitext(RAWimageName)[0]
	imageRawExtension = os.path.splitext(RAWimageName)[1]
# and 2) create path for all temporary image.
	TIFimagePath=os.path.join(config_path["tmp_dir"], imageBaseName + "_tmp.tif")
	TIFimage2Path=os.path.join(config_path["tmp_dir"], imageBaseName + "_tmp2.tif")
	TIFimage3Path=os.path.join(config_path["out_dir_tif"], imageBaseName + ".tif")
	RAWimagePath=os.path.join(config_path["raw_dir"], imageBaseName + imageRawExtension)
	ImageProfilePath=os.path.join(config_path["profile_used_dir"], imageBaseName + ".pp3")
	print("Converting Image " + RAWimagePath)

# We create, for each and every images, a random generator that will be used to create (randomly) a development process file.
# To ensure the randomness and reproductibility of the developppement process, we propose to seed each generator by the MD5 hashsum of image filename
# To generate several version of the same dataset you can use, for instance, the following commands (which hash a value from system time to generate a random seed for every image)
# imageSeed=int.from_bytes(md5( (round(time.time() * 100000)**2).to_bytes(32, byteorder='big') ).digest(), 'big') % 2**32
	imageSeed=int.from_bytes(md5(bytes(imageBaseName,'utf-8')).digest(), 'big') % 2**32
	rg = devRandomGenerator(config_process["jpeg_qf"], config_process["jpeg_qf_probabilities"], config_process["crop_size"], config_process["demosaicing"], config_process["demosaicing_probabilities"], config_process["resize_kernel"], config_process["resize_kernel_prob"], seed=imageSeed)

# The very first step consists in generating a random development file for the given image ; thus, if such a file exists, the associated image has already been processed.
# This is used to allows a cheap, yet efficient parallelization by simply launching several time the same conversion script (see Section "Parallelization" in the pdf documentation)
	if not os.path.exists( ImageProfilePath ):
# INITIALIZATION: random selection of development / processing parameters and storing into a pp3 file
		DevList = {"name" : imageBaseName,
					  "choice" : {"usm" : rg.r.binomial(1, config_process["prob_usm"]), 
								  "denois" : rg.r.binomial(1,config_process["prob_denoise"]),
								  "usm_if_denois" : rg.r.binomial(1, config_process["prob_usm_if_denoise"]),
								  "denois_if_usm" : rg.r.binomial(1, config_process["prob_denoise_if_usm"])
								  },
					 "dem" : rg.dem["dem_algorithm"](),
					 "subsampling_type" : rg.r.choice([0,1,2],p=[config_process["prob_resize_and_crop"], config_process["prob_resize_only"], config_process["prob_crop_only"]] ),
					 "resize_kernel" : rg.resize_kernel["kernel"](),
					 "resize_weight" : rg.resize_weight["factor"](),
					 "crop_size" : rg.crop["size"](),
					 "qf" : rg.QF["QF"]()
					 }

		dumpFile=open("/tmp/dumpOutPut.txt","wb") # this file is used to dump output from rawtherapee and x3f_extract which are quite verbose and cannot be used in quiet mode :(
# FIRST STEP: APPLYING DEMOSAICING !
# Note that, we used rawtherapee version 5.7 which seems, as opposed to version 5.3, to handle efficiently X3F Sigma foveon trichromatic sensor
		if imageRawExtension.upper() == ".X3F":
# However, some X3F images still cannot be processed with rawtherapee; for this reason we try first to apply rawtherappe; if it fails, we call x3f_extractor executable.
			print("[WARNING] Sigma Foveon X3F raw file ! Trying RawTherapee" )
			#This is a typical use of call to execute the rawtherapee-cli command (note that the output are dumped to /tmp/ )
			call(["rawtherapee-cli", "-a", "-q", "-t", "-b16", "-o", TIFimagePath, "-p", os.path.join(config_path["dem_profile_dir"] , DevList["dem"]) , "-c", RAWimagePath ], stdout=dumpFile, stderr=dumpFile)
# This is the "if rawtherappe fails" which is tested as "if not image file is generated"
			if not os.path.exists(TIFimagePath):
				#This is a typical use of binary x3f_extract to dump tiff data from X3F file (note that the output are dumped to /tmp/ )
				call(["./x3f_extract", "-q", "-tiff", "-no-denoise", "-no-sgain", RAWimagePath ], stdout=dumpFile, stderr=dumpFile)
				#This script automatically write output image into the same directory, we move this file to match TIFimagePath variable
				shutil.move(os.path.join(RAWimagePath + ".tif") ,  TIFimagePath)
			if not os.path.exists(TIFimagePath):
				print("[ERROR] neither rawtherapee nor x3f_extract managed to read this file! Are you sure it is not corrupted ?!?")
# if not X3F raw image files, we call also rawtherapee 
		else :
			call(["rawtherapee-cli", "-a", "-q", "-t", "-b16", "-o", TIFimagePath, "-p", os.path.join(config_path["dem_profile_dir"] , DevList["dem"]) , "-c", RAWimagePath ], stdout=dumpFile, stderr=dumpFile)

# Before moving forward, we ensure that the TIF image (resulting for demosaicing of RAW) does exist; indeed some raw images files format cannot be read.
		if os.path.exists(TIFimagePath):

# SECOND STEP: RESIZING and CROPPING
			#First of all, we carry out the resizing ; thi requires one extra parameter (the resizing factor) that depends on the image size ; 
			#To deal with this we call the resizing and get the factor as an output ....
			DevList["subsampling_factor"] = imProc.image_randomizeResizing(TIFimagePath, TIFimage2Path, DevList['crop_size'][0], DevList['crop_size'][1], subsampling_type=DevList['subsampling_type'], kernel=DevList['resize_kernel'], resize_weight=DevList['resize_weight'], resize_factor_UB=config_process["resize_factor_upperBound"])
			#... Then, and only then, we can write dump the profiles of the image in the associated file.
			rg.generateRandomRTProfile(imageDevList = DevList, outputPath = ImageProfilePath, backupfile=backup_file_path )

			if os.path.exists( TIFimage2Path ) :

# FORTH (and main) STEP: generating processing pipeline file and using rawtherapee
				call(["rawtherapee-cli", "-a", "-q", "-t", "-b8", "-o", TIFimage3Path , "-p", ImageProfilePath  , "-c", TIFimage2Path ], stdout=dumpFile, stderr=dumpFile)

				if os.path.exists( TIFimage3Path ) : 
# LAST STEP: (mere) jpeg compression
					imProc.jpeg_compression(infile=TIFimage3Path  , outpath=os.path.join(config_path["out_dir"], imageBaseName + ".jpg") ,  qf= DevList["qf"])

# Eventually, we double check that the associated JPEG image exists; if not we keep the TIF temporary files for backup and debugging
					if os.path.exists( os.path.join(config_path["out_dir"], imageBaseName + ".jpg") ) :
# We can either keep tiff (uncompressed) image
						if keepUncompressed :
							call(["rm", TIFimagePath, TIFimage2Path ])
							print("[SUCCESS] Images " , imageBaseName + ".tif and" , imageBaseName + ".jpg" , " Converted successfully ")
# or keep only the jpg, in such case, we remove ALL itermediate images
						else :
							call(["rm", TIFimagePath, TIFimage2Path, TIFimage3Path ])
							print("[SUCCESS] Image " , imageBaseName + ".jpg" , " Converted successdully ")

# Print out possible causes that lead not to develop the given RAW images, for logging.
					else:
						print("[ERROR] Ultimate JPEG FAILED FOR" + RAWimagePath )
				else:
					print("[ERROR] Last conversion (RAWTHERAPEE) FAILED FOR" + RAWimagePath )
			else:
				print("[ERROR] SUBSAMPLING FAILED FOR" + RAWimagePath )
		else:
			print("[ERROR] Image " + RAWimagePath + " can hardly be converted to TIFF: skipped")
	else: 
		print("[WARNING] Image: " + imageBaseName + ".jpg already processed: skipped ")


# **************************#
# BEGINING OF THE SCRIPT #
# **************************#
if __name__ == '__main__':
	# First of all, we check out if some specified directories need to be created and do so.
	for d in config_path:
		if not os.path.exists(config_path[d]):
			# Creates the directory with classic permissions
			print("Created --> " + config_path[d])
			os.makedirs(config_path[d], 0o755)

# List of RAW images into the specified directory
#	RAWimagesName = os.listdir(config_path["raw_dir"])
	RAWimagesName = sorted(os.listdir(config_path["raw_dir"]))

# Random selection of a subset of images (the total number of image picked is specified in config_process --> number_of_output_images)
# We selected random indices
	imageIndices=np.arange(len(RAWimagesName))
	np.random.shuffle(imageIndices)
	imageIndices=imageIndices[0:min(config_process["number_of_output_images"] , len(RAWimagesName) )]
	print("Number of images to be created/converted : ", min(config_process["number_of_output_images"] , len(RAWimagesName) ) ) 

# The script can be launched using multiprocessing
# Default configuration is to use half of the number of cores ... you can set this value to something higher (Remi used 3/4 of total number of cores)
	numCores = int( multiprocessing.cpu_count() / 2)
	Parallel(n_jobs=numCores)(delayed(From_RAW_to_JPG)(RAWimageName=RAWimagesName[index]) for index in imageIndices)

# This alternative consists is the same processing ... only without multiprocessing
#Instead, we simply loop over all images
	#for index in imageIndices:
	#	From_RAW_to_JPG(RAWimageName=RAWimagesName[index])
