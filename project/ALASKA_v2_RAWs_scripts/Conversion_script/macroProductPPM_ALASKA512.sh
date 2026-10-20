#!/bin/sh
#
cpt=0
#Create a destination dir called ppmDir
ppmDir=/utilisateurs/cogrannr/Img_datasets/ALASKA_v2/ALASKA_v2_BOSSstyle/ALASKA_v2_BOSSstyle_TIFF_512_COLOR
#Directory from which RAW images are read
inputDir=/utilisateurs/cogrannr/Img_datasets/ALASKA_v2/ALASKA_v2_RAWs

# HERE we declare (1) the configuration of raw conversion and (2) a random file name in folder /tmp/ in which this configuration will be written
tmpcfg=`mktemp `
echo "<?xml version="1.0" encoding="utf-8"?>
<UFRaw Version='7'>
<WindowMaximized>1</WindowMaximized>
<Interpolation>ppg</Interpolation>
<WB>Camera WB</WB>
<LensfunAuto>no</LensfunAuto>
<BaseLinearCurve Current='yes'></BaseLinearCurve>
<LinearCurve Current='yes'></LinearCurve>
<MatrixInputProfile Current='yes'>Color matrix</MatrixInputProfile>
<sRGBOutputProfile Current='yes'>sRGB</sRGBOutputProfile>
<SystemDisplayProfile Current='yes'>System default</SystemDisplayProfile>
</UFRaw>" > $tmpcfg


#IMPORTANT NOTE: to create grayscale images, simply replace EVERYWHERE ppm with pgm

for i in `ls $inputDir/` ; do # This is the path where raw images are stored
  echo "Image $cpt ($i)"
  if [ -f  $ppmDir/${i%.*}.ppm ]
  then
      echo "image $ppmDir/${i%.*}.ppm ALREADY EXIST !! SKIPPED !!"
      continue
  else
    echo "Image $cpt ($i)"
    #Raw Conversion
    # BEWARE : We handle X3F in a separated way ; we start differentiating X3F sigma trichromatic sensor (that cannot be processed by ufraw ...)
    if [ `echo $i | grep -ci X3F` -eq 1 ];
    then
        ./x3f_extract -q -ppm -no-denoise $inputDir/$i
        mv  $inputDir/$i.ppm /tmp/${i%.*}.ppm
    # BEWARE : This else case is for all other types of raw files
    else
        ufraw-batch --conf=$tmpcfg $inputDir/$i --output=/tmp/${i%.*}.ppm
    fi
    #Convertion for auto orientation to have an oriented image
    convert -auto-orient /tmp/${i%.*}.ppm /tmp/${i%.*}.2.ppm 
    mv /tmp/${i%.*}.2.ppm /tmp/${i%.*}.ppm

    line=`identify /tmp/${i%.*}.ppm`
#    echo "$line"
    arg1=`echo $line | cut -d " " -f3-3 | cut -d " " -f2-2`
    length=`echo $arg1 | cut -d "x" -f1-1`
    height=`echo $arg1 | cut -d "x" -f2-2`
    if [ "$height" -lt "$length" ] ; then 
	maxSize=`echo $length`
	newLength=`echo $length*512/$height | bc`
	newHeight=512
	newShiftLength=`echo $newLength/2-512/2 |bc`
	newShiftHeight=0
	convert -resize x`echo $newHeight` /tmp/${i%.*}.ppm /tmp/${i%.*}.R.ppm
    else
	maxSize=`echo $height`
	newLength=512
	newHeight=`echo $height*512/$length | bc`
	newShiftLength=0
	newShiftHeight=`echo $newHeight/2-512/2 |bc`
	convert -resize `echo $newLength`x /tmp/${i%.*}.ppm /tmp/${i%.*}.R.ppm

    fi

    if [ "$newLength" -lt 512 ] ; then 
	newLength=512
    fi

    if [ "$newHeight" -lt 512 ] ; then 
	newHeight=512
    fi

    #convert -depth 1 -crop 512x512+`echo $newShiftLength`+`echo $newShiftHeight` $i.R.ppm `echo $ppmDir/$cpt.ppm`
    convert -crop 512x512+`echo $newShiftLength`+`echo $newShiftHeight` /tmp/${i%.*}.R.ppm `echo $ppmDir/${i%.*}.ppm`
    rm /tmp/${i%.*}.ppm 
    rm /tmp/${i%.*}.R.ppm

    cpt=`echo $cpt+1 |bc`
  fi
done

rm $tmpcfg -f
