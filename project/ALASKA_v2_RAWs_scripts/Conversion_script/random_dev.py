import numpy as np
import shutil
from PIL import Image
# Script used to randomly select the development paramters from RAW files to JPEG images.
# This merely consists of a initialiser ( the function __init__ ) and a "development profile" random generator.
# The initialiser is used to set all the development parameters; for simplicity we recall that those are the following:
#   1) "dem_algorithm"  --> the demosaicing algorithm
#   2) "resize"         --> a (sometimes used) resizing factor
#   3) "denoise"         --> directional pyramid denoising settings
#                       --> We use two parameters here "luminance" which corresponds to the denoising strenght and "detail" which is defines, roughly speaking, the level of edge preserving 
#   4) sharpening (with Unsharpening mask) settings
#                       --> We use two parameters there "radius" which corresponds to the size of the (convolution) unsharpening mask and "amount" which correspond to the amount of unsharpening the denoising strenght and "detail" which is defines, roughly speaking, the level of edge preserving 
#   5) Edge enhancement tools (micro-contrast) settings
#                       --> We use two parameters there "radius" which corresponds to the size of the (convolution) unsharpening mask and "amount" which correspond to the amount of unsharpening the denoising strenght and "detail" which is defines, roughly 
# More details can be found at: https://rawpedia.rawtherapee.com
#
# Eventually, outside from rawtherapee software, we also specifies:
#   6) JPEG compression quality factor
#   7) final image size with of course two paramters.

KERNEL_dict = {Image.NEAREST: "NEAREST", Image.BILINEAR:"BILINEAR", Image.BICUBIC: "BICUBIC", Image.LANCZOS: "LANCZOS"}
class devRandomGenerator:
# Initialiser whose main goal is to define the statistical distributions for all the parameters considered
# **************************#
# Main function: initialiser #
# **************************#
# The goal of this function is to select randomly  function 
    def __init__(self, qf, qf_probs, crop_size, dem, dem_probs, resize_kernel, resize_kernel_probs, seed=None):
        
        
        
        #First define the different distributions
        usm_radius_values = np.arange(0.3,3+0.01,0.01)
        usm_radius_prob = np.logspace(1,0.1,num=271)
        usm_radius_prob = usm_radius_prob/np.sum(usm_radius_prob)
        
        usm_amount_values = np.arange(0,1001,1)
        usm_amount_prob = np.concatenate([np.logspace(0,250,num=250, base=1.005),
                                          np.logspace(0,751,num=751, base=0.985)*(1.005**250)])
        usm_amount_prob = usm_amount_prob/np.sum(usm_amount_prob)
        
        denois_lum_values = np.arange(0,100+1,1)
        denois_lum_prob = np.concatenate([np.logspace(0,20,num=20, base=1.0025),
                                          np.logspace(0,81,num=81, base=0.990)*(1.0025**20)])
        denois_lum_prob = denois_lum_prob/np.sum(denois_lum_prob)

        self.r = np.random.RandomState(seed)

        self.dem = {"dem_algorithm": lambda: self.r.choice(dem,p=dem_probs) }
        self.usm = {"radius": lambda: self.r.choice(usm_radius_values,p=usm_radius_prob),
                    "amount": lambda: self.r.choice(usm_amount_values,p=usm_amount_prob) }

        self.denois = {"luminance": lambda: self.r.choice(denois_lum_values,p=denois_lum_prob),
                       "detail": lambda: self.r.randint(low=0, high=60)}

        self.microcontrast = {"quantity": lambda: min(round(self.r.gamma(1, 0.5) * 100), 100),
                              "uniformity": lambda: max(0, round(self.r.normal(30, 5)))}


        self.QF = {"QF": lambda: int(self.r.choice(qf, p=qf_probs))}

        self.crop = {"size": lambda: self.r.choice(crop_size,2)}
        self.resize_kernel = {"kernel": lambda: self.r.choice(resize_kernel, p=resize_kernel_probs)}
        self.resize_weight = {"factor": lambda: self.r.uniform(0,1)}

# Random profile according to the probabilities associated with each developement step, as step in the variable process_config from the main script ALASKA_conversion.py, we pick, or not, a random value for each paramter following the distribution defined in the initialiser
# The developement process is eventually written into a rawtherapee compatible pp3 file.
    def generateRandomRTProfile(self, imageDevList, outputPath, backupfile):
        radius=0
        amount=0
        luminance=0
        detail=0
        USM_before_DENOISE=1

# This is the pp3 file in which developement parameters will be written for later used in rawtherapee.
        currentProfile = open(outputPath, 'w+')
# Writing of the header of pp3 file.
        currentProfile.write("[Version]\nAppVersion=5.4\nVersion=331\n\n")
# Specifies if denoising is applied prior or after sharpening. 
        if (self.r.binomial(1, 0.5) == 1):  # probability of 1/2 to start with unsharpening 
# There we start we unsharpening mask and pick randomly the associated paramters (raduis and amount)
            if(imageDevList["choice"]["usm"] == 1):
                radius = self.usm["radius"]()
                amount = self.usm["amount"]()
                currentProfile.write(
                    "[Sharpening]\nEnabled=true\nMethod=usm\nRadius={}\nAmount={}\nThreshold=20;80;2000;1200;\n\n".format(radius, amount))

# and, in needed, specifies the parameters for the denoising
                if(imageDevList["choice"]["denois_if_usm"] == 1):
                    luminance = self.denois["luminance"]()
                    detail = self.denois["detail"]()
                    currentProfile.write(
                        "[Directional Pyramid Denoising]\nEnabled=true\nEnhance=false\nMedian=false\nLuma={}\nLdetail={}\n\n".format(luminance, detail))

# there the steps are applied in the other way round, i.e denoising first ....
        else:
            USM_before_DENOISE=0
            if(imageDevList["choice"]["denois"] == 1):
                luminance = self.denois["luminance"]()
                detail = self.denois["detail"]()
                currentProfile.write(
                    "[Directional Pyramid Denoising]\nEnabled=true\nEnhance=false\nMedian=false\nLuma={}\nLdetail={}\n\n".format(luminance, detail))
# ... and then unsharpening mask.
                if(imageDevList["choice"]["usm_if_denois"] == 1):
                    radius = self.usm["radius"]()
                    amount = self.usm["amount"]()
                    currentProfile.write(
                        "[Sharpening]\nEnabled=true\nMethod=usm\nRadius={}\nAmount={}\nThreshold=20;80;2000;1200;\n".format(radius, amount))

        currentProfile.close()

# Optional: one can log all developement parameters for all images into a single file. If so we write into a slighty more verbose the developement parameters
        if backupfile is not None:
            BackupProfile = open(backupfile, 'a+')
            if (radius == 0 and amount == 0):
                USM_set="OFF"
            else :
                USM_set="ON"  



            if (luminance == 0 and detail == 0):
                DENOISE_set="OFF" 
            else :
                DENOISE_set="ON" 

            RESIZE_factor = imageDevList["subsampling_factor"]
            if (imageDevList["subsampling_type"] == 0):
                RESIZE_set="ON_WITH_CROP"
                RESIZE_kernel = KERNEL_dict[imageDevList["resize_kernel"]]
            elif imageDevList["subsampling_type"] == 1:
                RESIZE_set="ON_ALONE" 
                RESIZE_kernel = KERNEL_dict[imageDevList["resize_kernel"]]
            elif imageDevList["subsampling_type"] == 2:
                RESIZE_set="CROP_ONLY"
                RESIZE_kernel = "NONE"
            else:
                RESIZE_set="OFF"

# The line written into the log file
            BackupProfile.write(" %30s | DEM %20s | %d | USM %3s , %5.2f , %6.2f | DENOISE %3s , %5.2f , %5.2f | RESIZE %s , %s , %5.5f | %4d x %4d | %3d \n"  %(imageDevList["name"] , imageDevList["dem"], USM_before_DENOISE, USM_set, radius , amount, DENOISE_set, luminance, detail, RESIZE_set, RESIZE_kernel, RESIZE_factor, imageDevList["crop_size"][0], imageDevList["crop_size"][1] , imageDevList["qf"]  ) )

        BackupProfile.close()

