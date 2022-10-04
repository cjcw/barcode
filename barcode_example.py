
#import sys
#sys.path.append('/path/to/script/')
import barcode as bcode

"""
USER GUIDE FOR BARCODE DETECTION AND STRENGTH TRACKING
Author: Charlie Clarke-Williams (2022)

PRIOR REQUIREMENTS
--------
To detect barcodes over regions of interest, the IMF instantaneous amplitude timecourses must be saved. To do this:  
    1. Run EMD (or a preferred variant) on the channels of interest.
    2. For each channel, to the resulting IMF arrays, compute the instantaneous phase/frequency/amplitude timecourses
    3. Check the main frequencies of each IMF so that IMFs with a frequency within a certain range can be selected (see: INPUTS; region_imfis)
    4. Save the instantaneous amplitudes as a numpy array, using this naming structure: 
        
        {sessionString} + {emdString} + '.ia' + {channelString}
        
# Example template
import emd
import os
import numpy as np
X # channel signal
sample_rate # sample rate of X

# STEP 1
imf = emd.sift.sift(X) 

# STEP 2
IP, IF, IA = emd.spectra.frequency_transform(imf, sample_rate, 'hilbert') # get instantanous amplitudes of the IMFs

# STEP 3
for imfi, freqs in enumerate(IF.T):
    print('imfi:', imfi, freqs.mean())


# STEP 4
sessionStr = 'subject-1_ddmmyyyy' # between different regions, this string should be identical over a given recording session
emdStr = '.emd'
channelStr = '.chi.3' # this can be anything which helps idetify the source of the recording - e.g. '.hippocampus'
os.chdir(/path/to/save/IA/)
np.save(sessionStr+emdStr+'.ia'+channelStr, IA)


INPUTS
--------
regions : list
    A list of regions to be used for barcode detection 
region_paths2IAs : dict
    Each region entry should contain either a single path or list of paths to the IMF amplitude file (see above)
region_imfis : dict
    Each region entry contains the IMF indices to be used for barcode detection. 
    It is recommended to select those which are faster than 15 Hz. This is shown in the Example template in STEP 3. 
emdStr : string
    String to identify the variant of emd that was used to extract the imfs. e.g. '.eEMD'
sample_rate : int
    The sampling rate of the data
save : bool
    Should the outputs be saved to barcodeDir
barcodeDir : string | None
    The directory path to save the detected barcode information 

"""

# inputs
regions
region_paths2IAs
region_imfis
emdStr
sample_rate
save=True
barcodeDir='/directory/to/save/barcodeInfo/'


# detect barcodes
ics, barcodes, dimInfo, pairwiseInds, sesh_strs, sesh_lens = \
bcode.detect_barcodes(regions, region_paths2IAs, region_imfis, emdStr, sample_rate, save=save, barcodeDir=barcodeDir)


# visualise a barcode
barcode = barcodes[0]
bcode.plot_barcode(barcode, regions, region_imfis, region_colors)
             
             
'''
# load and unpack detected barcode information
import pickle as pk
saveStr = bcode.get_regionImf_saveStr(regions, region_imfis, emdStr)
with open(barcodeDir+'detectedBarcodes_info'+saveStr+'.pkl', 'rb') as h:
    detectedBarcodes_info = pk.load(h)
ics = detectedBarcodes_info['ics']
barcodes = detectedBarcodes_info['barcodes']
dimInfo = detectedBarcodes_info['dimInfo']
pairwiseInds = detectedBarcodes_info['pairwiseInds']
sesh_strs = detectedBarcodes_info['sesh_strs']
sesh_lens = detectedBarcodes_info['sesh_lens']
'''

# compute and save barcode strengths
for seshi in range(len(sesh_lens)):
    print(seshi)
    barcodeStrengths = bcode.get_barcodeStrengths(seshi, sesh_strs, sesh_lens, ics, regions, region_imfis, region_paths2IAs, 
                                                  dimInfo, pairwiseInds, emdStr, barcodeDir)
    




