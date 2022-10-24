# barcode
Code for barcode detection and strength tracking

### Requirements
To detect barcodes over regions of interest, the IMF instantaneous amplitude timecourses must first be saved. To do this:  
    1. Run EMD (or a preferred variant) on the channels of interest.  
    2. For each channel, to the resulting IMF arrays, compute the instantaneous phase/frequency/amplitude timecourses  
    3. Check the main frequencies of each IMF so that IMFs with a frequency within a certain range can be selected (see: INPUTS; region_imfis)  
    4. Save the instantaneous amplitudes as a numpy array, using this naming structure:  
        
        {sessionString} + {emdString} + '.ia' + {channelString}
        
The code was built using Python 3.6.8.

### Notes
Example use can be found in the barcode_example.py script.
