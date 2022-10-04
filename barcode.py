
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
#import ccwBaseFunctions as ccw
import itertools
from scipy import stats
import pickle as pk
import seaborn as sns
import scipy
from sklearn.decomposition import PCA
import scipy.signal as sig
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import KMeans



 ### --- USER CONFIG --- ###
def get_region_imfi2titFuncs(regions):
    """
    User can modify this function as needed. 
    Returns a dictionary of functions. The key for each dictionary entry is the region and the function takes imf index (imfi) 
    as an argument and returns an oscillatory title for that region imfi. As a default, the title will be set to:
    {region}+'_IMF-'+{imfi+1}, but this can be modified by the user - e.g:
    
    region_imfi2titFuncs = {}
    for region in regions:
        if region in ['hippocampus']:
            def imfi2tit(imfi):
                if imfi < 3:
                    if imfi == 0:
                        oscStr = 'fast-gamma'
                    elif imfi == 1:
                        oscStr = 'mid-gamma'
                    elif imfi == 2:
                        oscStr = 'slow-gamma'
                    title = region+'_'+oscStr
                else:
                    title = region+'_IMF-'+str(imfi+1)
                return title
        else:
            def imfi2tit(imfi):
                title = region+'_IMF-'+str(imfi+1)
                return title
        
        region_imfi2titFuncs[region] = imfi2tit
    
    Parameters
    ----------
    regions : ndarray | list
        the brain regions of interest
    
    Returns
    -------
    region_imfi2titFuncs : dict
        each region dictionary entry contains a function which takes imfi and returns a corresponding title for that imf index. 
        The titles set for each region can  can be modified by the user accordingly - see above.
    """
    
    region_imfi2titFuncs = {}
    for region in regions:
        def imfi2tit(imfi):
            title = region+'_IMF-'+str(imfi+1)
            return title
        region_imfi2titFuncs[region] = imfi2tit
    return region_imfi2titFuncs


 ### --- WORK FUNCTIONS --- ###
def smooth(x, smoothSDs=1):
    """
    Parameters
    ----------
    x : ndarray
        1D array to smooth
    
    Returns
    -------
    x : ndarray
        1D array, smoothed 
    """
    if smoothSDs is None:
        return x
    return scipy.ndimage.filters.gaussian_filter1d(x, smoothSDs)

def get_regionImf_saveStr(regions, region_imfis, emdStr):
    """
    Parameters
    ----------
    regions : ndarray | list
        The brain regions of interest
    region_imfis : dict
        Each region entry contains the IMF indices to be used
    emdStr : string
        String to identify the variant of emd that was used to extract the imfs. e.g. '.eEMD'
    
    Returns
    -------
    saveStr : string
        String which identifies the regions and imfs of interest
    """
    saveStr = emdStr+'_'
    for region in regions:
        if region in region_imfis.keys():
            saveStr += region+'-'
            for imfi in region_imfis[region]:
                saveStr += str(imfi+1)
            saveStr += '_'
    saveStr = saveStr[:-1]
    return saveStr



def get_ampNormFuncs(regions, region_imfis, region_paths2IAs, normalise_opts):
    """
    Parameters
    ----------
    regions : ndarray | list
        The brain regions of interest
    region_imfis : dict
        Each region entry contains the IMF indices to be used
    region_paths2IAs : dict
        Each region entry contains a string (or session-wise list of strings) that is the complete path to the file to load the 
        IMF instantaneous amplitudes ([time x nImfs] array called IA in emd package) for that region. These should be named
        according to the guidelines outlined in the user guide.
    normalise_opts : dict
        Dictionaty with normalise options. See user guide for more information.
        
    Returns
    -------
    region_sesh_imfi_normFuncs : dict
        A nested dictionary whereby region_sesh_imfi_normFuncs[region][seshi][imfi] contains a function used to normalise 
        the corresponding imf amplitude of that region and session.
    """
    
    nSessions = len(region_paths2IAs[regions[0]])

    if normalise_opts['norm'] is not None and not normalise_opts['each_sesh']:
        print('normalising amplitudes over all sessions using', normalise_opts['norm'])
        reg_imf_m_sd = {}
        for region in regions:
            IA = np.column_stack([np.load(region_paths2IAs[region][seshi]) for seshi in range(nSessions)])
            reg_imf_m_sd[region] = np.column_stack([IA.mean(axis=0), IA.std(axis=0)])

    region_sesh_imfi_normFuncs = {}
    for region in regions:
        region_sesh_imfi_normFuncs[region] = {}
        for seshi in range(nSessions):
            region_sesh_imfi_normFuncs[region][seshi] = {}
            if normalise_opts['norm'] is None:
                for imfi in region_imfis[region]:

                    def norm_amp(amp):
                        amp = smooth(amp, normalise_opts['smoothSDs'])
                        return amp

                    region_sesh_imfi_normFuncs[region][seshi][imfi] = norm_amp
            else:
                if normalise_opts['each_sesh']:
                    if all([region == regions[0], seshi == 0]):
                        print('normalising amplitudes for each session using', normalise_opts['norm'])
                    IA = np.load(region_paths2IAs[region][seshi])
                    imf_m_sd = np.column_stack([IA.mean(axis=0), IA.std(axis=0)])

                for imfi in region_imfis[region]:
                    if normalise_opts['each_sesh']:
                        m, sd = imf_m_sd[imfi, :]
                    else:
                        m, sd = reg_imf_m_sd[region][imfi, :]

                    if normalise_opts['norm'] == 'sd':
                        def norm_amp(amp):
                            amp = smooth(np.divide(amp, sd), normalise_opts['smoothSDs'])
                            return amp
                    elif normalise_opts['norm'] == 'z':
                        def norm_amp(amp):
                            amp = smooth((amp - m) / sd, normalise_opts['smoothSDs'])
                            return amp
                    else:
                        raise ValueError
                        print("Warning: amplitude normalise option (normalise_opts['norm']) not recognised. It should be one of: ")
                        print(" 'sd' | 'z' | None ", 'sd is recommended')

                    region_sesh_imfi_normFuncs[region][seshi][imfi] = norm_amp


    return region_sesh_imfi_normFuncs

def get_pairwiseInds(nDims_1d, mask_within=True):
    """
    Parameters
    ----------
    nDims_1d : int
        Total number of imfs used for barcode detection
        
    Returns
    -------
    pairwiseInds : ndarray
        2D [nDims_2d x 2] array. Each index row corresponds to the the pair of 1D indices used for the 2D index. 
    """
    
    pairwiseInds = np.row_stack([i for i in itertools.combinations(np.arange(nDims_1d), 2)])
    if not mask_within:
        pairwiseInds = np.row_stack([pairwiseInds, np.row_stack([[i, i] for i in range(nDims_1d)])])
    return pairwiseInds
    

def regImfs_1d_to_2d(obVecs_1d, mask_within=True):
    """
    Converts the 1-dimensional imf amplitude observation matrix [nObs x nImfs] to a 2D co-engagement one.
    """
    
    nDims_1d = obVecs_1d.shape[1]
    pairwiseInds = get_pairwiseInds(nDims_1d, mask_within=mask_within)
    
    obVecs_2d = np.column_stack([np.multiply(obVecs_1d[:, x], obVecs_1d[:, y]) for x, y in pairwiseInds])
    
    return obVecs_2d, pairwiseInds

    

def runICA(dims, nICs, max_iter=500):
    """
    run Independent Component Analysis (ICA) on a feature matrix 
    """
    from sklearn.decomposition import FastICA
    model = FastICA(n_components=nICs, max_iter=max_iter)
    proj = model.fit(dims.T).transform(dims.T)
    return proj.T

def flipSign(vec):
    """ flip based on mean of all neg vs mean of all pos """
    neg = np.mean(np.abs([i for i in vec.ravel() if i < 0]))
    pos = np.mean(np.abs([i for i in vec.ravel() if i > 0]))
    if neg > pos:
        return -vec
    else:
        return vec
    
    
def fix_region_paths2IAs(region_paths2IAs):
    """ 
    If a single string is given as a region entry for region_paths2IAs (i.e., a single session is to be used), 
    this function will convert it to a list 
    """
    for region in region_paths2IAs:
        if isinstance(region_paths2IAs[region], str):
            region_paths2IAs[region] = [region_paths2IAs[region]]
    return region_paths2IAs

def check_region_paths2IAs(regions, region_paths2IAs, emdStr):
    """
    Checks:
        (1) number of paths to IAs are the same for each region
        (2) recording duration for each region is the same, for each session 
        (3) that the paths are appropriately named and distinct for each session 
        
        for (3): for each session, the paths for each region should be identical up to the variable, emdStr')
    
    
    Parameters
    ----------
    regions : ndarray | list
        The brain regions of interest
    region_paths2IAs : dict
        Each region entry contains a string (or session-wise list of strings) that is the complete path to the file to load the 
        IMF instantaneous amplitudes ([time x nImfs] array called IA in emd package) for that region. These should be named
        according to the guidelines outlined in the user guide.
    emdStr : string
        String to identify the variant of emd that was used to extract the imfs. e.g. '.eEMD'
        
    Returns
    -------
    sesh_lens : list
        Each session entry is a integer specifying the length of the session amplitude file  
    sesh_strs : list
        Each session entry is a string identifying the session amplitude file to load (coming after the directory path). See 
        the user guide for more information.
    nSessions : int
        The number of sessions detected.    
    """
    # (1)
    if len(np.unique([len(region_paths2IAs[region]) for region in regions])) != 1:
        raise ValueError('differing number of sessions found for different regions')
    nSessions = len(region_paths2IAs[regions[0]])
    
    # (2)
    sesh_lens = []
    error_count = 0
    for seshi in range(nSessions):
        region_seshLens = np.array([np.load(region_paths2IAs[region][seshi]).shape[0] for region in regions])
        if len(np.unique(region_seshLens)) != 1:
            print('Warning: IMF durations for different regions are of different length for seshi =', seshi)
            error_count += 1
        else:
            sesh_lens.append(region_seshLens[0])
    if error_count:
        raise ValueError('IMF durations for different regions are of different length for '+\
                         str(error_count)+'/'+str(nSessions)+' sessions')
    sesh_lens = np.array(sesh_lens)
    
    # (3)
    sesh_strs = []
    error_count = 0
    for seshi in range(nSessions):
        region_seshiStrs = [region_paths2IAs[region][seshi].split('/')[-1].split(emdStr)[0] for region in regions]
        if len(np.unique(region_seshiStrs)) != 1:
            print('Warning: IMF amplitude path name error for seshi =', seshi)
            error_count += 1
        else:
            sesh_strs.append(region_seshiStrs[0])
    if error_count:
        raise ValueError('IMF amplitude path name error for '+str(error_count)+' sessions. See documentation for help.')
    sesh_strs = np.array(sesh_strs)
    if len(np.unique(sesh_strs)) != len(sesh_strs):
        raise ValueError('Unable to get unique session strings. See documentation for help.')
        
    nSessions = len(sesh_strs)
    
    return sesh_lens, sesh_strs, nSessions

def save_pickle(name, obj):
    with open(name+'.pkl', 'wb') as handle:
        pk.dump(obj, handle, pk.HIGHEST_PROTOCOL)
        
def save_barcodes(regions, region_imfis, emdStr, ics, barcodes, dimInfo, pairwiseInds, sesh_strs, sesh_lens, barcodeDir):
    """
    Save barcode information as a pickle dictionary.
    """
    detectedBarcodes_info = {}
    for k, x in zip(['ics', 'barcodes', 'dimInfo', 'pairwiseInds', 'sesh_strs', 'sesh_lens'], 
                    [ics, barcodes, dimInfo, pairwiseInds, sesh_strs, sesh_lens]):
        detectedBarcodes_info[k] = x
        
    saveStr = get_regionImf_saveStr(regions, region_imfis, emdStr)
    os.chdir(barcodeDir)
    save_pickle('detectedBarcodes_info'+saveStr, detectedBarcodes_info)
    print('barcodes saved: ', 'detectedBarcodes_info'+saveStr+'.pkl')
    print('to this directory: ', barcodeDir)
        
def detect_barcodes(regions, region_paths2IAs, region_imfis, emdStr, sample_rate, save=False, barcodeDir=None, nICs=None, downsample_interval_ms=500,
                    normalise_opts = {'norm' : 'sd', 'smoothSDs' : 1, 'each_sesh' : True}, sesh_excludeWindows=None):
    """
    Parameters
    ----------
    regions : ndarray | list
        The brain regions of interest
    region_paths2IAs : dict
        Each region entry contains a string (or session-wise list of strings) that is the complete path to the file to load the 
        IMF instantaneous amplitudes ([time x nImfs] array called IA in emd package) for that region. These should be named
        according to the guidelines outlined in the user guide.
    region_imfis : dict
        Each region entry contains the IMF indices to be used.
    emdStr : string
        String to identify the variant of emd that was used to extract the imfs. e.g. '.eEMD'
    sample_rate : int
        The sampling rate of the data
    save : bool
        Should the outputs be saved to barcodeDir
    barcodeDir : string | None
        The directory path to save the detected barcode information 
    nICs : int | None
        The number of barcodes to detect. If none, then 1.5x the number of 1D dimensions will be used (recommended).
    downsample_interval_ms : int
        The sample interval between feature vectors for barcode detection
    normalise_opts : dict
        Options to normalise IMF amplitudes prior to detection:
            'norm' : 'sd' | 'z' | None
            'each_sesh' : bool (if True, session amplitudes will be normalised independently)
            'smoothSDs' : int | None (standard deviation of gaussian kernal to smooth amplitude)
    sesh_excludeWindows : list | None
        Each session list entry contains a [nWindows x 2] numpy array which details start and end points of windows to 
        exclude for barcode detection.

    Returns
    -------
    ics : ndarray
        [nBarcodes x nPairs] independent component vectors
    barcodes : ndarray
        [nBarcodes x nIMFs x nIMFs] independent components, translated into 2D barcodes
    dimInfo : dataframe
        Dataframe detailing the IMF index and region corrresponding to each IMF amplitude (1D index)
    pairwiseInds : ndarray
        [nDims_2d x 2] array. Each index row corresponds to the the pair of 1D indices used for the 2D index
    sesh_strs : list
        Each session entry is a string identifying the session amplitude file to load (coming after the directory path). See 
        the user guide for more information.
    sesh_lens : list
        Each session entry is a integer specifying the length of the session amplitude file
    """
    # check and format inputs
    region_paths2IAs = fix_region_paths2IAs(region_paths2IAs)

    # check same paths to IAs are the same for each region
    sesh_lens, sesh_strs, nSessions = check_region_paths2IAs(regions, region_paths2IAs, emdStr)
    
    
    downsample_interval = int(sample_rate*(downsample_interval_ms / 1000))
    if sesh_excludeWindows is None:
        sesh_excludeWindows = [None]*nSessions
    elif len(sesh_excludeWindows) != nSessions:
        raise ValueError('Number of excludeWindows different from number of sessions.') 
        
    sesh_excludeTimes = []
    for excludeWindows in sesh_excludeWindows:
        if excludeWindows is None:
            sesh_excludeTimes.append(np.array([]))
        else:
            sesh_excludeTimes.append(np.concatenate([np.arange(start, end) for start, end in excludeWindows]))

    sesh_downsampleInds = [np.setdiff1d(np.arange(downsample_interval, sesh_len, downsample_interval), excludeTimes) \
                           for sesh_len, excludeTimes in zip(sesh_lens, sesh_excludeTimes)]

    for region in region_imfis:
        region_imfis[region] = np.array(sorted(region_imfis[region]))[::-1]

    region_sesh_imfi_normFuncs = get_ampNormFuncs(regions, region_imfis, region_paths2IAs, normalise_opts)

    region_dims_1d = {}
    for region in regions:
        region_dims_1d[region] = []
        for seshi in range(nSessions):
            IA = np.load(region_paths2IAs[region][seshi])
            IA = np.column_stack([region_sesh_imfi_normFuncs[region][seshi][imfi](IA[:, imfi]) for imfi in region_imfis[region]])

            region_dims_1d[region].append(IA[sesh_downsampleInds[seshi], :])

        region_dims_1d[region] = np.row_stack(region_dims_1d[region])

    dat4ica = np.column_stack([region_dims_1d[region] for region in regions])
    dimInfo = []
    for region in regions:
        for imfi in region_imfis[region]:
            dimInfo.append({'region' : region, 'imfi' : imfi})
    dimInfo = pd.DataFrame(dimInfo)
    
    if nICs is None:
            print('Number of ICs was not specified. using 1.5 * nDims_1D for 2D barcdes as recommended')
            nICs = int(np.ceil(dat4ica.shape[1]*1.5))
    dat4ica, pairwiseInds = regImfs_1d_to_2d(dat4ica)
    
    ics = runICA(dat4ica, nICs)
    ics = np.row_stack([flipSign(ic) for ic in ics])
    barcodes = np.array([vec2barcode(ic, pairwiseInds) for ic in ics])
    
    if save_barcodes:
        if barcodeDir is None:
            print('Barcodes will not be saved as barcodeDir has not been specified')
        else:
            save_barcodes(regions, region_imfis, emdStr, ics, barcodes, dimInfo, pairwiseInds, sesh_strs, sesh_lens, barcodeDir)
        
    return ics, barcodes, dimInfo, pairwiseInds, sesh_strs, sesh_lens

# fix - sort memory error
def get_barcodeStrengths(seshi, sesh_strs, sesh_lens, ics, regions, region_imfis, region_paths2IAs, dimInfo, pairwiseInds,
                         emdStr, barcodeDir, chunk=(False, 10000)):
    """
    Parameters
    ----------
    seshi : int
        The session index to run
    sesh_strs : list
        Each session entry is a string identifying the session amplitude file to load (coming after the directory path). See 
        the user guide for more information.
    sesh_lens : list
        Each session entry is a integer specifying the length of the session amplitude file
    ics : ndarray
        [nBarcodes x nPairs] independent component vectors
    regions : ndarray | list
        The brain regions of interest
    region_imfis : dict
        Each region entry contains the IMF indices to be used
    region_paths2IAs : dict
        Each region entry contains a string (or session-wise list of strings) that is the complete path to the file to load the 
        IMF instantaneous amplitudes ([time x nImfs] array called IA in emd package) for that region. These should be named
        according to the guidelines outlined in the user guide.
    dimInfo : dataframe
        Dataframe detailing the IMF index and region corrresponding to each IMF amplitude (1D index)
    pairwiseInds : ndarray
        [nDims_2d x 2] array. Each index row corresponds to the the pair of 1D indices used for the 2D index
    emdStr : string
        String to identify the variant of emd that was used to extract the imfs. e.g. '.eEMD'
    barcodeDir : string | None
        The directory path to save the detected barcode information. If None, barcode strengths will not be saved
    chunk : tuple
        In case of memory error, use chunking. The first element is a bool specifying whether chunking should be applied, and
        the second is the chunk length amount. If memory error persists, decrease this value.
      
    Returns
    -------
    barcodeStrengths : ndarray
        [nBarcodes x nTimeSamples] barcode strengths array
    """
    
    if barcodeDir is None:
        print('Warning: barcode strengths will not be written')
    
    sesh_str = sesh_strs[seshi]
    sesh_len = sesh_lens[seshi]

    nBarcodes = ics.shape[0]
    # check and format inputs
    region_paths2IAs = fix_region_paths2IAs(region_paths2IAs)
    # check same paths to IAs are the same for each region
    sesh_lens, sesh_strs, nSessions = check_region_paths2IAs(regions, region_paths2IAs, emdStr)
    
    barcodeStrengths = [[] for i in range(nBarcodes)]

    if chunk[0]:
        chunkBins = np.arange(0, sesh_len, chunk[1])
        if chunkBins[-1] < sesh_len:
            chunkBins = np.append(chunkBins, sesh_len)
        chunkBins = np.column_stack([chunkBins[:-1], chunkBins[1:]])
    else:
        chunkBins = np.column_stack([0, sesh_len])

    region_ia = {}
    for region in regions:
        region_ia[region] = np.load(region_paths2IAs[region][seshi])    
    dimAmps = np.row_stack([region_ia[info['region']][:, info['imfi']] for _, info in dimInfo.iterrows()])
    del region_ia

    try:
        for chSt, chEn in chunkBins:
            chLen = chEn-chSt
            dat4strengths = np.row_stack([np.multiply(dimAmps[x, chSt:chEn], dimAmps[y, chSt:chEn]) for x, y in pairwiseInds])
            for ici, ic in enumerate(ics):
                barcodeStrengths[ici].append(np.array([np.dot(ic, dati) for dati in dat4strengths.T]))
            del dat4strengths
        barcodeStrengths = np.row_stack([np.concatenate(barcodeStrengths[ici]) for ici in range(nBarcodes)])
    except MemoryError:
        print('Memory error: 2D data to load is too large')
        print('To fix, set the argument chunk=(True, 10000). If the problem persists, decrease the value of chunk[1] to 1000 | 100')
        return
    
    if barcodeDir is not None:
        saveStr = get_regionImf_saveStr(regions, region_imfis, emdStr)
        filename = sesh_str+saveStr+'.strengths'
        os.chdir(barcodeDir)
        np.save(filename, barcodeStrengths)
        print('wrote:', filename)
    
    return barcodeStrengths
    

def plot_barcode(barcode, regions, region_imfis, region_colors,
                 regFontsize=10, regGrid=True, cmap='RdBu_r', 
                 setvRange=(False, None, None), cbar=False, addLabs=True, xDimLabsRot=None, yDimLabsRot=None, mask_tri=False,
                 useDimLabs=False, dimLabsRight=True, dimLabsTop=True, dimFontsize=10):
    """
    Visualise a barcode.
    
    Parameters
    ----------
    barcode : ndarray
        [nIMFs x nIMFs]
    regions : ndarray | list
        The brain regions of interest
    region_imfis : dict
        Each region entry contains the IMF indices to be used
    region_colors : dict
        Each region entry contains a colour for that region
    regFontsize : int
        Fontsize for region labels (default = 10)
    regGrid : bool
        Apply gridlines to seperate regions (default = True)
    cmap : str
        The colourmap for the barcode weights. see the package ('seaborn') for more information (default = 'RdBu_r')
    setvRange: tuple
        Length 3 tuple specifying [0] if the z range should be set and [1:2] the minimum and maximum weight values for the color scale.
        If the first tuple element is False, then the min/max will be set to +/- the absolute max of all barcode weights (default).
    cbar : bool
        Should a color bar be shown
    addLabs : bool
        Should dimension labels be shown 
    xDimLabsRot : int | None
        the rotation of the x-labels
    yDimLabsRot : int | None
        the rotation of the y-labels
    mask_tri : bool
        Should only the upper triangle of the barcode will be shown (default = False)
    useDimLabs : bool
        Should the IMF dimension labels be shown instead of region labels (default = False)
    dimLabsRight : bool
        Show the y labels on the right or left
    dimLabsTop : bool
        Show the x labels on the top or bottom
    dimFontsize : int
        Fontsize for the IMF dimension labels
    """
    gridProps = {'col' : 'k', # color of the grid
                 'lw' : 2,  # grid linewidth
                 'colBord' : 'k', # border color
                 'lwBord' : 2} # border linewidth
        
    
    if mask_tri:
        barcode = np.tril(barcode, k=-1)
    
    reg_nDims = np.array([len(region_imfis[region]) for region in regions])

    nRegs = len(regions)
    
    regTits = []
    regCols = []
    dimLabs = []
    dimLabCols = []
    
    region_imfi2titFuncs = get_region_imfi2titFuncs(regions)

    for region in regions:
        regTits.append(region)
        regCols.append(region_colors[region])
        for imfi in region_imfis[region]:
            dimLabs.append(region+'_'+region_imfi2titFuncs[region](imfi))
            dimLabCols.append(region_colors[region])
    #
    if setvRange[0]:
        vmin = setvRange[1]
        vmax = setvRange[2]
    else:
        maxWeight = np.max(np.abs(barcode))
        vmin = -maxWeight
        vmax = maxWeight
    #
    sns.heatmap(barcode, cmap=cmap, square=True, vmin=vmin, vmax=vmax, cbar=cbar,
                xticklabels=False, yticklabels=False)
    #
    if addLabs:
        if useDimLabs:
            #
            labMids = np.array(range(len(dimLabs)))+0.5

            if dimLabsRight:
                if dimLabsTop:
                    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, 
                                    left=False, right=True, labelleft=False, labelright=True)
                else:
                    plt.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, 
                                    left=False, right=True, labelleft=False, labelright=True)

            else:
                if dimLabsTop:
                    plt.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, 
                                    left=True, right=False, labelleft=True, labelright=False)
                else:
                    plt.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True, 
                                    left=True, right=False, labelleft=True, labelright=False)

            if dimLabsTop:
                if xDimLabsRot is None:
                    xDimLabsRot=-90
            else:
                if xDimLabsRot is None:
                    xDimLabsRot = 90


            plt.xticks(labMids, dimLabs, rotation=xDimLabsRot, fontsize=dimFontsize)
            if yDimLabsRot is None:
                yDimLabsRot = 0
            plt.yticks(labMids, dimLabs, rotation=yDimLabsRot, fontsize=dimFontsize)
            for tickLab, col in zip(plt.gca().get_xticklabels(), dimLabCols):
                tickLab.set_color(col)
            for tickLab, col in zip(plt.gca().get_yticklabels(), dimLabCols):
                tickLab.set_color(col)

        else:
            midShifts = np.array([dimsPerReg/2. for dimsPerReg in reg_nDims])
            regMids = np.subtract(np.cumsum(reg_nDims), midShifts)
            plt.xticks(regMids, regTits, fontsize=regFontsize, fontweight='bold')
            for tickLab, regCol in zip(plt.gca().get_xticklabels(), regCols):
                tickLab.set_color(regCol)
            
            plt.yticks(regMids, regTits, fontsize=regFontsize, fontweight='bold')
            for arg, f in zip([xDimLabsRot, yDimLabsRot], [plt.xticks, plt.yticks]):
                if arg is not None:
                    f(rotation=arg)
            for tickLab, regCol in zip(plt.gca().get_yticklabels(), regCols):
                tickLab.set_color(regCol)
    
    indMin = 0
    indMax = reg_nDims.sum()

    plt.xlim(indMin, indMax)
    plt.ylim(indMin, indMax)
    if regGrid:
        # plot inside grid
        regSeps = np.concatenate([[0], np.cumsum(reg_nDims)])
        for sep in regSeps:
            plt.hlines(y=sep, xmin=0, xmax=np.size(barcode, 0), color=gridProps['col'], lw=gridProps['lw'])
            plt.vlines(x=sep, ymin=0, ymax=np.size(barcode, 1), color=gridProps['col'], lw=gridProps['lw'])

        #
        plt.hlines(y=[indMin, indMax], xmin=indMin, xmax=indMax, color=gridProps['colBord'], lw=gridProps['lwBord'])
        plt.vlines(x=[indMin, indMax], ymin=indMin, ymax=indMax, color=gridProps['colBord'], lw=gridProps['lwBord'])
        
        