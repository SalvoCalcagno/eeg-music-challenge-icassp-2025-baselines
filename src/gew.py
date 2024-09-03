import re
import sys
import warnings
import numpy as np
import pandas as pd
from typing import NamedTuple
import matplotlib.pyplot as plt

if (sys.version_info.minor<8):
    def prod(iterable, start=1):
        print('custom')
        res = start
        for i in iterable:
            res = res * i
        return res
else:
    from math import prod

#***************
#PART 1: GEW definition and utils
#***************

#In this part we define the gew object and the main interface.
#Functions to load gew emotions from rating files are provided.
#The main interface of gew emotion is a tuple of type *(EMOTION, INTENSITY)*

# emotions' dictionaries [ITALIAN]
emotions = {
    0: 'Interesse',
    1: 'Divertimento',
    2: 'Orgoglio',
    3: 'Gioia',
    4: 'Piacere',
    5: 'Contentezza',
    6: 'Amore',
    7: 'Ammirazione',
    8: 'Sollievo',
    9: 'Compassione',
    10: 'Tristezza',
    11: 'Colpa',
    12: 'Rimpianto',
    13: 'Vergogna',
    14: 'Delusione',
    15: 'Paura',
    16: 'Disgusto',
    17: 'Disprezzo',
    18: 'Odio',
    19: 'Rabbia',
    20: 'NO EMOTION FELT',
    21: 'DIFFERENT EMOTION FELT'
}

# emotions' dictionaries [ENGLISH]
emotions = {
    0: 'Interest',
    1: 'Amusement',
    2: 'Pride',
    3: 'Joy',
    4: 'Pleasure',
    5: 'Contentment',
    6: 'Love',
    7: 'Admiration',
    8: 'Relief',
    9: 'Compassion',
    10: 'Sadness',
    11: 'Guilt',
    12: 'Regret',
    13: 'Shame',
    14: 'Disappointment',
    15: 'Fear',
    16: 'Disgust',
    17: 'Contempt',
    18: 'Hate',
    19: 'Anger',
    20: 'NO EMOTION FELT',
    21: 'DIFFERENT EMOTION FELT'
}

reverse_emotions = {v: k for k, v in emotions.items()}

vd_coordinates = {
    0: (0.61, 0.25),
    1: (0.67, 0.19),
    2: (0.72, 0.15),
    3: (0.68, 0.07),
    4: (0.71, 0.02),
    5: (0.77, -0.03),
    6: (0.58, -0.16),
    7: (0.66, -0.09),
    8: (0.66, -0.36),
    9: (-0.05, -0.55),
    10: (-0.68, -0.35),
    11: (-0.57, -0.27),
    12: (-0.70, -0.19),
    13: (-0.61, -0.16),
    14: (-0.77, -0.12),
    15: (-0.61, 0.07),
    16: (-0.68, 0.20),
    17: (-0.55, 0.43),
    18: (-0.45, 0.43),
    19: (-0.37, 0.47),
    20: (0, 0),
    21: None
}

# convert a rating row in to a list of gew tuples
def dumps(rating):
    """
    It converts a rating row into a list of tuples. Each tuple is in *(emotion_id, intensity)* format.
    
    Parameters
    ------------
    rating : Union[pandas.core.series.Series, dict]
        a rating row or a dictionary. It must be contain emotion families *gew_fam1* and *gew_fam2* as strings and emotion intensities *gew_int1* and *gew_int2* as integers.
       
    Returns
    -------------
    list
        A list of tuples. Each tuple has got *emotion identifier* and *emotion intensity*.
    
    Examples
    ------------
    
    >>> rating = {
    ...    'gew_fam1': 'Interesse',
    ...    'gew_int1': 4,
    ...    'gew_fam2': 'Tristezza',
    ...    'gew_int2': 2
    ... }
    >>> dumps(rating)
    [(0, 4), (10, 2)]
    
    """
    if type(rating)!=pd.core.series.Series and type(rating)!=dict:
        raise TypeError("Rating must be pd.core.series.Series or dict, not" + str(type(rating)))
    # access ratings once
    fam1 = rating['gew_fam1']
    int1 = rating['gew_int1']
    fam2 = rating['gew_fam2']
    int2 = rating['gew_int2']
    # type checking
    if(type(fam1)!=str):
        raise TypeError("Rating family (first emotion) is expected to be a string, " + str(type(fam1)) + " received!")
    if(type(fam2)!=str):
        raise TypeError("Rating family (second emotion) is expected to be a string, " + str(type(fam2)) + " received!")
    # check different emotion for fam1
    if(re.search(emotions[21], fam1)!=None):
        different_emotion = fam1.split(' - ')
        fam1 = different_emotion[0]
        different_emotion = different_emotion[1]
        print("A different emotion was detected: " + different_emotion)
    # convert first rating
    gew1 = (reverse_emotions[fam1], int1)
    # check the presence of second emotion
    if(fam2=='e'):
        return [gew1, None]
    else:
        # check different emotion for fam2
        if(re.search(emotions[21], fam2)!=None):
            different_emotion = fam2.split(' - ')
            fam2 = different_emotion[0]
            different_emotion = different_emotion[1]
            print("A different emotion was detected: " + different_emotion)
        # convert second rating
        gew2 = (reverse_emotions[fam2], int2)
        return [gew1, gew2]

# convert a list of gew tuples in a dataframe 
def loads(rating):
    """
    It converts a a list of gew tuples into a pandas dataframe.
    
    Parameters
    ------------
    rating : list of tuple
        A list of tuples. Each tuple myst be in *(emotion_id, intensity)* format.
       
    Returns
    -------------
    pandas.core.frame.DataFrame
        A dataframe containing the rating. Dataframe contains emotion families *gew_fam1* and *gew_fam2* (strings) and emotion intensities *gew_int1* and *gew_int2* (integers) as columns.
    
    Examples
    ------------
    >>> rating = [(13, 2), (15, 4)]
    >>> loads(rating)
    	gew_fam1	gew_int1	gew_fam2	gew_int2
    0	Vergogna	2		Paura		4
    """
    
    if type(rating) is not list:
        raise TypeError("Rating must be list, not" + str(type(rating)))
    if len(rating)<2:
        raise Exception("Rating must be a list of two tuples, only " + str(len(rating)) + " element/s were given")
    elif len(rating)>2:
        warnings.warn("Only the first two ratings are used for computation, the other " + str(len(rating) - 2) + " element/s will be ignored")
    if type(rating[0]) is not tuple or (type(rating[1]) is not tuple and rating[1]!=None):
        raise TypeError("Rating items must be tuples, not" + str(type(rating[0])))
                      
    gew1 = rating[0]
    gew2 = rating[1]
    result = {}
    result['gew_fam1'] = emotions[gew1[0]]
    result['gew_int1'] = gew1[1]
    if(gew2!=None):
        result['gew_fam2'] = emotions[gew2[0]]
        result['gew_int2'] = gew2[1]
    else:
        result['gew_fam2'] = 'e'
        result['gew_int2'] = 0
    return pd.DataFrame(result, index=[0])

class GEW(NamedTuple):
    """
    A class representig the Geneva Emotion Wheel rating. 
    It is a named tuple, thus you can access the rating by brackets or by dotted notation.
    
    Arguments
    --------------
    emotion : str
        a string representing the felt emotion in human readble format.
    intensity : int
        a value representing emotion intensity (or arousal). It could be in [0,5] range.
        
    Examples
    ---------------
    >>> rating = GEW("Interesse", 4)
    >>> #access GEW by attributes by dotted notation
    >>> rating.emotion
    Interesse
    >>> #access GEW by indices by brackets notation
    >>> rating[1]
    4
       
    """
    emotion: str
    intensity: int

# convert a GEW object in a tuple
def dump(gew):
    """
    It converts a GEW object in a *(emotion_id, intensity)* formatted tuple. 
    
    Parameters
    ------------
    gew : gew.GEW
        The GEW object to be converted
       
    Returns
    -------------
    tuple
        A *(emotion_id, intensity)* formatted tuple.
    """
    if gew.emotion == 'e':
        return None
    if re.search(emotions[21], gew.emotion):
        return (21, gew.intensity)
    return (reverse_emotions[gew.emotion], gew.intensity)
    
# convert a tuple in a GEW object
def load(t_gew : tuple):
    """
    It converts a *(emotion_id, intensity)* formatted tuple in GEW object. 
    
    Parameters
    ------------
    t_gew : tuple
        A *(emotion_id, intensity)* formatted tuple.
        
       
    Returns
    -------------
    gew.GEW
        The converted GEW object
    """
    return GEW(emotions[t_gew[0]], t_gew[1])

#*********************
#PART 2: GEW conversions
#*********************

#In this part we expose functions in order to:
#- transform gew objects into class ids
#- plot data distributions, given a transform function and a ds
#- get data distribution, given a transform function

def _checkGewEmotionFormat(gew_emotion, neutral_allowed=True, different_allowed=True, single_id_allowed=True):
    #check type
    if single_id_allowed and type(gew_emotion)!=tuple and type(gew_emotion)!=int:
        raise TypeError("Not Valid Emotion. Please provide emotion in gew format (Emotion, Intensity) or an iteger emotion id.")
    if not single_id_allowed and type(gew_emotion)!=tuple:
        raise TypeError("Not Valid Emotion. Please provide emotion in gew format (Emotion, Intensity).")
    #check tuple format
    if isinstance(gew_emotion, tuple):
        #check length
        if len(gew_emotion)<2:
            raise Exception("Not Valid Emotion. Please provide emotion in gew format (Emotion, Intensity).")
        #check types
        if type(gew_emotion[0])!=int:
            raise TypeError("(Emotion, Intensity) format accepts only integers, not " + str(type(gew_emotion[0])) + ".")
        if type(gew_emotion[1])!=int:
            raise TypeError("(Emotion, Intensity) format accepts only integers, not " + str(type(gew_emotion[1])) + ".")
        emotion = gew_emotion[0]
        intensity = gew_emotion[1]
        #check intensity values
        if intensity>5 or intensity<0:
            raise ValueError("Intensity values must be in range [0, 5], you provided " + str(intensity) + ".")
    else:
        if type(gew_emotion)!=int:
            raise TypeError("Emotion ID must be integer, not " + str(type(gew_emotion[0])) + ".")
        emotion = gew_emotion
    #check emotion values
    if not different_allowed and not neutral_allowed:
        correct_value_condition = emotion in range(20)
        s = '[0, 19]'
    elif not different_allowed and neutral_allowed:
        correct_value_condition = emotion in range(21)
        s = '[0, 20]'
    elif different_allowed and neutral_allowed:
        correct_value_condition = emotion in range(22)
        s = '[0, 21]' 
    elif different_allowed and not neutral_allowed:
        correct_value_condition = emotion in range(20) or emotion==21
        s = '[0, 19] U 21'   
    if not correct_value_condition:
        raise ValueError("Emotion values must be in range " + s + ", you provided " + str(emotion) + ".")
    
# CONVERSION OF GEW EMOTIONS INTO VALENCE-AROUSAL-DOMINANCE MODEL
def vad_coordinates(gew_emotion):
    """
    It Converts a gew emotion tuple *(EMOTION, INTENSITY)* to Valence-Arousal-Dominance coordinates.
    
    Parameters
    ------------
    gew_emotion : (int, int) 
         gew emotion tuple *(EMOTION, INTENSITY)*. *EMOTION* is emotion identifier. 
         You can use `gew.emotions` dictionary to extract the emotion associated with the given id. 
         *INTENSITY* is the emotion intensity associated to *EMOTION*.
         
    Returns
    ------------
    (float, float, float)
        Valence-Arousal-Dominance coordinates (V, A, D). All values are expressed in [-1, 1] range.
    """
    _checkGewEmotionFormat(gew_emotion, neutral_allowed=True, different_allowed=False, single_id_allowed=False)
    V, D = vd_coordinates[gew_emotion[0]]
    A = round(2*gew_emotion[1]/5 - 1, 2)
    
    return V, A, D

# CONVERSION OF GEW EMOTIONS INTO CLASS IDs

def gew_to_hldv4(gew_emotion, min_arousal=0):
    """
    It Converts a gew emotion tuple *(EMOTION, INTENSITY)* to High/Low Dominance/Valence.
    Neutral class (*NO EMOTION FELT*) is not considered as a class.
    Other classes (*DIFFERENT EMOTIONS*) are not considered as a class.
    
    Dominance is considered Low in range [5, 15[, High in range [15, 5[
    Valence is considered Low in range [10, 20[, High in range [0, 10[
    VD graph is thus divided into 4 different areas:
    
        - HDHV (High Dominance, High Valence)
        - LDHV (Low Dominance, High Valence)
        - LDLV (Low Dominance, Low Valence)
        - HDLV (High Dominance, Low Valence)
        
    Parameters
    ------------
    gew_emotion : Union[int, (int, int)] 
         gew emotion tuple *(EMOTION, INTENSITY)*. *EMOTION* is emotion identifier. 
         You can use `gew.emotions` dictionary to extract the emotion associated with the given id. 
         *INTENSITY* is the emotion intensity associated to *EMOTION*.
         You can also pass only *EMOTION*, without specifying the respective intensity.
    min_arousal : int
        The minimal intensity level to consider emotion as non-neutral. Do not use it: always ignored, used only to standardize conversion interface.
         
    Returns
    ------------
    int
        A class identifier:
            0. HDHV
            1. LDHV
            2. LDLV
            3. HDLV
    """
    _checkGewEmotionFormat(gew_emotion, neutral_allowed=False, different_allowed=False, single_id_allowed=True)
    if isinstance(gew_emotion, int):
        emotion = gew_emotion
    elif isinstance(gew_emotion, tuple):
        emotion = gew_emotion[0]
    # HDHV
    if emotion<5:
        return 0
    # LDHV
    elif emotion>=5 and emotion<10:
        return 1
    # LDLV
    elif emotion>=10 and emotion<15:
        return 2
    # HDLV
    elif emotion>=15 and emotion<20:
        return 3
    
def gew_to_hldv5(gew_emotion, min_arousal=3):
    """
    It Converts a gew emotion tuple *(EMOTION, INTENSITY)* to High/Low Dominance/Valence.
    Neutral class (*NO EMOTION FELT*) is considered as a separate class.
    Emotions with low *INTENSITY* (arousal) are considered neutral emotions.
    The minimal intensity level to consider emotion as non-neutral is specified by `min_arousal` parameter.
    Other classes (*DIFFERENT EMOTIONS*) are not considered as a class.
    
    Dominance is considered Low in range [5, 15[, High in range [15, 5[
    Valence is considered Low in range [10, 20[, High in range [0, 10[
    VD graph is thus divided into 4 different areas:
    
        - NEUTRAL (Low Intensity Values)
        - HDHV (High Dominance, High Valence)
        - LDHV (Low Dominance, High Valence)
        - LDLV (Low Dominance, Low Valence)
        - HDLV (High Dominance, Low Valence)
        
    Parameters
    ------------
    gew_emotion : (int, int) 
         gew emotion tuple *(EMOTION, INTENSITY)*. *EMOTION* is emotion identifier. 
         You can use `gew.emotions` dictionary to extract the emotion associated with the given id. 
         *INTENSITY* is the emotion intensity associated to *EMOTION*.
    min_arousal : int
        The minimal intensity level to consider emotion as non-neutral.
         
    Returns
    ------------
    int
        A class identifier:
            0. NEUTRAL
            1. HDHV
            2. LDHV
            3. LDLV
            4. HDLV
    """
    _checkGewEmotionFormat(gew_emotion, neutral_allowed=True, different_allowed=False, single_id_allowed=False)
    emotion = gew_emotion[0]
    arousal = gew_emotion[1]
    # Neutral emotions
    if emotion==20 or arousal<min_arousal:
        return 0
    # HDHV
    elif emotion<5:
        return 1
    # LDHV
    elif emotion>=5 and emotion<10:
        return 2
    # LDLV
    elif emotion>=10 and emotion<15:
        return 3
    # HDLV
    elif emotion>=15 and emotion<20:
        return 4
    
def gew_to_8(gew_emotion, use_neutral=False, use_different=False, min_arousal=2):
    """
    It Converts a gew emotion tuple *(EMOTION, INTENSITY)* to either 8, 9 or 10 , depending on paramters, 
    custom classes used in the first pahse of the experiment (data gathering).
    
    Parameters
    ------------
    gew_emotion : Union[int, (int, int)] 
         gew emotion tuple *(EMOTION, INTENSITY)*. *EMOTION* is emotion identifier. 
         You can use `gew.emotions` dictionary to extract the emotion associated with the given id. 
         *INTENSITY* is the emotion intensity associated to *EMOTION*.
         You can also pass only *EMOTION*, without specifying the respective intensity. In this case `min_arousal` is ignored.
    use_neutral : bool
        if True use *NO EMOTION FELT* as neutral class.
        Emotions with intensity levels less than `min_arousal` are also considered as NEUTRAL emotions.
        if False, *NO EMOTION FELT* emotions are discarded as incorrect.
    use_different : bool
        if True use *DIFFERENT EMOTION FELT* as a different class or (TODO) map into other classes.
        if False *DIFFERENT EMOTION FELT* emotions are discarded as incorrect.
    min_arousal : int
        The minimal intensity level to consider emotion as non-neutral. It is ignored if `gew_emotion` is int.
    
    Returns
    ------------
    int
        A class identifier:
            0. Interesse-Divertimento-Orgoglio
            1. Gioia-Piacere
            2. Contentezza-Amore-Ammirazione
            3. Sollievo-Compassione
            4. Tristezza
            5. Delusione-Vergogna-Rimpianto-Colpa
            6. Paura
            7. Disgusto-Disprezzo-Odio-Rabbia
            8. NO EMOTION FELT
            9. DIFFERENT EMOTION FELT
    """
    _checkGewEmotionFormat(gew_emotion, neutral_allowed=use_neutral, different_allowed=use_different, single_id_allowed=True)
    if isinstance(gew_emotion, int):
        emotion = gew_emotion
        arousal = min_arousal     
    elif isinstance(gew_emotion, tuple):
        emotion = gew_emotion[0]
        arousal = gew_emotion[1]

    if emotion == 21:
        return 9
    elif use_neutral and (emotion == 20 or arousal < min_arousal):
        return 8
    elif emotion in range(0, 3):
        return 0
    elif emotion in range(3, 5):
        return 1
    elif emotion in range(5, 8):
        return 2
    elif emotion in range(8, 10):
        return 3
    elif emotion == 10:
        return 4
    elif emotion in range(11, 15):
        return 5
    elif emotion == 15:
        return 6
    elif emotion in range(16, 20):
        return 7
        
def gew_to_6a(gew_emotion, min_arousal=0):
    """
    It Converts a gew emotion tuple *(EMOTION, INTENSITY)* to 6 intensity classes.
    All different *INTENSITY* levels are mapped into different classes.
    
    Parameters
    ------------
    gew_emotion : Union[int, (int, int)] 
         gew emotion tuple *(EMOTION, INTENSITY)*. *EMOTION* is emotion identifier. 
         You can use `gew.emotions` dictionary to extract the emotion associated with the given id. 
         *INTENSITY* is the emotion intensity associated to *EMOTION*.
    min_arousal : int
        The minimal intensity level to consider emotion as non-neutral. Do not use it: always ignored, used only to standardize conversion interface.
    
    Returns
    ------------
    int
        A class identifier corresponding to the given intensity level. It is in range [0, 5].
    """
    _checkGewEmotionFormat(gew_emotion, neutral_allowed=False, different_allowed=False, single_id_allowed=False)
    return gew_emotion[1]
        
def gew_to_5a(gew_emotion, min_arousal=0):
    """
    It Converts a gew emotion tuple *(EMOTION, INTENSITY)* to 5 intensity classes.
    *INTENSITY* levels 0 and 1 are mapped into a single class.
    All *INTENSITY* levels grater than 1 are mapped into different classes. 
    
    Parameters
    ------------
    gew_emotion : Union[int, (int, int)] 
         gew emotion tuple *(EMOTION, INTENSITY)*. *EMOTION* is emotion identifier. 
         You can use `gew.emotions` dictionary to extract the emotion associated with the given id. 
         *INTENSITY* is the emotion intensity associated to *EMOTION*.
    min_arousal : int
        The minimal intensity level to consider emotion as non-neutral. Do not use it: always ignored, used only to standardize conversion interface.
    
    Returns
    ------------
    int
        A class identifier corresponding to the given intensity level. It is in range [0, 4].
    """
    _checkGewEmotionFormat(gew_emotion, neutral_allowed=False, different_allowed=False, single_id_allowed=False)
    arousal = gew_emotion[1]
    if arousal == 0:
        return 0
    else:
        return arousal - 1
        
def gew_to_emotion(gew_emotion, min_arousal=0):
    """
    It Converts a gew emotion tuple *(EMOTION, INTENSITY)* 20 emotion classes.
    Neutral class (*NO EMOTION FELT*) is considered as a separate class.
    Emotions with low *INTENSITY* (arousal) are considered neutral emotions.
    The minimal intensity level to consider emotion as non-neutral is specified by `min_arousal` parameter.
    Other classes (*DIFFERENT EMOTIONS*) are considered as a class.
    
    Parameters
    ------------
    gew_emotion : (int, int) 
         gew emotion tuple *(EMOTION, INTENSITY)*. *EMOTION* is emotion identifier. 
         You can use `gew.emotions` dictionary to extract the emotion associated with the given id. 
         *INTENSITY* is the emotion intensity associated to *EMOTION*.
    min_arousal : int
        The minimal intensity level to consider emotion as non-neutral.
        
    Returns
    ------------
    int
        A class identifier. You can use `gew.emotions` dictionary to extract the emotion associated with the returned identifier.
    """
    _checkGewEmotionFormat(gew_emotion, neutral_allowed=True, different_allowed=True, single_id_allowed=False)
    emotion = gew_emotion[0]
    arousal = gew_emotion[1]
    if emotion==21:
        return 21
    elif arousal<min_arousal:
        return 20
    else:
        return emotion

# GET DATA DISTRIBUTIONS
def get_data_distribution(gew_labels, num_classes, transform_function, **args):
    """
    Get data distribution of data, given original gew labels and a transform function.
    
    Parameters
    ------------
    gew_labels : list of tuple
        Dataset labels. Please provide a list of gew emotion tuple in the format *(EMOTION, INTENSITY)*.
    num_classes : int 
        Number of classes obtained with `transform_function`
    transform_function : function
        Function to be used to convert `gew_labels` into a list of class identifiers. You can use one of `gew_to_xxx` functions of the present module, but also a custom function.
    **args 
        Named args to be passed to `transform_function`
    
    Returns
    ------------
    list of int
        List of size `num_classes`, whose i-th entry is the number of occurencies in the dataset that has been mapped to i-th class.
    """
    
    new_labels = []
    for gew_emotion in gew_labels:
        new_labels.append(transform_function(gew_emotion, **args))
    return [new_labels.count(i) for i in range(num_classes)]

# PLOT DATA DISTRIBUTIONS
def plot_data_distribution(gew_labels, transform_function, normalize=True, verbose=False, **args):
    """
    Plot data distributions of data, given original gew labels and a transform function.
    If `transform_function` output depends on arousal parameter, data are plotted with all possible `min_arousal` values
    in *n* different bar plots, one for each `min_arousal` value.
    
    Parameters
    ----------------
    gew_labels : list of tuple
        Dataset labels. Please provide a list of gew emotion tuple in the format *(EMOTION, INTENSITY)*.
    verbose : bool
        if True, print some additional infos (used for debug).
    normalize : bool
        if True, normalize output counts into [0, 1] range.
    transform_function : function
        Function to be used to convert `gew_labels` into a list of class identifiers. You can use one of `gew_to_xxx` functions of the present module, but also a custom function.
    **args 
        Named args to be passed to `transform_function`
    """
    
    # configure subplots, fig_size, ticks and plt params
    if transform_function==gew_to_hldv4:
        grid = (1, 1)
        classes = ['HDHV','LDHV','LDLV','HDLV']
    elif transform_function==gew_to_hldv5:
        grid = (2, 3)
        classes = ['N', 'HDHV','LDHV','LDLV','HDLV']
    elif transform_function==gew_to_8:
        grid = (2, 3)
        classes = ['I','G','Cn','S','T','Cl','P','Dg','N','D']
    elif transform_function==gew_to_5a:
        grid = (1, 1)
        classes = [x for x in range(1, 6)]
    elif transform_function==gew_to_6a:
        grid = (1, 1)
        classes = [x for x in range(6)]
    elif transform_function==gew_to_emotion:
        grid = (2, 3)
        classes = [x for x in range(22)]
    else: 
        raise Exception('transform_function not valid')
    num_classes = len(classes)
    fig = plt.figure(figsize=(20,10))
    
    for min_arousal in range(0, prod(grid)):
        if min_arousal>6:
            break
        new_labels = []
        for gew_emotion in gew_labels:
            #new_labels.append(gew_to_8(gew_emotion, min_arousal=min_arousal, use_different=True, use_neutral=True))
            #new_labels.append(gew_to_hldv5(gew_emotion, min_arousal=min_arousal))
            new_labels.append(transform_function(gew_emotion, min_arousal=min_arousal, **args))
        new_labels_count = [new_labels.count(i) for i in range(num_classes)]
        if verbose:
            print('DATA DISTRIBUTION. Function: ' + str(transform_function) + ' Min Arousal: ' + str(min_arousal))
            for class_id in range(num_classes):
                  print('Class ' + str(class_id) + ': ' + str(new_labels_count[class_id])) 
        
        fig.add_subplot(grid[0], grid[1], min_arousal + 1)
        x = np.linspace(0,num_classes-1,num_classes,endpoint=True)
        width = 1  # the width of the bars
        plt.xticks(x, tuple(classes))
        if normalize:
            plt.bar(x, [new_label_count/len(new_labels) for new_label_count in new_labels_count])
        else: 
            plt.bar(x, new_labels_count)
        plt.title(min_arousal)

def plot_data_distribution_grouped(gew_labels, transform_function, verbose = False, **args):
    """
    Plot data distributions of data, given original gew labels and a transform function.
    If `transform_function` output depends on arousal parameter, data are plotted with all possible `min_arousal` values
    in a single grouped bar plot.
    
    Parameters
    ----------------
    gew_labels : list of tuple
        Dataset labels. Please provide a list of gew emotion tuple in the format *(EMOTION, INTENSITY)*.
    verbose : bool
        if True, print some additional infos (used for debug).
    transform_function : function
        Function to be used to convert `gew_labels` into a list of class identifiers. You can use one of `gew_to_xxx` functions of the present module, but also a custom function.
    **args 
        Named args to be passed to `transform_function`
    """
    
    # configure subplots, fig_size, ticks and plt params
    if transform_function==gew_to_hldv4:
        grid = (1, 1)
        classes = ['HDHV','LDHV','LDLV','HDLV']
    elif transform_function==gew_to_hldv5:
        grid = (2, 3)
        classes = ['N', 'HDHV','LDHV','LDLV','HDLV']
    elif transform_function==gew_to_8:
        grid = (2, 3)
        classes = ['I','G','Cn','S','T','Cl','P','Dg','N','D']
    elif transform_function==gew_to_5a:
        grid = (1, 1)
        classes = [x for x in range(1, 6)]
    elif transform_function==gew_to_6a:
        grid = (1, 1)
        classes = [x for x in range(6)]
    elif transform_function==gew_to_emotion:
        grid = (2, 3)
        classes = [x for x in range(22)]
    else: 
        raise Exception('transform_function not valid')
    num_classes = len(classes)
    
    for min_arousal in range(6):
        if min_arousal>6:
            break
        new_labels = []
        for gew_emotion in gew_labels:
            #new_labels.append(gew_to_8(gew_emotion, min_arousal=min_arousal, use_different=True, use_neutral=True))
            #new_labels.append(gew_to_hldv5(gew_emotion, min_arousal=min_arousal))
            new_labels.append(transform_function(gew_emotion, min_arousal=min_arousal, **args))
        new_labels_count = [new_labels.count(i) for i in range(num_classes)]
        if verbose:
            print('DATA DISTRIBUTION. Function: ' + str(transform_function) + ' Min Arousal: ' + str(min_arousal))
            for class_id in range(num_classes):
                  print('Class ' + str(class_id) + ': ' + str(new_labels_count[class_id])) 
        
        x = np.linspace(0,num_classes-1,num_classes,endpoint=True)
        width = 1/6 - 0.01 # the width of the bars
        graph = plt.bar(x - (2.5 - min_arousal)*width, new_labels_count, width, label=min_arousal)
        plt.xticks(x, tuple(classes))
