import mne
import torch
import warnings
import numpy as np

class FixedCrop(object):
    """Crop the EEG in a sample from a given start point.

    Parameters
    --------------
    output_size : int
        Desired output size (on time axis).
    start : int
        Start point for crop operation. It is relative to the observation i.e. *start* = 0 is the start sample.
    """

    def __init__(self, output_size, start=0):
        assert isinstance(output_size, int)
        assert isinstance(start, int)
        if isinstance(output_size, int):
            self.output_size = output_size 
        if isinstance(start, int):
            self.start = start

    def __call__(self, sample):
        eeg, label= sample['eeg'], sample['label']
        
        # Check eeg instance type
        is_eeg_numpy = isinstance(eeg, np.ndarray)

        d = eeg.shape[1] if is_eeg_numpy else eeg[:][0].shape[1] 
        new_d = self.output_size
        if (self.start + self.output_size)>=d:
            raise ValueError("start + output_size exceeds the sample length")

        start = self.start
        stop = start + self.output_size

        eeg = eeg[:, start:stop] if is_eeg_numpy else eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'eeg': eeg, 'label': label}
    
class RandomCrop(object):
    """Crop randomly the EEG in a sample.

    Parameters
    --------------
    output_size : int
        Desired output size (on time axis).
    """

    def __init__(self, output_size):
        assert isinstance(output_size, int)
        if isinstance(output_size, int):
            self.output_size = output_size

    def __call__(self, sample):
        eeg, label= sample['eeg'], sample['label']

        # Check eeg instance type
        is_eeg_numpy = isinstance(eeg, np.ndarray)

        d = eeg.shape[1] if is_eeg_numpy else eeg[:][0].shape[1]
        new_d = self.output_size
        if new_d>=d:
            raise ValueError("output_size exceeds the sample length")

        start = np.random.randint(0, d - new_d)
        stop = start + self.output_size

        eeg = eeg[:, start:stop] if is_eeg_numpy else eeg.crop(tmin=eeg.times[start], tmax=eeg.times[stop],  include_tmax=False)
        
        return {'eeg': eeg, 'label': label}

class PickData(object):
    """Pick only EEG channels in raw data. Use this transform only if the eeg is contains all the channels (76)
    """
    
    def __call__(self, sample):
        eeg, label= sample['eeg'], sample['label']
        #set montage
        data_chans = eeg.ch_names[4:36]
        eeg.pick_channels(data_chans)
        
        return {'eeg': eeg, 'label': label}
    
class SetMontage(object):
    """Set 10-20 montage to the Raw object. Use this transform only if the eeg is a mne.RawBase object.
    """
    
    def __call__(self, sample):
        eeg, label= sample['eeg'], sample['label']
        #set montage
        ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
        eeg.set_montage(ten_twenty_montage)
        
        return {'eeg': eeg, 'label': label}
    
class ToArray(object):
    """Convert the eeg file in a sample to numpy.ndarray."""

    def __call__(self, sample):
        eeg, label= sample['eeg'], sample['label']
        # discard times, select only array data
        if not isinstance(eeg, np.ndarray):
            eeg = eeg[:][0]
        else:
            warnings.warn("This operation is unuseful, since your data are alreay in numpy format")
        return {'eeg': eeg, 'label': label}
    
class ToTensor(object):
    """Convert the eeg in a sample from numpy.ndarray format to torch.Tensor.
    It should be inkoed as last transform.
    
    Parameters
    -------------
    interface : str
        Could be 'dict' or 'unpacked_values'. It changes the output interface.
    eeg_tensor_type : str
        Could be 'float32' or 'foat64'. It changes the tensor type.
    label_interface : str
        Could be 'tensor' or 'long'. It changes the label type.
    """
    
    def __init__(self, interface='dict', eeg_tensor_type = 'float32', label_interface='tensor'):
        self.interfaces = ['dict', 'unpacked_values']
        self.eeg_tensor_types = ['float64', 'float32']
        self.label_interfaces = ['tensor', 'long']
        
        assert isinstance(interface, str)
        if interface not in self.interfaces:
            raise ValueError("interface must be one of " + str(self.interfaces))
        if isinstance(interface, str):
            self.interface = interface
            
        assert isinstance(eeg_tensor_type, str)
        if eeg_tensor_type not in self.eeg_tensor_types:
            raise ValueError("eeg_tensor_type must be one of " + str(self.eeg_tensor_types))
        if isinstance(eeg_tensor_type, str):
            self.eeg_tensor_type = eeg_tensor_type
        
        assert isinstance(label_interface, str)
        if label_interface not in self.label_interfaces:
            raise ValueError("label_interface must be one of " + str(self.label_interfaces))
        if isinstance(label_interface, str):
            self.label_interface = label_interface

    def __call__(self, sample):
        eeg, label= sample['eeg'], sample['label']
        
        if self.eeg_tensor_type=='float32':
            eeg = eeg.astype(np.float32)
        eeg = torch.from_numpy(eeg)
            
        if self.label_interface=='tensor':
            label = torch.LongTensor([label])
        
        if self.interface=='dict':
            return {'eeg': eeg, 'label': label}
        elif self.interface=='unpacked_values':
            return eeg, label
        
class Standardize(object):
    """
    Standardize the EEG data in a sample.
    """
    
    def __call__(self, sample):
        eeg, label= sample['eeg'], sample['label']
        
        mean = eeg.mean(axis=1, keepdims=True)
        std = eeg.std(axis=1, keepdims=True)
        eeg = (eeg - mean) / std
        
        return {'eeg': eeg, 'label': label}
