#!usr/bin/env python
"""
Classes and methods used to create a background subtraction of a given image

Nick Walkden, May 2015
"""

import numpy as np
#import cv2
#from pyFastcamTools.create_log import create_log
from frame_history import frame_history
#from pyFastcamTools.Frames import Frame


class background_subtractor:
    """
    Class mimicing opencv background subtractors

    Develop a background model from a frame history

    Assume that the input is a greyscale image

    """

    def __init__(self, history=None):

        self._history = frame_history()
        self.background_model = None
        self._STATIC_HISTORY = False
        #self._history.N = 10  # Use 10 frames by default
        if history is not None:
            self.set_history(history)

    def apply(self, frame, get_foreground=True,non_negative=False):

        # First add the current frame to the history
        if not self._STATIC_HISTORY:
            # print('Adding frame to bgsub history')
            self._history.add_frame(frame, allow_duplicate=False)

        if get_foreground:
            self.get_background()
            foreground = frame[:] - self.background_model
            if non_negative: foreground[foreground < 0.0] = 0
            return foreground

    def set_history(self, history):

        if isinstance(history, (int, long, float)):
            self._history.loop_frames = history  # Number of frames to store in history
        else:
            self._history.set(history)  # Set the history to the given frames and do not reset
            self._STATIC_HISTORY = True


class background_subtractor_median(background_subtractor):
    """
    Take the median of each pixel in the frame history
    """

    def __init__(self, history=None):
        background_subtractor.__init__(self)

    def get_background(self):
        self.background_model = np.median(self._history.frames, axis=0)


class background_subtractor_min(background_subtractor):
    """
    Take the median of each pixel in the frame history
    """

    def __init__(self, history=None):
        background_subtractor.__init__(self, history)
        
    def get_background(self):
        self.background_model = np.min(self._history.frames, axis=0)


class background_subtractor_mean(background_subtractor):
    """
    Take the mean of each pixel in the frame history
    """

    def __init__(self, history=None):
        background_subtractor.__init__(self)
        
    def get_background(self):
        self.background_model = np.mean(self._history.frames, axis=0)


class background_subtractor_FFT(background_subtractor):
    def __init__(self, history=None):
        background_subtractor.__init__(self)

    def get_background(self):
        if self._history.nt < 2:
            self.background_model = 0.0
            return
        Rfft = np.fft.rfft(self._history.frames, axis=0)
        # zero out all but DC and Nyquist component
        Rfft[2:-2, ...] = 0.0

        result = np.fft.irfft(Rfft, axis=0)
        self.background_model = result[-1, ...]

# class backgroundSubtractorSVD(backgroundSubtractor):

def run_bgsub_min(frames,Nbg,return_bg=False):
    nt,nx,ny = frames.shape
    
    temp = np.empty((nt,nx,ny,Nbg))
    temp[...,0] = frames
    for i in np.arange(Nbg-1)+1:
        temp[...,i] = np.roll(frames,i,axis=0)
    
    if not return_bg:
        return frames - temp.min(axis=-1)
    else:
        return frames - temp.min(axis=-1),temp.min(axis=-1)
    
    
    
    
    