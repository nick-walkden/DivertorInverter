#!/usr/bin/env python
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

"""
FrameHistory object for working with sets of Frames
"""

import os
import sys
import numpy as np
from copy import copy, deepcopy



class frame_history(object):
    """
    Simple class to store a movie as a history of frames

    Frames within a frame history can be accessed by indexing, ie

    frames = frameHistory()

    frame = frames[1,:,:,:]
    """
    
    def __init__(self, frames=None,timestamps=None,frame_numbers=None, descriptor=None, loop_frames=0, verbose=True):
        
        
        if frames is not None and len(frames.shape) < 3:
            frames = frames[np.newaxis,:,:]
            if timestamps is not None:
                timestamps = np.asarray(timestamps)
            if frame_numbers is not None:
                frame_numbers = np.asarray(frame_numbers)
                
        
        self.frames = frames 
        self.timestamps = timestamps
        self.frame_numbers = frame_numbers
        self.descriptor_string = descriptor
        self.loop_frames = loop_frames
        self.verbose = verbose
             
        if frames is not None:
            self.nt = frames.shape[0]
            self.nx = frames.shape[1]
            self.ny = frames.shape[2]
        else:
            self.nt,self.nx,self.ny = 0,0,0 
        
        #Fill times and frame numbers with nan if not values given
        if timestamps is None and frames is not None:
            self.timestamps = np.nan*np.ones(self.nt)
        
        if frame_numbers is None and frames is not None:
            self.frame_numbers = np.nan*np.ones(self.nt)
            
    def clear(self):

        self.frames = None
        self.timestamps=None
        self.frame_numbers=None
        self.descriptor_string=None
        self.loop_frames=0
        self.verbose=True
        self.nt,self.nx,self.ny = 0,0,0
    
    def set_frames(self, frames, timestamps=None,frame_numbers=None):
        """
        Set the frame history to an existing frame history instance or build from a numpy array of frames
        """

        if type(frames) == type(self):
            for key, value in frames.__dict__.iteritems():
                self.__dict__[key] = value
        else:
            if isinstance(frames, np.ndarray):
                # Passed an array containing stack of frames                
                self.frames = frames
                self.timestamps = timestamps
                self.frame_numbers=frame_numbers
                
                
                self.nt = frames.shape[0]
                self.nx = frames.shape[1]
                self.ny = frames.shape[2]
                
                if timestamps is None:
                    self.timestamps = np.nan*np.ones(self.nt)
        
                if frame_numbers is None:
                    self.frame_numbers = np.nan*np.ones(self.nt)
                

    def add_frame(self, frame, timestamp=np.nan,frame_number=np.nan,allow_duplicate=True):
        """ Add existing Frame object
        """
        print(frame)
        if self.frames is None:
            if self.verbose:
                print("  [frame_history.add_frame] No frames detected in history, initializing with current frame")
            self.frames = frame[np.newaxis,:,:]
            self.timestamps =np.array([timestamp])
            self.frame_numbers = np.array([frame_number])
            self.nt = 1
            return

        #check if the frame is a du
        if not allow_duplicate:
            if np.any(np.equal(self.frames,frame).all((1,2))):
                if self.verbose:
                    print("  [frame_history.add_frame] Frame detected as duplicate, returning.")
                return

        self.nt += 1
        
        
        
        #If we are looping, then operate a last-in first-out queue
        if self.loop_frames > 0 and self.nt > self.loop_frames:
            
            self.frames[:-1] = self.frames[1:]
            self.frames[-1] = frame
            
            self.timestamps[:-1] = self.timestamps[1:]
            self.timestamps[-1] = timestamp
            
            self.frame_numbers[:-1] = self.frame_numbers[1:]
            self.frame_numbers[-1] = frame_number 
             
            
        else:
            self.frames = np.append(self.frames,frame[np.newaxis,:,:],axis=0)
            self.timestamps = np.append(self.timestamps,timestamp)
            self.frame_numbers = np.append(self.frame_numbers,frame_number)
            
    def sort_frames_by_time(self):
        inds = np.argsort(self.timestamps)
        
        self.frames = self.frames[inds]
        self.timestamps = self.timestamps[inds]
        self.frame_numbers = self.frame_numbers[inds]
        
        
    def sort_frames_by_number(self):
        inds = np.argsort(self.frame_numbers)
        
        self.frames = self.frames[inds]
        self.timestamps = self.timestamps[inds]
        self.frame_numbers = self.frame_numbers[inds]
    
    def get_frames_in_range(self,timestamps=None,frame_numbers=None):
        
        inds = np.ones(nt,dtype=bool)
        
        if timestamps is not None:
            inds = inds & self.timestamps >= timestamps[0] & self.timestamps <= timestamps[1]
        
        if frame_numbers is not None:
            inds = inds & self.frame_numbers >= frame_numbers[0] & self.frame_numbers <= frame_numbers[1]
            
        return frame_history(self.frames[inds],self.timestamps[inds],self.frame_numbers[inds],self.descriptor_string,self.loop_frames,self.verbose)   
        
    def get_frames_at_times(self,timestamps,return_nearest=True):
        
        if return_nearest:
            return self._get_frames_at_nearest(timestamps,self.timestamps)
        else:
            return self._get_frames_at_values(timestamps,self.timestamps)
    
    def get_frames_at_numbers(self,frame_numbers,return_nearest=True):
        if return_nearest:
            return self._get_frames_at_nearest(frame_numbers,self.frame_numbers)
        else:                                 
            return self._get_frames_at_values (frame_numbers,self.frame_numbers)
    
    def _get_frames_at_values(self,values,frame_values):
        
        inds = np.where(np.isin(frame_values,values))[0]
        
        return frame_history(self.frames[inds],self.timestamps[inds],self.frame_numbers[inds],self.descriptor_string,self.loop_frames,self.verbose)
    
    def _get_frames_at_nearest(self,values,frame_values):
        
        inds = np.min(np.abs(frame_values[:,np.newaxis] - values[np.newaxis,:]),axis=0)
                
        return self._get_frames_at_values(frame_values[inds],values)

    def __iter__(self):
        """
        Iterate frames using a call like

        for frame in frameHistory:

        """
        for N in np.arange(self.nt):
            yield self.frames[N]

    def __getitem__(self, index, mask=False):
        """
        Access frames in the frameHistory using
        frameHistory()[i,j,k,...]
        """
        return self.frames[index]

    def __setitem__(self, index, setvalue):
        """
        Set individual frames using index
        """
        self.frames[index] = setvalue

    def __str__(self):
        
        return self.descriptor_string

    
    def animate(self,loop=True):
        pass
    
    