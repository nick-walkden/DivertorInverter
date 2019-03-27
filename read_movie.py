#!/usr/bin/env python
from __future__ import print_function
import cv2
import numpy as np
from frame_history import frame_history
from movie_reader import ipx_reader,mraw_reader,imstack_reader
import logging

def read_movie(filename,Nframes=None,stride=1,startpos=0,endpos=-1,verbose=True,startframe=None,endframe=None,starttime=None,endtime=100.0,
              transforms=[],trigger_time = -0.1):

    """
    Function to read in a movie file using openCV and store as a frameHistory

    Arguments:
        filename	-> 	name of the movie file to read from
                    OR
                    MAST shot number to read

    keywords:
        Nframes		-> 	Number of frames to read	Default: None, read entire movie
        stride		->	Read frames with a stride	Default: 1, read every frame
        startpos	-> 	Relative position to start	Default: 0, start from beginning of movie
                    reading from, 0 = start,
                    1 = end, 0.xx is xx% through
                    the movie
        endpos		->	Relative position to end 	Default: 1, read until end of movie
                    reading
        transforms	->	Apply any or all of the following transforms to the image data
                        'reverse_x' : reverse the x dimension
                        'reverse_y' : reverse the y dimension
                        'transpose' : transpose the image (flip x and y)

    Example:

        frames = readMove('myMovie.mov',Nframes=100,stride=2,startpos=0.3,endpos=0.5)

        This will read every 2nd frame of a movie from the file 'myMovie.mov' starting from 30% into the movie and
        ending after 100 frames, or when it reaches 50% through, whichever comes first

    """
    frames = frame_history(descriptor="File: "+filename)
    if '.' not in filename or filename.split('.')[-1] == 'ipx' or filename.split('.')[-1] == 'mraw':
        if filename.split('.')[-1] == 'ipx':
            print('here')
            vid = ipx_reader(filename=filename)
            
            if startpos is not None and startframe is None:
                startframe = int(startpos*vid.file_header['numFrames'])
            if endpos is not None and endframe is None:
                endframe = int(np.abs(endpos)*vid.file_header['numFrames'])
            elif endframe is -1:
 
                endframe = int(1.0*vid.file_header['numFrames'])
        else:
            if filename.split('.')[-1] == 'mraw':
                vid = mraw_reader(filename=filename)
                
            else:
                vid = imstack_reader(directory=filename)
            if startpos is not None and startframe is None:
                startframe = int(startpos*int(vid.file_header['TotalFrame']))
            if endpos is not None and endframe is None:
                print(vid.file_header['TotalFrame'])
                if endpos > -1:
                    endframe = int(endpos*int(vid.file_header['TotalFrame']))
                else:
                    endframe = int(1.0*int(vid.file_header['TotalFrame']))

        if Nframes is None:
            Nframes = endframe - startframe + 1
        
        vid.set_frame_number(startframe)
        if starttime is not None:
            vid.set_frame_time(starttime)
        N = 0
        for i in np.arange(Nframes*stride):
            ret,frame,header = vid.read(transforms=transforms)
            print(frame)
            if ret and (not N + startframe > endframe)  and (not float(header['time_stamp']) > endtime):
                if i % stride == 0:
                    if verbose:
                        print("Reading movie frame {} at time {}".format(header['frame_number'],header['time_stamp']))
                    
                    frames.add_frame(frame,header['time_stamp'],header['frame_number'])
                    N += 1
            else:
                break
        pass
    else:
        #Reading a non-ipx file with openCV

        vid = cv2.VideoCapture(filename)
        #Set the starting point of the video
        vid.set(2,startpos)

        times = []
        frameNos = []

        for i in np.arange(Nframes*stride):
            ret,frame = vid.read()
            if ret and not vid.get(2)>endpos:
                #Succesful read
                if i%stride==0:
                    #Only take every stride frames
                    frames.add_frame(frame,vid.get(0),vid.get(1))
                    if verbose:
                        print("Reading movie frame "+str(i), end='\r')
            else:
                break
    vid.release()
    return frames


