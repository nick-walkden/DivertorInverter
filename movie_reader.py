import cv2
import numpy as np
import struct
from copy import deepcopy as copy
from glob import glob
"""
This module contains various classes used to read movie files from photron cameras in the following formats:

- ipx_reader --> .ipx files
- mraw_reader --> mraw files (8,10,12 or 16 bit-depth)
- imstack_reader --> read a stack of images as a movie
"""



class movie_reader(object):
    """
    Base class for movie readers
    """

    def __init__(self,filename=None):       

        self.file_header = {}
        self._current_position =  0
        self._current_frame = 0 

        if filename is not None:
            self.open(filename)


    def open(self,file):
        #try:
        self._open(file)
        #except:
        #raise IOError("ERROR: Failed to open file : "+file)

    def read(self,transforms=[]):
        works,im_data,header = self._read()
        if works:
            if 'reverse_x' in transforms:
                im_data = im_data[::-1]
            if 'reverse_y' in transforms:
                im_data = im_data[...,::-1]
            if 'transpose' in transforms:
                im_data = im_data.T
        return works,im_data,header

    def set_frame_number(self,frame_number):
        self._set_frame_number(frame_number)
    
    def set_frame_time(self,frame_time):
        self._set_frame_time(frame_time)

    def reset(self):
        self._reset()

    def release(self):
        try:
            self._file.close()
        except:
            pass
    def get(self,field):
        try:
            return self.file_header[field]
        except: 
            print("Property "+field+" not found in header of file "+self._filename+"\n")
            return None

class imstack_reader(movie_reader):
    
    def __init__(self,file_list=None,directory=None,header_file=None):
        movie_reader.__init__(self,None)
        self._current_index = 0
        if file_list is not None:
            self.file_list = list(file_list)
            self.directory = "."
        elif directory is not None:
            self.file_list = get_sorted_image_files(directory)
            self._directory=directory
        cih_file = None             
        if header_file is not None:
            try:
                cih_file = open(header_file,'rb')
            except:
                raise IOError("ERROR: Cannot identify header file : "+header_file)
        elif directory is not None:
            headers = glob(directory+"/*.cih")
            if headers:
                #List is not empty  
                cih_file = open(headers[0],'rb')
        
        #Read and store the file header 
        if cih_file is not None:
            for line in cih_file:
                #print line
                if line[0] == '#':
                    pass
                else:
                    line_split = line.split(':')
                    if len(line_split)>1:
                        self.file_header[line_split[0].replace(' ','')] = line_split[1] 
                    else:
                        pass
            #Size of image frame in bits (no of pixels times size of unsigned int)  
            self._frame_size = int(self.file_header['ImageWidth'])*int(self.file_header['ImageHeight'])*2           
            self._pixels = int(self.file_header['ImageWidth'])*int(self.file_header['ImageHeight'])

    def _open(self,file):
        pass

    def _read(self):
        filename = self.file_list[self._current_index]
        im_file = open(filename,'rb')
        byte_data = bytearray(im_file.read())
        image_data = cv2.imdecode(np.asarray(byte_data,dtype=np.uint8),-1)
        image_data.reshape(int(self.file_header['ImageHeight']),int(self.file_header['ImageWidth']))
        self._current_index += 1
        file_split = self.file_list[self._current_index].split('.')[0]
        self._current_frame = int(file_split.split('S00')[-1][2:])
        #print(self._current_frame)
        try:
            time = -0.1 + (float(self.file_header['StartFrame'])/float(self.file_header['RecordRate(fps)'])) + float(self._current_frame)/float(self.file_header['RecordRate(fps)'])
        except:         
            time = float(self._current_frame)/float(self.file_header['RecordRate(fps)'])
        im_file.close() 
        #print(time)    
        return True,image_data,{'size':self._frame_size,'time_stamp':time}

    def _set_frame_number(self,frame_number):
        filesplit = self.file_list[0].split('.')    
        filesplit[0] = filesplit[0][:-6]+str(int(frame_number)).zfill(6)
        self._current_index = self.file_list.index('.'.join(filesplit))
    
    def _set_frame_time(self,time):
        #time = -0.1 + (float(self.file_header['StartFrame'])/float(self.file_header['RecordRate(fps)'])) + float(self._current_frame)/float(self.file_header['RecordRate(fps)'])
        frame_num = (time + 0.1 - (float(self.file_header['StartFrame']))/float(self.file_header['RecordRate(fps)']))*float(self.file_header['RecordRate(fps)'])
        
        print(frame_num)
        print(self.file_header['StartFrame'])
        print(self.file_header['RecordRate(fps)'])
        frame_num = int(frame_num)

        self.set_frame_number(frame_num)
    

class ipx_reader(movie_reader):
    """
    Class to read .ipx video files 
    
    Class Attributes:
    
        ipxReader.file_header               Dictionary containing information from the 
                                            file header
            
    Class Methods:
    
        ipxReader.open(filename)            Open the ipx file for reading and read in file header
        
        bool,np.ndarray,dict = ipxReader.read() 
                                Read the next movie frame, return the frame data as a
                                numpy ndarray, return the frame header as a dict
        
        ipxReader.release()             Release the loaded file
        
        ipxReader.get(property)             Return the value of the property if found in the file header

        ipxReader.reset()               Reset the reader to begin at the start of the file again

        ipxReader.set_frame_number(number)      Set the reader to read at the frame number given

        ipxReader.set_time(time)            Set the reader to read at the time given

    Instantiate using 

        myReader = ipxReader()              Returns a bare reader that is not linked to any file
        myReader = ipxReader(shot=xxxxx)        Returns a reader linked to the rbb0xxxxx.ipx file in the $MAST_IMAGES
                                directory
        mtReader = ipxReader(filename='myfile.ipx') Returns a reader linked to the file 'myfile.ipx'    
                    
                NOTE: the shot keyword takes precedence over the filename keyword so 

                    myReader = ipxReader(filename='myfile.ipx',shot=99999) 
            
                will be linked to shot 99999, rather than myfile.ipx
                
    """
    
    
    #Store .ipx formatting lists as class variables 
    __IPX_HEADER_FIELDS = ['ID','size','codec','date_time','shot',
                            'trigger','lens','filter','view','numFrames',
                            'camera','width','height','depth','orient',
                            'taps','color','hBin','left','right','vBin',
                            'top','bottom','offset_0','offset_1','gain_0',
                            'gain_1','preExp','exposure','strobe','board_temp',
                            'ccd_temp']
                            
    __IPX_HEADER_LENGTHS = [8,4,8,20,4,4,24,24,64,4,64,2,2,2,4,
                            2,2,2,2,2,2,2,2,2,2,4,4,4,4,4,4,4]
                            
    __IPX_HEADER_OFFSETS = [0]  
    for length in __IPX_HEADER_LENGTHS:
        __IPX_HEADER_OFFSETS.append(__IPX_HEADER_OFFSETS[-1]+length)
    __IPX_HEADER_OFFSETS.pop()
        
    # s = string                    
    # I = unsigned int
    # i = int
    # f = float
    # H = unsigned short                    
    __IPX_HEADER_TYPES = ['s','I','s','s','i','f','s','s','s',
                        'I','s','H','H','H','I','H','H','H',
                        'H','H','H','H','H','H','H','f','f',
                        'I','I','I','f','f']
                
    def __init__(self,filename=None,shot=None,camera='rbb'):
        
        movie_reader.__init__(self,filename=filename)        
        if shot:
            shot_string = str(shot)
            shot_string_list = list(shot_string)
            MAST_PATH = "/net/fuslsa/data/MAST_IMAGES/0"+shot_string_list[0]+shot_string_list[1]+"/"+shot_string+"/"+camera+"0"+shot_string+".ipx"
            self.open(MAST_PATH)

                
    def _open(self,filename):

        self._file = open(filename,'rb')
        self._filename = filename
            
        #Read and store the file header 
        for i in np.arange(len(self.__IPX_HEADER_FIELDS)):
            self._file.seek(self.__IPX_HEADER_OFFSETS[i])
            
            try:
                if self.__IPX_HEADER_TYPES[i] == 's':
                    format_string = '<'+str(self.__IPX_HEADER_LENGTHS[i])+self.__IPX_HEADER_TYPES[i]    
                else:
                    format_string = '<'+self.__IPX_HEADER_TYPES[i]  
                self.file_header[self.__IPX_HEADER_FIELDS[i]], = struct.unpack(format_string,self._file.read(self.__IPX_HEADER_LENGTHS[i]))
            except:
                print("WARNING: Unable to read Header field "+self.__IPX_HEADER_FIELDS[i])
                self.file_header[self.__IPX_HEADER_FIELDS[i]] = None
                    
        self._current_position = self.file_header['size']
    
    def _read_frame_header(self):
        try:
            if self._file is None:
                print("WARNING: No file opened, returning")
                return
            
            self._file.seek(self._current_position)
            
            #Read the frame header
            header = {}
            header['size'], = struct.unpack('<I',self._file.read(4))
            self._current_position += 4
            
            self._file.seek(self._current_position)
            header['time_stamp'], = struct.unpack('<d',self._file.read(8))
            self._current_position += 8
            
            return True,header
            
        except:
            #Ensure that the reader closes gracefully
            print("\nWARNING: End of file detected. Closeing.")         
            self.release()
            
            return False,None
        
    def _read(self):        
        try:
            if self._file is None:
                print("WARNING: No file opened, returning")
                return
            
            #Read the frame header
            ret,header = self._read_frame_header()
            header['frame_number'] = self._current_frame
            #Now read in the frame data as bytes
            self._file.seek(self._current_position)
            byte_data = np.fromfile(self._file,count=header['size'],dtype=np.uint8)
            self._current_position += header['size'] - 12
            
            #Now decode the jpg2 data using opencv, keeping output colorstyle as input colorstyle
            image_data = cv2.imdecode(byte_data,cv2.IMREAD_UNCHANGED)
            image_data.reshape(self.file_header['height'],self.file_header['width'])
            
            self._current_frame += 1
            
            return True,image_data,header
            
        except:
            #Ensure that the reader closes gracefully
            print("\nWARNING: End of file detected. Closeing.")         
            self.release()
            
            return False,None,None

    def _skip_frame(self):
        ret,header = self._read_frame_header()
        self._current_position += header['size'] - 12
        return header

    def _reset(self):
        self._current_position = self.file_header['size']
        self._current_frame = 0 
    
    def _set_frame_number(self,frame_number):
        #always return to the top
        self.reset()
        while self._current_frame < frame_number:
            header = self._skip_frame()
            self._current_frame += 1

    def _set_frame_time(self,set_time):
        self.reset()
        time = 0.0
        while time < set_time:
            header = self._skip_frame()
            self._current_frame += 1
            time = header['time_stamp']


class mraw_reader(movie_reader):

    def __init__(self,filename=None,shot=None,trigger_time = -0.1):
        """
        NOTE: trigger_time is not read from header file, so specify at initialization
        """
        movie_reader.__init__(self,filename=filename)            
        self.trigger_time = trigger_time
        
        
    def _open(self,filename):

        filetitle = filename[:-(len(filename.split('.')[-1])+1)]
        
        #print filetitle
        #Contains the frames
        try:
            file_mraw = filetitle+".mraw"
            self._file = open(file_mraw,'rb')
        except:
            raise IOError("ERROR: Cannot identify mraw file : "+filetitle+".mraw")

        #Contains the header
        try:
            file_cih = filetitle+".cih"
            cih_file = open(file_cih,'rb')
        except:
            raise IOError("ERROR: Cannot identify header file : "+filetitle+".cih")

        self._filename = filename
        
        #Read and store the file header 
        for line in cih_file:
            #print line
            if line[0] == '#':
                pass
            else:
                line_split = line.split(':')
                if len(line_split)>1:
                    self.file_header[line_split[0].replace(' ','')] = line_split[1] 
                else:
                    pass
        #Size of image frame in bytes (no of pixels times size of unsigned int)  
        #NOTE: frame_size is only needed for 8 or 16 bit_depth movies.
        self._frame_size = int(self.file_header['ImageWidth'])*int(self.file_header['ImageHeight'])*int(self.file_header['ColorBit'])/8           
        self._pixels = int(self.file_header['ImageWidth'])*int(self.file_header['ImageHeight'])

    def _read(self):        
        #try:
        if self._file is None:
            print("WARNING: No file opened, returning")
            return
        bit_depth = int(self.file_header['ColorBit'])
        #Now read in the frame data as a byte array
        
        if bit_depth == 16: 
            self._file.seek(self._current_position)
            image_data = np.fromfile(self._file,count=self._pixels,dtype=np.uint16)
        elif bit_depth == 8: 
            self._file.seek(self._current_position)
            image_data = np.fromfile(self._file,count=self._pixels,dtype=np.uint8)
        elif bit_depth == 12: image_data = unpack_mraw_frame_12bit(self._file,self._pixels,start_frame = self._current_frame)
        elif bit_depth == 10: image_data = unpack_mraw_frame_10bit(self._file,self._pixels,start_frame = self._current_frame)
        else: raise InputError("Error: bit_depth "+str(bit_depth)+" is not supported by mrawReader.")
            
        image_data = np.reshape(image_data,(int(self.file_header['ImageHeight']),int(self.file_header['ImageWidth'])))
        self._current_frame += 1
        self._current_position += self._frame_size
        time = self.trigger_time + (float(self.file_header['StartFrame'])+ float(self._current_frame))/float(self.file_header['RecordRate(fps)'])
                 
        return True,image_data,{'size':self._frame_size,'time_stamp':time,'frame_number':float(self.file_header['StartFrame'])+ float(self._current_frame)}

    def _skip_frame(self):
        self._current_position += self._frame_size 
        self._current_frame += 1

    def _reset(self):
        self._current_position = 0
        self._current_frame = 0 
    
    def _set_frame_number(self,frame_number):
        #always return to the top
        self._current_position = self._frame_size*frame_number
        self._current_frame = frame_number

    def _set_frame_time(self,set_time):
        self.reset()
        time = 0.0
        while time < set_time:
            self._skip_frame()
            self._current_frame += 1
            time = self.trigger_time + (float(self.file_header['StartFrame'])+ float(self._current_frame))/float(self.file_header['RecordRate(fps)'])


def unpack_mraw_frame_10bit(file,n_pixels,start_frame=0):
    """
    Function to unpack an mraw image frame
    
    Input:
        file    file object     mraw file to read from
        n_pixels    int     number of pixels in image frame
        packed_bit_depth    int     bit depth of the image in the mraw file
        unpacked_bit_depth  int     bit depth to give image back in
        start_frame     int     movie frame to begin read from
    
    Return:
        1d list of pixel values in frame
        
    Example:
        file = open('my_mraw_file.mraw','rb')
        packed_bit_depth = 12   #Default for an mraw file
        output_bit_depth = 16   #Give image back as uint16
        n_pixels = 128*256      #Image is a 128*256 frame
        start_frame = 10
        
        frame = unpack_mraw_frame(file,n_pixels,packed_bit_depth,unpacked_bit_depth,start_frame)
    """
    
    start_byte = start_frame*n_pixels*10/8
    file.seek(start_byte)
    image = []
    
    n_bytes = n_pixels*10/8
    
    int_array = np.fromfile(file,count=n_bytes,dtype=np.uint8)
    
    bytes_1 = int_array[::5]
    bytes_2 = int_array[1::5]   
    bytes_3 = int_array[2::5]
    bytes_4 = int_array[3::5]   
    bytes_5 = int_array[4::5]

 
    # Here 4 pixels from the image are shared between 5 bytes of data like
    #
    #  |    byte 1      |      byte 2       |     byte 3        | byte 4            |      byte 5    |
    #  |o o o o o o o o | o o | o o o o o o | o o o o | o o o o | o o o o o o | o o | o o o o o o o o|
    #  |    Pixel 1           |       Pixel 2         |      Pixel 3          |        Pixel 4       |
    #
    # byte 2 is shared between pixel and we need only the right-most bits for pixel 2 and
    # only the left most bits for pixel 1. 
    
    # right-most bits of byte 2 = Most significant bits of Pixel 2
    # left-most bits of byte 2  = Least significant bits of Pixel 1
 
    pix_1 = np.array(4.0*bytes_1 + np.right_shift(bytes_2,6),dtype=np.uint16)
    pix_2 = np.array(16.0*np.bitwise_and(bytes_2,0b111111) + np.right_shift(bytes_3,4),dtype=np.uint16)
    pix_3 = np.array(64.0*np.bitwise_and(bytes_3,0b1111) + np.right_shift(bytes_4,2),dtype=np.uint16)
    pix_4 = np.array(256.0*np.bitwise_and(bytes_4,0b11) + bytes_5,dtype=np.uint16)
    #try:
    image = (np.dstack([pix_1,pix_2,pix_3,pix_4])).reshape((1,n_pixels))[0]
    #except:
    #    image = np.zeros(n_pixels)
    return image

def unpack_mraw_frame_12bit(file,n_pixels,start_frame=0):
    """
    Function to unpack an mraw image frame
    
    Input:
        file    file object     mraw file to read from
        n_pixels    int     number of pixels in image frame
        packed_bit_depth    int     bit depth of the image in the mraw file
        unpacked_bit_depth  int     bit depth to give image back in
        start_frame     int     movie frame to begin read from
    
    Return:
        1d list of pixel values in frame
        
    Example:
        file = open('my_mraw_file.mraw','rb')
        packed_bit_depth = 12   #Default for an mraw file
        output_bit_depth = 16   #Give image back as uint16
        n_pixels = 128*256      #Image is a 128*256 frame
        start_frame = 10
        
        frame = unpack_mraw_frame(file,n_pixels,packed_bit_depth,unpacked_bit_depth,start_frame)
    """
    
    start_byte = start_frame*n_pixels*12/8
    file.seek(start_byte)
    image = []
    
    n_bytes = n_pixels*12/8
    
    int_array = np.fromfile(file,count=n_bytes,dtype=np.uint8)
    
    bytes_1 = int_array[::3]
    bytes_2 = int_array[1::3]   
    bytes_3 = int_array[2::3]

 
    # Here 2 pixels from the image are shared between three bytes of data like
    #
    #  |    byte 1     |      byte 2     |     byte 3    |
    #  |o o o o o o o o|o o o o | o o o o|o o o o o o o o|
    #  |         Pixel 1        |          Pixel 2       |
    #
    # byte 2 is shared between pixel and we need only the right-most bits for pixel 2 and
    # only the left most bits for pixel 1. 
    
    # right-most bits of byte 2 = Most significant bits of Pixel 2
    # left-most bits of byte 2  = Least significant bits of Pixel 1
 
    pix_1 = np.array(16.0*bytes_1 + np.right_shift(bytes_2,4),dtype=np.uint16)
    pix_2 = np.array(256.0*np.bitwise_and(bytes_2,0b1111) + bytes_3,dtype=np.uint16)
    
    try:
        image = (np.dstack([pix_1,pix_2])).reshape((1,n_pixels))[0]
    except:
        image = np.zeros(n_pixels)
    return image
        
def get_sorted_image_files(directory,priority='png'):
    
    """Return a sorted list of image files in a given directory"""
    
    #First get a list of file, start with priority, then all other image file types
    im_types = ['png','jpg','bmp','tif']
    im_types.remove(priority)
    
    file_list = glob(directory+'/*.'+priority)
    if not file_list:
        for im_type in im_types:
            file_list = glob(directory+'/*.'+im_type)
            if file_list:
                break

    #Currently assume standard mraw output filename
    sorted_list = sorted(file_list,key=lambda file_name: int(file_name.split('.')[0].split('S00')[-1][3:]))
    #print(file_list)
    #print(sorted_list)

    return sorted_list

    

