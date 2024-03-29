"""
To write points without errors make sure the original file has at least one spot object! 
You can create a fake point in Imaris, then save the file. The point will be overwritten.
There also seems to be some limitations to file size...

"""
import h5py
import numpy

import ligate.IO as io


def openFile(filename, mode = "a"):
    """Open Imaris file as hdf5 object
        
    Arguments:
        filename (str): file name
        mode (str): argument to h5py.File
    
    Returns:
        object: h5py object
    """        
    return h5py.File(filename, mode)

    
def closeFile(h5file):
    """Close Imaris hdf5 file object
    
    Arguments:
        h5file (object): h5py opject
    
    Returns:
        bool: success
    """ 
    
    return h5file.close()


def readDataSet(h5file, resolution = 0, channel = 0, timepoint = 0):
    """Open Imaris file and returns hdf5 image data
    
    Arguments:
        h5file (object): h5py object
        resolution (int): resolution level
        channel (int): color channel
        timepoint (int): time point
    
    Returns:
        array: image data
    """
    
    dsname = "/DataSet/ResolutionLevel " + str(resolution) + "/TimePoint " + str(timepoint) + "/Channel " + str(channel) + "/Data"
    return h5file.get(dsname)


def dataSize(filename, resolution = 0, channel = 0, timepoint = 0, **args):
    """Read data size of the imaris image data set
    
    Arguments:
        filename (str):  imaris file name
        resolution (int): resolution level
        channel (int): color channel
        timepoint (int): time point
    
    Returns:
        tuple: image data size
    """

    f = openFile(filename)
    
    ds = readDataSet(f, resolution = resolution, channel = channel, timepoint = timepoint)
    dims = list(ds.shape)
    dims = (dims[2], dims[1], dims[0])
    
    return io.dataSizeFromDataRange(dims, **args)

 
def dataZSize(filename, **args):
    """Read z data size of the imaris image data set
    
    Arguments:
        filename (str):  imaris file name
        resolution (int): resolution level
        channel (int): color channel
        timepoint (int): time point
    
    Returns:
        int: image z data size
    """  
    
    dsize = dataSize(filename, **args)
    return dsize[2]
    


def readData(filename, x = all, y = all, z = all, resolution = 0, channel = 0, timepoint = 0, **args):
    """Read data from imaris file

    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
        resolution (int): resolution level
        channel (int): color channel
        timepoint (int): time point
    
    Returns:
        array: image data
    """ 
    
    f = h5py.File(filename, "r")
    dataset = readDataSet(f, resolution = resolution, channel = channel, timepoint = timepoint)
    dsize = dataset.shape
    
    rz = io.toDataRange(dsize[0], r = z)
    ry = io.toDataRange(dsize[1], r = y)
    rx = io.toDataRange(dsize[2], r = x)
    
    data = dataset[rz[0]:rz[1],ry[0]:ry[1],rx[0]:rx[1]]
    data = data.transpose((2,1,0)); # imaris stores files in reverse x,y,z ordering
    #data = dataset[x[0]:x[1],y[0]:y[1],z[0]:z[1]]
    
    f.close()
    
    return data

   
def getDataSize(h5file):
    """Get the full data size in pixel from h5py imaris object
    
    Arguments:
        h5file (object): h5py object
    
    Returns:
        tuple: image data size
    """ 
    
    pn = '/DataSetInfo/Image'
    ia = h5file.get(pn).attrs
    
    x = int(''.join(ia["X"]))
    y = int(''.join(ia["Y"]))
    z = int(''.join(ia["Z"]))
    
    return numpy.array([x,y,z])
    


def getDataExtent(h5file):
    """Get the spatial extent of data from h5py imaris object
    
    Arguments:
        h5file (object): h5py object
    
    Returns:
        array: spatial extend of image
    """ 

    pn = '/DataSetInfo/Image'
    ia = h5file.get(pn).attrs
    
    x1 = float(''.join(ia["ExtMin0"]))
    y1 = float(''.join(ia["ExtMin1"]))
    z1 = float(''.join(ia["ExtMin2"]))
   
    x2 = float(''.join(ia["ExtMax0"]))
    y2 = float(''.join(ia["ExtMax1"]))
    z2 = float(''.join(ia["ExtMax2"]))
    
    return numpy.array([[x1,y1,z1], [x2,y2,z2]])



def getScaleAndOffset(h5file):
    """Determine scale and offset to transform pixel to spatial coordinates as used by imaris
    
    Arguments:
        h5file (object): h5py object
    
    Returns:
        tuple: image scale (length / pixel) and offset (from origin)
    """     
    
    ds = getDataSize(h5file)
    de = getDataExtent(h5file)
    
    sc = de[1,:] - de[0,:]
    sc = sc / ds
    
    return (sc, de[0,:])


def transformPointsToImaris(points, scale = (4.0625, 4.0625, 3), offset = (0,0,0)):
    """Transform pixel coordinates of cell centers to work in Imaris
    
    Arguments:
        points (array): point coordinate array
        scale (tuple): spatial scale of the image data
        offset (tuple): spatial offset of the image data
    
    Returns:
        array: scaled points
    """
    
    if len(scale) == 1:
        scale = (scale, scale, scale)
    
    for i in range(3):
        points[:,i] = points[:,i] * scale[i] + offset[i]
        
    return points


def writePoints(filename, points, mode = "o", radius = 0.5, scale = all, offset = None):
    """Write points to Imaris file 
    
    Arguments:
        filename (str): imaris file name 
        points (array): point data
        mode (str): 'o'= override, 'a'=add
        radius (float): size of each point
        scale (tuple or all): spatial scaling of points
        offset (tuple or None): spatial offset of points
    
    Returns:
        str: file name of imaris file
        
    Note:
        This routine is still experimental !
    """
    
    if isinstance(filename, str):
        h5file = openFile(filename)
    else:
        h5file = filename
      
    #delete Scene8 info so do not need to write it
    s8 = "/Scene8"
    if h5file.get(s8) != None:
        del h5file[s8]
    
    #get number of point sets
    ct = "Scene/Content"
    ca = h5file.get(ct).attrs
    np = ca.get("NumberOfPoints")
    if np == None:
        ca.create("NumberOfPoints", 1, dtype = "uint64")
        np = 0
        mode = "a"
    
    #write points
    if mode == "a":  # add points -> need to test further
        
        #update number of points(4.0625, 4.0625, 3)
        np = np + 1
        ca["NumberOfPoints"] = np
        
        # Create Pointset and attributes
        pn = ct + "/Points" + str(int(np-1))
        
        histlog = numpy.array([''], dtype='|S1')
        name = numpy.array(['Spots ' + str(int(np))], dtype='|S' + str(6 + len(str(int(np)))))
        material = numpy.array([ '<bpMaterial mDiffusion="0.8 0.8 0.8" mSpecular="0.2 0.2 0.2" mEmission="0 0 0" mAmbient="0 0 0" mTransparency="0" mShinyness="0.1"/>\n'], dtype='|S133')
        iid = 200000 + np # int64
        unit = numpy.array(['um'], dtype='|S2')
        descrp = numpy.array([''], dtype='|S1')
                
        pg = h5file.create_group(pn)
        pa = pg.attrs
        pa.create("HistoryLog", histlog)
        pa.create("Name", name)
        pa.create("Material", material)
        pa.create("Id", iid, dtype = 'uint64')
        pa.create("Unit", unit)
        pa.create("Description", descrp)
        
        #add TimeInfos
        
        tb = h5file.get('DataSetTimes/TimeBegin')
        tb = tb[0][1]
        
        #del h5file[pn + "/TimeInfos"]
        h5file.create_dataset(pn + "/TimeInfos", data = numpy.array([tb[0:23]], dtype='|S23'))
        
    else:
        pn = ct + "/Points" + str(int(np-1))
        
    
    #  make points
    npts = points.shape[0]
    
    if points.shape[1] != 3:
        raise StandardError("Points shape is not (n,3)!")
    
    #points = points[:,[0,1,2]]; # todo: check exchange of coordinates
    
    #scale points frfom pixel to     
    if scale is all: # automatically determine scaling for points    
        (scale, offset) = getScaleAndOffset(h5file)
    
    if offset is None:
        offset = numpy.array([0,0,0])
     
    pointsS = points.copy()
    pointsS = transformPointsToImaris(pointsS, scale = scale, offset = offset)
    
    pts = numpy.c_[pointsS, radius * numpy.ones(npts)]
    ts =  numpy.zeros(npts)
    
    # write points
    pnt = pn + '/Time'
    pnc = pn + '/CoordsXYZR'
    
    if h5file.get(pnt) != None:
        del h5file[pnt]
    h5file.create_dataset(pnt, shape=(npts,1), dtype='i64', data=ts)
    
    
    if h5file.get(pnc) != None:
        del h5file[pnc]
    h5file.create_dataset(pnc, shape=pts.shape, dtype='f32', data=pts)
    
    if isinstance(filename, str):
        h5file.close()
    
    return filename
    


def writeData(filename, **args):
    """Write image data to imaris file
    
    Note:
        Not implemented yet !
    """
    raise RuntimeError("Writing data to imaris file not implemented yet""")


def readPoints(filename):
    """Read points from imaris file
    
    Note:
        Not implemented yet !
    """
    raise RuntimeError("Reading points form imaris file not implemented yet""")


def copyData(source, sink):
    """Copy a imaris file from source to sink
    
    Arguments:
        source (str): file name pattern of source
        sink (str): file name pattern of sink
    
    Returns:
        str: file name patttern of the copy
    """ 
    io.copyFile(source, sink)