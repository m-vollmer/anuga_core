"""Specialised functionality to read centroid data from sww files. 
This code was pulled from plot_utils as they are more general.

We could put it in sww.py, but keep it separate for the time being 
because importing this module at the top level interferes with spawning 
of parallel processes in the parallel tests. 
"""
        
import copy
import numpy
import numpy as num
from anuga.file.netcdf import NetCDFFile

class get_output(object):
    """Read in data from an .sww file in a convenient form
       e.g. 
        p = plot_utils.get_output('channel3.sww', minimum_allowed_height=0.01)
        
       p then contains most relevant information as e.g., p.stage, p.elev, p.xmom, etc 
    """
    def __init__(self, filename, minimum_allowed_height=1.0e-03, timeSlices='all', verbose=False):
                # FIXME: verbose is not used
        self.x, self.y, self.time, self.vols, self.stage, \
                self.height, self.elev, self.friction, self.xmom, self.ymom, \
                self.xvel, self.yvel, self.vel, self.minimum_allowed_height,\
                self.xllcorner, self.yllcorner, self.timeSlices, self.starttime = \
                _read_output(filename, minimum_allowed_height, copy.copy(timeSlices))
        self.filename = filename
        self.verbose = verbose

############################################################################

def _read_output(filename, minimum_allowed_height, timeSlices):
    """
     Purpose: To read the sww file, and output a number of variables as arrays that 
              we can then e.g. plot, interrogate 

              See get_output for the typical interface, and get_centroids for
                working with centroids directly
    
     Input: filename -- The name of an .sww file to read data from,
                        e.g. read_sww('channel3.sww')
            minimum_allowed_height -- zero velocity when height < this
            timeSlices -- List of time indices to read (e.g. [100] or [0, 10, 21]), or 'all' or 'last' or 'max'
                          If 'max', the time-max of each variable will be computed. For xmom/ymom/xvel/yvel, the
                           one with maximum magnitude is reported
    
    
     Output: x, y, time, stage, height, elev, xmom, ymom, xvel, yvel, vel
             x,y are only stored at one time
             elevation may be stored at one or multiple times
             everything else is stored every time step for vertices
    """

    # Open ncdf connection
    fid = NetCDFFile(filename)
    
    time = fid.variables['time'][:]

    # Treat specification of timeSlices
    if (timeSlices == 'all'):
        inds = list(range(len(time)))
    elif (timeSlices == 'last'):
        inds = [len(time)-1]
    elif (timeSlices == 'max'):
        inds = 'max' #
    else:
        try:
            inds = list(timeSlices)
        except:
            inds = [timeSlices]
    
    if (inds != 'max'):
        time = time[inds]
    else:
        # We can't really assign a time to 'max', but I guess max(time) is
        # technically the right thing -- if not misleading
        time = time.max()

    
    # Get lower-left
    xllcorner = fid.xllcorner
    yllcorner = fid.yllcorner
    starttime = fid.starttime

    # Read variables
    x = fid.variables['x'][:]
    y = fid.variables['y'][:]

    stage = getInds(fid.variables['stage'], timeSlices=inds)
    elev = getInds(fid.variables['elevation'], timeSlices=inds)

    # Simple approach for volumes
    vols = fid.variables['volumes'][:]

    # Friction if it exists
    if('friction' in fid.variables):
        friction=getInds(fid.variables['friction'], timeSlices=inds) 
    else:
        # Set friction to nan if it is not stored
        friction = elev * 0. + numpy.nan

    # Trick to treat the case where inds == 'max'
    inds2 = copy.copy(inds)
    if inds == 'max':
        inds2 = list(range(len(fid.variables['time'])))
    
    # Get height
    if ('height' in fid.variables):
        height = fid.variables['height'][inds2]
    else:
        # Back calculate height if it is not stored
        #height = fid.variables['stage'][inds2]+0.
        height = numpy.zeros((len(inds2), stage.shape[1]), dtype='float32')
        for i in range(len(inds2)):
            height[i,:] = fid.variables['stage'][inds2[i]]

        if(len(elev.shape)==2):
            height = height-elev
        else:
            for i in range(height.shape[0]):
                height[i,:] = height[i,:]-elev
    height = height*(height>0.)

    # Get xmom
    #xmom = fid.variables['xmomentum'][inds2]
    #ymom = fid.variables['ymomentum'][inds2]
    xmom = numpy.zeros((len(inds2), stage.shape[1]), dtype='float32')
    ymom = numpy.zeros((len(inds2), stage.shape[1]), dtype='float32')
    for i in range(len(inds2)):
        xmom[i,:] = fid.variables['xmomentum'][inds2[i]]
        ymom[i,:] = fid.variables['ymomentum'][inds2[i]]
    
    # Get vel
    h_inv = 1.0/(height+1.0e-12)
    hWet = (height > minimum_allowed_height)
    xvel = xmom * h_inv * hWet
    yvel = ymom * h_inv * hWet
    vel = (xmom**2 + ymom**2)**0.5*h_inv*hWet

    if inds == 'max':
        height = height.max(axis=0, keepdims=True)
        vel = vel.max(axis=0, keepdims=True)
        xvel = getInds(xvel, timeSlices=inds, absMax=True)
        yvel = getInds(yvel, timeSlices=inds, absMax=True)
        xmom = getInds(xmom, timeSlices=inds, absMax=True)
        ymom = getInds(ymom, timeSlices=inds, absMax=True)

    fid.close()

    return x, y, time, vols, stage, height, elev, friction, xmom, ymom,\
           xvel, yvel, vel, minimum_allowed_height, xllcorner,yllcorner, inds, starttime

######################################################################################

class get_centroids(object):
    """
    Extract centroid values from the output of get_output, OR from a
        filename  
    See _read_output or _get_centroid_values for further explanation of
        arguments
    e.g.
        # Case 1 -- get vertex values first, then centroids
        p = plot_utils.get_output('my_sww.sww', minimum_allowed_height=0.01) 
        pc=util.get_centroids(p, velocity_extrapolation=True) 

        # Case 2 -- get centroids directly
        pc=plot_utils.get_centroids('my_sww.sww', velocity_extrapolation=True) 

    NOTE: elevation is only stored once in the output, even if it was
          stored every timestep.
          Lots of existing plotting code assumes elevation is a 1D
          array. 
          But as a hack for the time being the elevation from the file 
          is available via elev_orig
    """
    def __init__(self, p, velocity_extrapolation=False, verbose=False,
                 timeSlices=None, minimum_allowed_height=1.0e-03):
        
        self.time, self.x, self.y, self.stage, self.xmom,\
            self.ymom, self.height, self.elev, self.elev_orig, self.friction, self.xvel,\
            self.yvel, self.vel, self.xllcorner, self.yllcorner, self.timeSlices= \
                _get_centroid_values(p, velocity_extrapolation,\
                                     timeSlices=copy.copy(timeSlices),\
                                     minimum_allowed_height=minimum_allowed_height,\
                                     verbose=verbose)

def _get_centroid_values(p, velocity_extrapolation, verbose, timeSlices, 
                         minimum_allowed_height):
    """
    Function to get centroid information -- main interface is through 
        get_centroids. 
        See get_centroids for usage examples, and read_output or get_output for further relevant info
     Input: 
           p --  EITHER:
                  The result of e.g. p=util.get_output('mysww.sww'). 
                  See the get_output class defined above. 
                 OR:
                  Alternatively, the name of an sww file
    
           velocity_extrapolation -- If true, and centroid values are not
            in the file, then compute centroid velocities from vertex velocities, and
            centroid momenta from centroid velocities. If false, and centroid values
            are not in the file, then compute centroid momenta from vertex momenta,
            and centroid velocities from centroid momenta
    
           timeSlices = list of integer indices when we want output for, or
                        'all' or 'last' or 'max'. See _read_output
    
           minimum_allowed_height = height at which velocities are zeroed. See _read_output
    
     Output: Values of x, y, Stage, xmom, ymom, elev, xvel, yvel, vel etc at centroids
    """

    # Figure out if p is a string (filename) or the output of get_output
    pIsFile = isinstance(p, str)
    if pIsFile: 
        fid = NetCDFFile(p) 
    else:
        fid = NetCDFFile(p.filename)

    # UPDATE: 15/06/2014 -- below, we now get all variables directly from the file
    #         This is more flexible, and allows to get 'max' as well
    #         However, potentially it could have performance penalities vs the old approach (?)

    # Make 3 arrays, each containing one index of a vertex of every triangle.
    vols = fid.variables['volumes'][:]
    vols0 = vols[:,0]
    vols1 = vols[:,1]
    vols2 = vols[:,2]
    
    # Get lower-left offset
    xllcorner = fid.xllcorner
    yllcorner = fid.yllcorner
   
    #@ Get timeSlices 
    # It will be either a list of integers, or 'max'
    l = len(vols)
    time = fid.variables['time'][:]
    nts = len(time) # number of time slices in the file 
    if timeSlices is None:
        if pIsFile:
            # Assume all timeSlices
            timeSlices=list(range(nts))
        else:
            timeSlices=copy.copy(p.timeSlices)
    else:
        # Treat word-based special cases
        if timeSlices == 'all':
            timeSlices=list(range(nts))
        if timeSlices == 'last':
            timeSlices=[nts-1]

    #@ Get minimum_allowed_height
    if minimum_allowed_height is None:
        if pIsFile:
            minimum_allowed_height=0.
        else:
            minimum_allowed_height=copy.copy(p.minimum_allowed_height)

    # Treat specification of timeSlices
    if timeSlices == 'all':
        inds = list(range(len(time)))
    elif timeSlices=='last':
        inds = [len(time)-1]
    elif timeSlices=='max':
        inds = 'max' #
    else:
        try:
            inds = list(timeSlices)
        except:
            inds = [timeSlices]
    
    if inds != 'max':
        time = time[inds]
    else:
        # We can't really assign a time to 'max', but I guess max(time) is
        # technically the right thing -- if not misleading
        time = time.max()

    # Get coordinates
    x = fid.variables['x'][:]
    y = fid.variables['y'][:]
    x_cent = (x[vols0] + x[vols1] + x[vols2]) / 3.0
    y_cent = (y[vols0] + y[vols1] + y[vols2]) / 3.0

    # Stage and height and elevation
    stage_cent = _getCentVar(fid, 'stage_c', time_indices=inds, vols=vols)
    elev_cent = _getCentVar(fid, 'elevation_c', time_indices=inds, vols=vols)

    # Hack to allow refernece to time varying elevation
    elev_cent_orig = elev_cent
    
    if len(elev_cent.shape) == 2:
        # Coerce to 1D array, since lots of our code assumes it is
        elev_cent = elev_cent[0,:]

    # Friction might not be stored at all
    try:
        friction_cent = _getCentVar(fid, 'friction_c', time_indices=inds, vols=vols)
    except:
        friction_cent = elev_cent*0.+numpy.nan
    
    # Trick to treat the case where inds == 'max'
    inds2 = copy.copy(inds)
    if inds == 'max':
        inds2 = list(range(len(fid.variables['time'])))
   
    # height
    height_cent = stage_cent + 0.
    for i in range(stage_cent.shape[0]):
        height_cent[i,:] = stage_cent[i,:] - elev_cent

    if 'xmomentum_c' in fid.variables:
        # The following commented out lines seem to only work on
        # some numpy/netcdf versions. So we loop
        #xmom_cent = fid.variables['xmomentum_c'][inds2]
        #ymom_cent = fid.variables['ymomentum_c'][inds2]
        xmom_cent = numpy.zeros((len(inds2), fid.variables['xmomentum_c'].shape[1]), dtype='float32')
        ymom_cent = numpy.zeros((len(inds2), fid.variables['ymomentum_c'].shape[1]), dtype='float32')
        height_c_tmp = numpy.zeros((len(inds2), fid.variables['stage_c'].shape[1]), dtype='float32')
        for i in range(len(inds2)):
            xmom_cent[i,:] = fid.variables['xmomentum_c'][inds2[i]]
            ymom_cent[i,:] = fid.variables['ymomentum_c'][inds2[i]]
            if 'height_c' in fid.variables:
                height_c_tmp[i,:] = fid.variables['height_c'][inds2[i]]
            else:
                height_c_tmp[i,:] = fid.variables['stage_c'][inds2[i]] - elev_cent

        # Vel
        hInv = 1.0/(height_c_tmp + 1.0e-12)
        hWet = (height_c_tmp > minimum_allowed_height)
        xvel_cent = xmom_cent*hInv*hWet
        yvel_cent = ymom_cent*hInv*hWet

    else:
        # Get important vertex variables
        xmom_v = numpy.zeros((len(inds2), fid.variables['xmomentum'].shape[1]), dtype='float32')
        ymom_v = numpy.zeros((len(inds2), fid.variables['ymomentum'].shape[1]), dtype='float32')
        stage_v = numpy.zeros((len(inds2), fid.variables['stage'].shape[1]), dtype='float32')
        for i in range(len(inds2)):
            xmom_v[i,:] = fid.variables['xmomentum'][inds2[i]]
            ymom_v[i,:] = fid.variables['ymomentum'][inds2[i]]
            stage_v[i,:] = fid.variables['stage'][inds2[i]]

        elev_v = fid.variables['elevation']
        # Fix elevation + get height at vertices
        if (len(elev_v.shape)>1):
            elev_v = numpy.zeros(elev_v.shape, dtype='float32')
            for i in range(elev_v.shape[0]):
                elev_v[i,:] = fid.variables['elevation'][inds2[i]]
            height_v = stage_v - elev_v
        else:
            elev_v = elev_v[:]
            height_v = stage_v + 0.
            for i in range(stage_v.shape[0]):
                height_v[i,:] = stage_v[i,:] - elev_v

        # Height at centroids        
        height_c_tmp = (height_v[:, vols0] + height_v[:,vols1] + height_v[:,vols2])/3.0
       
        # Compute xmom/xvel/ymom/yvel
        if velocity_extrapolation:

            xvel_v = xmom_v * 0.
            yvel_v = ymom_v * 0.

            hInv = 1.0 / (height_v + 1.0e-12)
            hWet = (height_v > minimum_allowed_height)

            xvel_v = xmom_v * hInv * hWet
            yvel_v = ymom_v * hInv * hWet

            # Final xmom/ymom centroid values
            xvel_cent = (xvel_v[:, vols0] + xvel_v[:, vols1] + xvel_v[:, vols2])/3.0
            xmom_cent = xvel_cent*height_c_tmp
            yvel_cent = (yvel_v[:, vols0] + yvel_v[:, vols1] + yvel_v[:, vols2])/3.0
            ymom_cent = yvel_cent * height_c_tmp

        else:
            hInv = 1.0 / (height_c_tmp + 1.0e-12)
            hWet = (height_c_tmp > minimum_allowed_height)

            xmom_v = numpy.zeros((len(inds2), fid.variables['xmomentum'].shape[1]), dtype='float32')
            ymom_v = numpy.zeros((len(inds2), fid.variables['ymomentum'].shape[1]), dtype='float32')
            for i in range(len(inds2)):
                xmom_v[i,:] = fid.variables['xmomentum'][inds2[i]]
                ymom_v[i,:] = fid.variables['ymomentum'][inds2[i]]

            xmom_cent = (xmom_v[:, vols0] + xmom_v[:, vols1] + xmom_v[:, vols2])/3.0
            xvel_cent = xmom_cent * hInv * hWet
            ymom_cent = (ymom_v[:, vols0] + ymom_v[:, vols1] + ymom_v[:, vols2])/3.0
            yvel_cent = ymom_cent * hInv * hWet

    # Velocity
    vel_cent = (xvel_cent**2 + yvel_cent**2)**0.5

    if inds == 'max':
        vel_cent = vel_cent.max(axis=0, keepdims=True)
        #vel_cent = getInds(vel_cent, timeSlices=inds)
        xmom_cent = getInds(xmom_cent, timeSlices=inds, absMax=True)
        ymom_cent = getInds(ymom_cent, timeSlices=inds, absMax=True)
        xvel_cent = getInds(xvel_cent, timeSlices=inds, absMax=True)
        yvel_cent = getInds(yvel_cent, timeSlices=inds, absMax=True)

    fid.close()
    
    return time, x_cent, y_cent, stage_cent, xmom_cent,\
             ymom_cent, height_cent, elev_cent, elev_cent_orig, friction_cent,\
             xvel_cent, yvel_cent, vel_cent, xllcorner, yllcorner, inds

####################################################################
def getInds(varIn, timeSlices, absMax=False):
    """
     Convenience function to get the indices we want in an array.
     There are a number of special cases that make this worthwhile
     having in its own function
    
     INPUT: varIn -- numpy array, either 1D (variables in space) or 2D
            (variables in time+space)
            timeSlices -- times that we want the variable, see read_output or get_output
            absMax -- if TRUE and timeSlices is 'max', then get max-absolute-values
     OUTPUT:
           
    """
    #import pdb
    #pdb.set_trace()

    if (len(varIn.shape)==2):
        # There are multiple time-slices
        if timeSlices == 'max':
            # Extract the maxima over time, assuming there are multiple
            # time-slices, and ensure the var is still a 2D array
            if( not absMax):
                var = (varIn[:]).max(axis=0, keepdims=True)
            else:
                # For variables xmom,ymom,xvel,yvel we want the 'maximum-absolute-value'
                varInds = abs(varIn[:]).argmax(axis=0)
                varNew = varInds * 0.
                for i in range(len(varInds)):
                    varNew[i] = varIn[varInds[i], i]
                var = varNew
                var = var.reshape((1, len(var)))
        else:
            var = numpy.zeros((len(timeSlices), varIn.shape[1]), dtype='float32')
            for i in range(len(timeSlices)):
                var[i,:]=varIn[timeSlices[i]]
            var.reshape((len(timeSlices), varIn.shape[1]))
    else:
        # There is 1 time slice only
        var = varIn[:]
    
    return var


def _getCentVar(fid, varkey_c, time_indices, absMax=False, vols = None, space_indices=None):
    """
        Convenience function used to get centroid variables from netCDF
        file connection fid
    """

    if vols is not None:
        vols0 = vols[:, 0]
        vols1 = vols[:, 1]
        vols2 = vols[:, 2]

    if varkey_c in fid.variables == False:
        # It looks like centroid values are not stored
        # In this case, compute centroid values from vertex values
        assert (vols is not None), "Must specify vols since centroid quantity is not stored"

        newkey=varkey_c.replace('_c','')
        if time_indices != 'max':
            # Relatively efficient treatment is possible
            var_cent = fid.variables[newkey]
            if len(var_cent.shape) > 1:
                # array contain time slices
                var_cent = numpy.zeros((len(time_indices), fid.variables[newkey].shape[1]), dtype='float32')
                for i in range(len(time_indices)):
                    var_cent[i,:] = fid.variables[newkey][time_indices[i]]
                var_cent = (var_cent[:, vols0] + var_cent[:, vols1] + var_cent[:, vols2]) / 3.0
            else:
                var_cent = fid.variables[newkey][:]
                var_cent = (var_cent[vols0] + var_cent[vols1] + var_cent[vols2]) / 3.0
        else:
            # Requires reading all the data
            tmp = fid.variables[newkey][:]
            try: # array contain time slices
                tmp=(tmp[:,vols0]+tmp[:,vols1]+tmp[:,vols2])/3.0
            except:
                tmp=(tmp[vols0]+tmp[vols1]+tmp[vols2])/3.0
            var_cent=getInds(tmp, timeSlices=time_indices, absMax=absMax)
    else:
        if time_indices != 'max':
            if(len(fid.variables[varkey_c].shape)>1):
                var_cent = numpy.zeros((len(time_indices), fid.variables[varkey_c].shape[1]), dtype='float32')
                for i in range(len(time_indices)):
                    var_cent[i,:] = fid.variables[varkey_c][time_indices[i]]
            else:
                var_cent = fid.variables[varkey_c][:]
        else:
            var_cent=getInds(fid.variables[varkey_c][:], timeSlices=time_indices, absMax=absMax)

    if space_indices is not None:
        # Maybe only return particular space indices. Could do this more
        # efficiently by only reading those indices initially, if that proves
        # important
        if (len(var_cent.shape)>1):
            var_cent = var_cent[:,space_indices]
        else:
            var_cent = var_cent[space_indices]

    return var_cent
                     
def sww_files_are_equal(filename1, filename2):
    """Read and compare numerical values of two sww files: filename1 and filename2
    
    If they are identical (up to a tolerance) the return value is True
    If anything substantial is different, the return value is False.
    """

    if not (filename1.endswith('.sww') and filename2.endswith('.sww')):
        msg = f'Filenames {filename1} and {filename2} must both end with .sww'
        raise Exception(msg)
    

    domain1_v = get_output(filename1)
    domain1_c = get_centroids(domain1_v)

    domain2_v = get_output(filename2)
    domain2_c = get_centroids(domain2_v)

    if not num.allclose(domain1_c.stage, domain2_c.stage):
        return False
        
    if not num.allclose(domain1_c.xmom, domain2_c.xmom):
        return False
        
    if not num.allclose(domain1_c.ymom, domain2_c.ymom):
        return False
        
    if not num.allclose(domain1_c.xvel, domain2_c.xvel):
        return False
        
    if not num.allclose(domain1_c.yvel, domain2_c.yvel):
        return False
        
    if not num.allclose(domain1_v.x, domain2_v.x):
        return False
        
    if not num.allclose(domain1_v.y, domain2_v.y):
        return False
        
    # Otherwise, they are deemed to be identical
    return True        
    
