"""Simple water flow example using ANUGA

Water flowing along a spiral wall and draining into a hole in the centre.
"""

#------------------------------------------------------------------------------
# Import necessary modules
#------------------------------------------------------------------------------
from math import acos, cos, sin, sqrt, pi
import anuga


#------------------------------------------------------------------------------
# Setup computational domain
#------------------------------------------------------------------------------
length = 10.
width = 10.
dx = dy = 0.05           # Resolution: Length of subdivisions on both axes

# Create a domain with named boundaries "left", "right", "top" and "bottom"
domain = anuga.rectangular_cross_domain(int(length/dx), int(width/dy),
                                        len1=length, len2=width)

domain.set_name('spiral_wall')               # Output name

#------------------------------------------------------------------------------
# Setup initial conditions
#------------------------------------------------------------------------------
def topography_circlewall(x, y):

    z = x * 0.   # Flat surface
    
    c = (6, 8)   # Centre
    r_inner = 1
    r_outer = 2
        
    N = len(x)
    for i in range(N):

        # Distance from centre to this point        
        d = sqrt((x[i] - c[0])**2 + (y[i] - c[1])**2)
        
        # Angle from centre to this point
        cos_theta = (x[i] - c[0]) / d     
        theta = acos(cos_theta)

        # Find distances from centre to inner and outer periphery along this vector
        x_inner = r_inner * cos(theta) + c[0]
        y_inner = r_inner * sin(theta) + c[1]                
        d_inner = sqrt((x_inner - c[0])**2 + (y_inner - c[1])**2)         
                
        x_outer = r_outer * cos(theta) + c[0]
        y_outer = r_outer * sin(theta) + c[1]        
        d_outer = sqrt((x_outer - c[0])**2 + (y_outer - c[1])**2) 

        # Raise elevation for points between inner and outer distance
        if d_inner < d < d_outer:
            z[i] += 2                
        
    return z            
    
def topography(x, y):

    z = x * 0.   # Flat surface
    
    c = (5, 5)   # Centre
    r_inner = 1
    r_outer = 2
        
    N = len(x)
    for i in range(N):

        # Distance from centre to this point        
        d = sqrt((x[i] - c[0])**2 + (y[i] - c[1])**2)
        
        # Angle from centre to this point
        cos_theta = (x[i] - c[0]) / d     
        theta = acos(cos_theta)
        

        # Find distances from centre to inner and outer periphery along this vector
        r = r_inner * theta * 0.5  # spiral form        
        x_inner = r * cos(theta) + c[0]
        y_inner = r * sin(theta) + c[1]                
        d_inner = sqrt((x_inner - c[0])**2 + (y_inner - c[1])**2)         
                
        r = r_outer * theta * 0.5  # spiral form
        x_outer = r * cos(theta) + c[0]
        y_outer = r * sin(theta) + c[1]        
        d_outer = sqrt((x_outer - c[0])**2 + (y_outer - c[1])**2) 

        # Raise elevation for points between inner and outer distance
        if d_inner < d < d_outer:
            z[i] += 2                
        
    return z            
            

domain.set_quantity('elevation', topography) # Use function for elevation
domain.set_quantity('friction', 0.01)        # Constant friction 
domain.set_quantity('stage',                 # Dry bed
                    expression='elevation')  

#------------------------------------------------------------------------------
# Setup boundary conditions
#------------------------------------------------------------------------------
Bi = anuga.Dirichlet_boundary([0.4, 0, 0])         # Inflow
Br = anuga.Reflective_boundary(domain)             # Solid reflective walls

domain.set_boundary({'left': Br, 'right': Bi, 'top': Br, 'bottom': Br})

#------------------------------------------------------------------------------
# Evolve system through time
#------------------------------------------------------------------------------
for t in domain.evolve(yieldstep=0.2, finaltime=0):
    domain.print_timestepping_statistics()
