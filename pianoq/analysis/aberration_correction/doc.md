# Rodrigo dimensions
## fiber modes
55 modes in his fiber, with 128X128 pixels for each mode

He calls this high-def, and then down-scales when performing the optimization. 

From 2D picture to 1d vetor do something like: pofile.reshape(1,self._npoints**2)

## TM pixel basis 
first dim is 41^2, and second dim is 35^2. 
I imagine the higher resolution is the output with the camera. 

# My things 
## the fiber
* current fiber is HPSC25, which is a step-index fiber, NA of 0.1, here: https://www.thorlabs.com/thorproduct.cfm?partnumber=HPSC25
* This was superseded by FG025LJA, which is supposed to have the same specs 

## Do now 
* generate modes with 128X128. Done. 
* Take Amit TM and translate to pixel basis 

# Questions
* Why does SI have more modes than GRIN? 
* I get ~150 modes (per polarization) with the SI, which is much more than the 
55 in the PRX or the ~35 in Guys work. Why is this? 
* Was Guys fiber also SI? 


# Summary:
* needed to generate modes from SI - means there aren't degenerate groups which is sad 
* There are much more theoretical modes than speckle grains - it is important to take only the first modes  
* The optimization code performs a resize - it is important to zero the field outside of the fiber core to not confuse the system
* Important to normalize the modes so that the energy ratio will be meaningful 
* The final TM isn't even close to diagonal, but that is probably a feature of the 4m long fiber with the data I looked at
