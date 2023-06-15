#!/bin/env python
from __future__ import print_function, division

import os, numpy as np, h5py, pyvista as pv


def size(meshDict,pointData):
	'''
	Return the size accoding to the type of data
	'''
	return meshDict['xyz'].shape[0] if pointData else meshDict['eltype'].shape[0]

def reshape_var(meshDict,var,point,ndim):
	'''
	Reshape a variable according to the mesh
	'''
	# Obtain number of points from the mesh
	npoints = size(meshDict,point)

	# Only reshape the variable if ndim > 1
	out = np.ascontiguousarray(var.reshape((npoints,ndim),order='C') if ndim > 1 else var)
	# Build 3D vector in case of 2D array
	if ndim == 2: out = np.hstack((out,np.zeros((npoints,1))))
	return out


## Functions to read an HDF5 dataset

PYLOM_H5_VERSION = (2,0)

def _h5_load_mesh(file):
	'''
	Load the mesh inside the HDF5 file
	'''
	if not 'MESH' in file.keys(): return None
	# Read mesh type
	mtype  = [c.decode('utf-8') for c in file['MESH']['type'][:]][0]
	# Read cell related variables
	conec  = np.array(file['MESH']['connectivity'][:,:],np.int32)
	eltype = np.array(file['MESH']['eltype'][:],np.int32) 
	cellO  = np.array(file['MESH']['cellOrder'][:],np.int32)
	# Read point related variables
	xyz    = np.array(file['MESH']['xyz'][:,:],np.double) 
	pointO = np.array(file['MESH']['pointOrder'][:],np.int32)
	# Fix the connectivity to start at zero
	conec -= conec.min()
	# Return
	return {'type':mtype,'xyz':xyz,'connectivity':conec,'eltype':eltype}

def _h5_load_variables(file):
	'''
	Load the variables inside the HDF5 file
	'''
	# Read time
	time = np.array(file['time'][:])
	# Read variables
	varDict = {}
	for v in file['VARIABLES'].keys():
		# Load point and ndim
		point = bool(file['VARIABLES'][v]['point'][0])
		ndim  = int(file['VARIABLES'][v]['ndim'][0])
		# Read the values
		value = np.array(file['VARIABLES'][v]['value'][:,:])
		# Generate dictionary
		varDict[v] = {'point':point,'ndim':ndim,'value':value}
	# Return
	return time, varDict

def h5_load(fname,mpio=True):
	'''
	Load a dataset in HDF5
	'''
	# Open file for reading
	file = h5py.File(fname,'r')
	# Check the file version
	version = tuple(file.attrs['Version'])
	if not version == PYLOM_H5_VERSION:
		printf('File version <%s> not matching the tool version <%s>!'%(str(file.attrs['Version']),str(PYLOM_H5_VERSION)))
		exit(-1)
	# Read the mesh
	meshDict = _h5_load_mesh(file)
	# Read the variables
	time, varDict = _h5_load_variables(file)
	file.close()
	return meshDict, time, varDict


## Functions to write a VTKH5

VTKFMT  = '%s-%08d-vtk.hdf'
VTKTYPE = 'UnstructuredGrid'
VTKVERS = np.array([1,0],np.int32)

ELTYPE2VTK = {
	0 : 0,  # Empty cell
	2 : 5,  # Triangular cell
	3 : 9,  # Quadrangular cell
	4 : 10, # Tetrahedral cell
	6 : 13, # Linear prism
	7 : 14, # Pyramid
	5 : 12, # Hexahedron
}

def _vtkh5_create_structure(file):
	'''
	Create the basic structure of a VTKH5 file
	'''
	# Create main group
	main = file.create_group('VTKHDF')
	main.attrs['Type']    = VTKTYPE
	main.attrs['Version'] = VTKVERS
	# Create cell data group
	main.create_group('CellData')
	main.create_group('PointData')
	main.create_group('FieldData')
	# Return created groups
	return main

def _vtkh5_connectivity_and_offsets(lnods):
	'''
	Build the offsets array (starting point per each element)

	'''
	# Compute the number of points per cell
	ppcell = np.sum(lnods >= 0,axis=1)
	# First we flatten the connectivity array
	lnodsf = lnods.flatten('c')
	# Now we get rid of any -1 entries for mixed meshes
	lnodsf = lnodsf[lnodsf>=0]
	# Now build the offsets vector
	offset = np.zeros((ppcell.shape[0]+1,),np.int32)
	offset[1:] = np.cumsum(ppcell)
	return lnodsf, offset

def _vtkh5_write_mesh_serial(file,xyz,lnods,ltype):
	'''
	Write the mesh and the connectivity to the VTKH5 file.
	'''
	# Create dataset for number of points
	npoints, ndim = xyz.shape
	file.create_dataset('NumberOfPoints',(1,),dtype=int,data=npoints)
	file.create_dataset('Points',(npoints,ndim),dtype=np.double,data=xyz)
	# Create dataset for number of cells
	lnods, offsets = _vtkh5_connectivity_and_offsets(lnods)
	ncells = ltype.shape[0]
	ncsize = lnods.shape[0]
	file.create_dataset('NumberOfCells',(1,),dtype=int,data=ncells)
	file.create_dataset('NumberOfConnectivityIds',(1,),dtype=int,data=ncsize)
	file.create_dataset('Connectivity',(ncsize,),dtype=int,data=lnods)
	file.create_dataset('Offsets',(ncells+1,),dtype=int,data=offsets)
	file.create_dataset('Types',(ncells,),dtype=np.uint8,data=ltype)
	# Return some parameters
	return npoints, ncells 

def _vtkh5_save_field_serial(main,instant,time,meshDict,varDict):
	'''
	Save the field component into a VTKH5 file (serial)
	'''
	# Open file for writing (append to a mesh)
	npoints = int(main['NumberOfPoints'][0])
	# Write dt and instant as field data
	main['FieldData'].create_dataset('InstantValue',(1,),dtype=int,data=instant)
	main['FieldData'].create_dataset('TimeValue',(1,),dtype=float,data=time)
	# Write the variables
	for var in varDict.keys():
		# Obtain in which group to write
		group = 'PointData' if varDict[var]['value'].shape[0] == npoints else 'CellData'
		# Create and write
		v = varDict[var]['value'][:,instant] if len(varDict[var]['value'].shape) > 1 else varDict[var]['value']
		v = reshape_var(meshDict,v,varDict[var]['point'],varDict[var]['ndim'])
		main[group].create_dataset(var,v.shape,dtype=v.dtype,data=v)


def vtkh5_save(casename,instant,time,meshDict,varDict,basedir='./'):
	'''
	Save the mesh component into a VTKH5 file (serial)
	'''
	os.makedirs(basedir,exist_ok=True)
	# Open file for writing
	file = h5py.File(os.path.join(basedir,VTKFMT%(casename,instant)),'w')
	# Create the file structure
	main = _vtkh5_create_structure(file)
	# Write the mesh
	ltypes = np.array([ELTYPE2VTK[i] for i in meshDict['eltype']])
	_vtkh5_write_mesh_serial(main,meshDict['xyz'],meshDict['connectivity'],ltypes)
	# Write the field
	_vtkh5_save_field_serial(main,instant,time,meshDict,varDict)
	# Close file
	file.close()


## Functions to plot using pyVista

def _cells_and_offsets(conec):
	'''
	Build the offsets and cells array to create am
	UnstructuredGrid
	'''
	# Compute points per cell
	ppcell = np.sum(conec >= 0,axis=1)
	# Compute cells for pyVista, with the number of points per cell on front
	cells = np.c_[ppcell,conec]
	# Now we get rid of any -1 entries for mixed meshes
	cellsf = cells.flatten('c')
	cellsf = cellsf[cellsf>=0]
	# Now build the offsets vector
	offset = np.zeros((ppcell.shape[0]+1,),np.int32)
	offset[1:] = np.cumsum(ppcell)
	return cellsf, offset

def plotSnapshot(meshDict,varDict,vars=[],instant=0,**kwargs):
	'''
	Plot using pyVista
	'''
	# First create the unstructured grid
	cells, offsets = _cells_and_offsets(meshDict['connectivity'])
	# Create the unstructured grid
	ltypes = np.array([ELTYPE2VTK[i] for i in meshDict['eltype']])
	ugrid  =  pv.UnstructuredGrid(offsets,cells,ltypes,meshDict['xyz']) if pv.vtk_version_info < (9,) else pv.UnstructuredGrid(cells,ltypes,meshDict['xyz'])
	print(ugrid)
	# Load the variables inside the unstructured grid
	for v in vars:
		var = varDict[v]['value'][:,instant] if len(varDict[v]['value'].shape) > 1 else varDict[v]['value']
		var = reshape_var(meshDict,var,varDict[v]['point'],varDict[v]['ndim'])
		if varDict[v]['point']:
			ugrid.point_data[v] = var
		else:
			ugrid.cell_data[v]  = var
	# Launch 

	return ugrid.plot(**kwargs)