import math
import numpy as np
import time
from scipy.ndimage.interpolation import map_coordinates, zoom
from scipy.ndimage.interpolation import geometric_transform

def rotate(point, angle):

    res = np.array(point)
    res[1] =  math.sin(angle)*point[2] + math.cos(angle)*point[1]
    res[2] = -math.sin(angle)*point[1] + math.cos(angle)*point[2]

    return res

def control_point_offsets_to_map(control_point_offsets, shape, interpolate_order=1):

    print("Creating dense offset map...")

    dims = len(shape)

    # upsample control points to shape of original image
    offsets = np.zeros((dims,) + shape)
    control_point_zoom = tuple(float(s)/c for s,c in zip(shape, control_point_offsets.shape[1:]))

    for d in range(dims):
        offsets[d] = zoom(control_point_offsets[d], zoom=control_point_zoom, mode='nearest', order=interpolate_order)

    return offsets

def create_identity_transformation(shape):

    axis_ranges = (np.arange(d, dtype=np.float) for d in shape)
    return np.meshgrid(*axis_ranges, indexing='ij')

def create_rotation_transformation(shape, angle):

    print("Creating rotation transformation with:")
    print("\tangle: " + str(angle))

    dims = len(shape)
    control_points = (2,)*dims

    # map control points to world coordinates
    control_point_scaling_factor = tuple(float(s-1) for s in shape)

    # rotate control points
    center = np.array([0.5*(d-1) for d in shape])

    control_point_offsets = np.zeros((dims,) + control_points, dtype=np.float)
    for control_point in np.ndindex(control_points):

        point = np.array(control_point)*control_point_scaling_factor
        center_offset = np.array([p-c for c,p in zip(center, point)])
        rotated_offset = rotate(center_offset, angle)
        displacement = rotated_offset - center_offset
        control_point_offsets[(slice(None),) + control_point] += displacement

    return control_point_offsets_to_map(control_point_offsets, shape)

def create_elastic_transformation(shape, num_control_points = 10, jitter_sigma = 1.0):

    dims = len(shape)

    try:
        control_points = tuple((d for d in num_control_points))
    except:
        control_points = (num_control_points,)*dims
    try:
        sigmas = [ s for s in jitter_sigma ]
    except:
        sigmas = [jitter_sigma]*dims

    assert np.prod(control_points) > 0, "Number of control points is not allowed to be zero"

    print("Creating elastic transformation with:")
    print("\tcontrol points per axis: " + str(control_points))
    print("\taxis jitter sigmas     : " + str(sigmas))

    # jitter control points
    control_point_offsets = np.zeros((dims,) + control_points, dtype=np.float)
    for d in range(dims):
        if sigmas[d] > 0:
            sigma = sigmas[d]
            control_point_offsets[d] = np.random.normal(scale=sigma, size=control_points)

    return control_point_offsets_to_map(control_point_offsets, shape, interpolate_order=3)

def apply_transformation(image, transformation, interpolate = True, outside_value = 0):

    print("Applying transformation...")
    order = 1 if interpolate == True else 0
    return map_coordinates(image, transformation, order=order, mode='constant', cval=outside_value)

if __name__ == "__main__":

    import h5py

    gt  = np.array(h5py.File('gt.hdf', 'r')['main'])
    # raw = np.array(h5py.File('raw.hdf', 'r')['main'])
    raw = np.zeros(gt.shape, dtype=np.float)
    raw[:] = 0.5
    raw[:,:10,:] = 0.75
    raw[:10,:10,:10] = 1
    raw[:,::10,:] = 1
    raw[:,:,::10] = 1

    h5py.File('raw_original.hdf', 'w')['main'] = raw
    h5py.File('gt_original.hdf', 'w')['main'] = gt

    transformation = create_identity_transformation(raw.shape)

    start = time.time()
    transformation += create_elastic_transformation(
            raw.shape,
            num_control_points = [3,10,10],
            jitter_sigma = [0.3, 10, 10])
    print("Created elastic transformation in " + str(time.time() - start)  + "s")

    start = time.time()
    transformation += create_rotation_transformation(
            raw.shape,
            math.pi/4)
    print("Created rotation transformation in " + str(time.time() - start)  + "s")

    start = time.time()
    raw = apply_transformation(raw, transformation)
    gt  = apply_transformation(gt, transformation, interpolate = False)
    print("Applied transformations in " + str(time.time() - start)  + "s")

    print("Saving results")

    h5py.File('raw_deformed.hdf', 'w')['main'] = raw
    h5py.File('gt_deformed.hdf', 'w')['main'] = gt
