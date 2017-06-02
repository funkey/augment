import math
import numpy as np
import time
from scipy.ndimage.interpolation import map_coordinates, zoom

def rotate(point, angle):

    res = np.array(point)
    res[0] =  math.sin(angle)*point[1] + math.cos(angle)*point[0]
    res[1] = -math.sin(angle)*point[0] + math.cos(angle)*point[1]

    return res

def upscale_transformation(transformation, output_shape, interpolate_order=1):

    input_shape = transformation.shape[1:]

    # print("Upscaling control points")
    # print("\tfrom               : " + str(input_shape))
    # print("\tto                 : " + str(output_shape))
    # print("\tinterpolation order: " + str(interpolate_order))

    dims = len(output_shape)
    scale = tuple(float(s)/c for s,c in zip(output_shape, input_shape))

    start = time.time()
    scaled = np.zeros((dims,)+output_shape, dtype=np.float32)
    for d in range(dims):
        zoom(transformation[d], zoom=scale, output=scaled[d], order=interpolate_order)
    # print("\tupsampled in " + str(time.time() - start) + "s")

    return scaled

def create_identity_transformation(shape, subsample=1):

    dims = len(shape)
    subsample_shape = tuple(max(1,int(s/subsample)) for s in shape)
    step_width = tuple(float(shape[d]-1)/(subsample_shape[d]-1) if subsample_shape[d] > 1 else 1 for d in range(dims))

    axis_ranges = (
            np.arange(subsample_shape[d], dtype=np.float32)*step_width[d]
            for d in range(dims)
    )
    return np.array(np.meshgrid(*axis_ranges, indexing='ij'), dtype=np.float32)

def create_rotation_transformation(shape, angle, subsample=1):

    dims = len(shape)
    subsample_shape = tuple(max(1,int(s/subsample)) for s in shape)
    control_points = (2,)*dims

    # map control points to world coordinates
    control_point_scaling_factor = tuple(float(s-1) for s in shape)

    # rotate control points
    center = np.array([0.5*(d-1) for d in shape])

    # print("Creating rotation transformation with:")
    # print("\tangle : " + str(angle))
    # print("\tcenter: " + str(center))

    control_point_offsets = np.zeros((dims,) + control_points, dtype=np.float32)
    for control_point in np.ndindex(control_points):

        point = np.array(control_point)*control_point_scaling_factor
        center_offset = np.array([p-c for c,p in zip(center, point)], dtype=np.float32)
        rotated_offset = np.array(center_offset)
        rotated_offset[-2:] = rotate(center_offset[-2:], angle)
        displacement = rotated_offset - center_offset
        control_point_offsets[(slice(None),) + control_point] += displacement

    return upscale_transformation(control_point_offsets, subsample_shape)

def create_elastic_transformation(shape, control_point_spacing = 100, jitter_sigma = 10.0, subsample = 1):

    dims = len(shape)
    subsample_shape = tuple(max(1,int(s/subsample)) for s in shape)

    try:
        spacing = tuple((d for d in control_point_spacing))
    except:
        spacing = (control_point_spacing,)*dims
    try:
        sigmas = [ s for s in jitter_sigma ]
    except:
        sigmas = [jitter_sigma]*dims

    control_points = tuple(
            max(1,int(round(float(shape[d])/spacing[d])))
            for d in range(len(shape))
    )

    # print("Creating elastic transformation with:")
    # print("\tcontrol points per axis: " + str(control_points))
    # print("\taxis jitter sigmas     : " + str(sigmas))

    # jitter control points
    control_point_offsets = np.zeros((dims,) + control_points, dtype=np.float32)
    for d in range(dims):
        if sigmas[d] > 0:
            control_point_offsets[d] = np.random.normal(scale=sigmas[d], size=control_points)

    return upscale_transformation(control_point_offsets, subsample_shape, interpolate_order=3)

def apply_transformation(image, transformation, interpolate = True, outside_value = 0, output = None):

    # print("Applying transformation...")
    order = 1 if interpolate == True else 0
    output = image.dtype if output is None else output
    return map_coordinates(image, transformation, output=output, order=order, mode='constant', cval=outside_value)
