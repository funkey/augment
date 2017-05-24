import time
import numpy as np
from .transform import create_identity_transformation, create_elastic_transformation, create_rotation_transformation, upscale_transformation, apply_transformation

def create_transformation(shape, control_point_spacing, jitter_sigma, subsample, angle):

    transformation = create_identity_transformation(shape, subsample=subsample)
    print("Allocated identity transformation")

    start = time.time()
    transformation += create_elastic_transformation(
            shape,
            control_point_spacing=control_point_spacing,
            jitter_sigma=jitter_sigma,
            subsample=subsample)
    print("Added elastic transformation in " + str(time.time() - start)  + "s")

    start = time.time()
    transformation += create_rotation_transformation(
            shape,
            angle=angle,
            subsample=subsample)
    print("Added rotation transformation in " + str(time.time() - start)  + "s")

    if subsample > 1:
        start = time.time()
        transformation = upscale_transformation(transformation, shape)
        print("Upscaled transformation to final output shape in " + str(time.time() - start)  + "s")

    return transformation

def augment_all(sources, targets, control_point_spacing = 100, jitter_sigma = 10, subsample = 1, angle = 0):
    '''Augment the volumes in sources, store the results in targets.

    Both sources and targets are lists of volumes. They can be HDF5 datasets.
    '''

    assert len(sources) == len(targets)
    if len(sources) == 0:
        raise RuntimeError("At least one source and one target have to be provided.")

    shape = sources[0].shape
    for i in range(len(sources)):
        assert sources[i].shape == shape, "Shapes of all sources and targets must be the same"
        assert targets[i].shape == shape, "Shapes of all sources and targets must be the same"

    start = time.time()

    transformation = create_transformation(
            shape,
            control_point_spacing,
            jitter_sigma,
            subsample,
            angle)

    for source, target in zip(sources, targets):

        is_label = source.dtype in [np.uint16, np.uint32, np.uint64]

        if is_label:

            need_scipy_workaround = source.dtype == np.uint64

            if need_scipy_workaround:
                if np.max(source) > np.uint32(-1):
                    raise RuntimeError('''Due to a bug in scipy's map_coordinates, we need to transform your uint64 array into uint32. Unfortunately, you have values too large to do that safely.''')
                source = np.array(source, dtype=np.uint32)
                t = np.zeros(source.shape, dtype=np.uint64)
                apply_transformation(
                    source,
                    transformation,
                    interpolate=False,
                    outside_value=source.dtype.type(-1),
                    output=t)
                t[t>=np.uint32(-1)] = np.uint64(-1)
                target[:] = t

            else:

                target[:] = apply_transformation(
                        source,
                        transformation,
                        interpolate=False,
                        outside_value=source.dtype.type(-1))

        else:
            target[:] = apply_transformation(
                    source,
                    transformation,
                    interpolate=True,
                    outside_value=0)

    print("augmentation finished in " + str(time.time() - start) + "s")
