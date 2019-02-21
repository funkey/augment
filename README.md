# augment

A simple elastic augmentation for ND arrays.

# Installation

`pip install augment-nd`

# Usage

```
import augment
import numpy as np
import math

# create some example data
image = np.zeros((100,500,500), dtype=np.float)
image[:] = 0.5
image[:,:10,:] = 0.75
image[:10,:10,:10] = 1
image[:,::10,:] = 1
image[:,:,::10] = 1

transformation = augment.create_identity_transformation(image.shape)

# jitter in 3D
transformation += augment.create_elastic_transformation(
        image.shape,
        num_control_points = [3,10,10],
        jitter_sigma = [0.3, 10, 10])

# rotate around z axis
transformation += augment.create_rotation_transformation(
        image.shape,
        math.pi/4)

# apply transformation
image = augment.apply_transformation(image, transformation)
```
