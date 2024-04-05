
from enum import Enum

class NG_View(Enum):
    """
    SVD is a special case, for all other views the parameters are directly taken from neuroglancer.
    """
    SVD = ("svd", None, None, None)
    LAT_RI = (
        "lateral-right"
      , [16770, 35264, 35758]
      , [-0.08796502649784088, -0.3553735613822937, 0.08154760301113129, -0.926996111869812]
      , 45276
    )
    ANT_RI = (
        "anterior-right"
      , [16770, 35264, 35758]
      , [0.01605566404759884, 0.35593777894973755, 0.13652965426445007, -0.9243431091308594]
      , 45276
    )
    VEN_RI = (
        "ventral-right"
      , [16770, 35264, 35758]
      , [-0.5976191759109497, 0.36472800374031067, -0.09956551343202591, -0.7070441842079163]
      , 45276
    )
    LAT_POS_VEN_RI = (
        "iso-right"
      , [16770, 35264, 35758]
      , [0.265063613653183, -0.6768622994422913, -0.167705237865448, -0.665938138961792]
      , 45276
    )

    GALLERY1 = (
        "gallery-view-01"
      , [4,-3,-4]
      , [-0.03744018, -0.02808013,  0.03744018, -0.99820237]
      , 22222
    )

    def __init__(self, path, camera_location, camera_orientation, scale):
        self.path = path
        self.location = camera_location
        self.orientation = camera_orientation
        self.scale = scale
