import numpy as np

LABEL_MAP = {
    0: "background",
    1: "rightHand",
    2: "rightUpLeg",
    3: "leftArm",
    4: "head",
    5: "leftLeg",
    6: "leftFoot",
    7: "torso",
    8: "rightFoot",
    9: "rightArm",
    10: "leftHand",
    11: "rightLeg",
    12: "leftForeArm",
    13: "rightForeArm",
    14: "leftUpLeg",
    15: "hips",
}

COLOR_MAP_INSTANCES = {
    0: (226., 226., 226.), #(174., 199., 232.),
    1: (120., 94., 240.), #purple 
    2: (254., 97., 0.), #orange
    3: (100., 143., 255.), #blue
    4: (220., 38., 127.), #pink
    5: (255., 176., 0.), #yellow
    6: (0., 255., 255.),
    7: (255., 204., 153.),
    8: (255., 102., 0.),
    9: (0., 128., 128.),
    10: (153., 153., 255.),
}

COLOR_MAP_PARTS = {
    0:  (226., 226., 226.),
    1:  (158.0, 143.0, 20.0),  # rightHand
    2:  (30,	74,	138),  # rightUpLeg     blue
    3:  (30,	74,	138),       # leftArm        blue
    4:  (167,	44,	44),      # head           red
    5:  (152.0, 78.0, 163.0),  # leftLeg
    6:  (76.0, 134.0, 26.0),   # leftFoot
    7:  (40	,84,	46),           # torso          green
    8:  (129.0, 0.0, 50.0),    # rightFoot
    9:  (239.0,	189.0,	68.0),      # rightArm       yellow
    10: (192.0, 100.0, 119.0), # leftHand
    11: (149.0, 192.0, 228.0), # rightLeg 
    12: (243.0, 232.0, 88.0),  # leftForeArm        
    13: (90., 64., 210.),      # rightForeArm
    14: (167,	44,	44), # leftUpLeg          red
    15: (129.0, 103.0, 106.0), # hips
}

map2color_instances = np.vectorize({key: item for key, item in COLOR_MAP_INSTANCES.items()}.get)
map2color_parts = np.vectorize({key: item for key, item in COLOR_MAP_PARTS.items()}.get)
