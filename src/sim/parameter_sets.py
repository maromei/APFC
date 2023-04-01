PARAM_SETS = [
    {"dB0": 0.04000, "Bx": 1.00000, "n0": 0.00000},
    {"dB0": 0.14019, "Bx": 0.45981, "n0": 0.84900},
    {"dB0": -0.13333, "Bx": 0.33333, "n0": 1.12000},
    {"dB0": 0.14019, "Bx": 0.45981, "n0": 0.84900},
    {"dB0": 0.14019, "Bx": 0.45981, "n0": 0.15100},
    {"dB0": 0.01200, "Bx": 0.98800, "n0": -0.03000},
    {"dB0": 0.01200, "Bx": 0.98800, "n0": 0.00000},
]
"""
Contains :code:`dB0` :code:`Bx` and :code:`n0` values.

Content is:

+-------+--------+-------+--------+
| index | dB0    | Bx    | n0     |
+=======+========+=======+========+
| 0     | 0.04000|1.00000| 0.00000|
+-------+--------+-------+--------+
| 1     | 0.14019|0.45981| 0.84900|
+-------+--------+-------+--------+
| 2     |-0.13333|0.33333| 1.12000|
+-------+--------+-------+--------+
| 3     | 0.14019|0.45981| 0.84900|
+-------+--------+-------+--------+
| 4     | 0.14019|0.45981| 0.15100|
+-------+--------+-------+--------+
| 5     | 0.01200|0.98800|-0.03000|
+-------+--------+-------+--------+
| 6     | 0.01200|0.98800|-0.00000|
+-------+--------+-------+--------+
"""

#: A default config
DEFAULT_CONFIG = {
    "Bx": 0.988,
    "dB0": 0.012,
    "n0": -0.03,
    "t": 0.5,
    "v": 0.3333333333333333,
    "beta": 1.0,
    "dt": 0.75,
    "initRadius": 10,
    "interfaceWidth": 5.0,
    "initEta": 0.18717797887081347,
    "thetaDiv": 6,
    "thetaCount": 50,
    "numPtsX": 1000,
    "numPtsY": 1,
    "xlim": 200,
    "keepEtaConst": False,
    "G": [[-0.8660254037844386, -0.5], [0.0, 1.0], [0.8660254037844386, -0.5]],
    "numT": 6000,
    "writeEvery": 200,
    "eqSteps": 0,
    "simPath": "",
    "simType": "base",
    "vary": False,
    "varyParam": "Bx",
    "varyStart": 0.0,
    "varyEnd": 1.0,
    "varyAmount": 100,
}
