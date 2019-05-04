import numpy as np
import matplotlib.pyplot as plt


# from pynverse import inversefunc


def face_distance_to_conf(face_distance, face_match_threshold):
    if face_distance > face_match_threshold:
        return (1.0 - face_distance) / ((1.0 - face_match_threshold) * 2)
    else:
        # return _face_distance_to_conf_non_linear(face_distance, face_match_threshold)
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * np.power((linear_val - 0.5) * 2, 0.2))


# def _face_distance_to_conf_non_linear(x, t):
#     return 1.0 + x * (np.power((1.0 - (x / t)), 0.2) - 1.0)


def graph_function(fn, step=0.00001):
    x = np.arange(0, 1 + step, step)
    plt.plot(
        x,
        [fn(i) for i in x]
    )
    plt.show()


def conf_to_face_distance(conf, table):
    return table[0][np.abs(table[1] - conf).argmin()]
