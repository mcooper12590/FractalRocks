import numpy as np
cimport numpy as np
cimport cython
DTYPE = np.float32

ctypedef np.float32_t DTYPE_t

#@cython.cdivision(True)
#cdef DTYPE_t div(DTYPE_t t, DTYPE_t b):
#    return t/b

@cython.boundscheck(False)
@cython.wraparound(False)
def corr2d(DTYPE_t [:,:] img, int dist):
    cdef Py_ssize_t y_max = img.shape[0]
    cdef Py_ssize_t x_max = img.shape[1]

    cdef Py_ssize_t e = <Py_ssize_t>dist

    cdef np.ndarray[DTYPE_t, ndim=2] g_r_e = np.zeros((<int>y_max,<int>x_max), dtype=DTYPE)
    cdef DTYPE_t [:,:] g_r_e_v = g_r_e
    cdef DTYPE_t n_non_zero, gray_sum
    cdef DTYPE_t gray_e
    cdef Py_ssize_t i, j

    for i in range(y_max):
        for j in range(x_max):
            n_non_zero = <DTYPE_t>0.0; gray_sum = <DTYPE_t>0.0
            if img[i][j] == 0.0:
                continue

            if i-e>0:
                gray_e = img[i-e][j]
                if gray_e != 0.0:
                    n_non_zero += 1.0
                    gray_sum += gray_e

            if i+e<y_max:
                gray_e = img[i+e][j]
                if gray_e != 0.0:
                    n_non_zero += 1.0
                    gray_sum += gray_e

            if j-e>0:
                gray_e = img[i][j-e]
                if gray_e != 0.0:
                    n_non_zero += 1.0
                    gray_sum += gray_e

            if j+e<x_max:
                gray_e = img[i][j+e]
                if gray_e != 0.0:
                    n_non_zero += 1.0
                    gray_sum += gray_e

            if n_non_zero != 0:
                g_r_e_v[i][j] = gray_sum/n_non_zero

            else:
                continue

    return g_r_e

def corr3d(DTYPE_t [:,:,:] img, int dist):
    cdef Py_ssize_t z_max = img.shape[0]
    cdef Py_ssize_t y_max = img.shape[1]
    cdef Py_ssize_t x_max = img.shape[2]

    cdef Py_ssize_t e = <Py_ssize_t>dist

    cdef np.ndarray[DTYPE_t, ndim=3] g_r_e = np.zeros((<int>z_max,<int>y_max,<int>x_max), dtype=DTYPE)
    cdef DTYPE_t [:,:,:] g_r_e_v = g_r_e
    cdef DTYPE_t n_non_zero, gray_sum
    cdef DTYPE_t gray_e
    cdef Py_ssize_t i, j, k

    for i in range(z_max):
        for j in range(y_max):
            for k in range(x_max):
                n_non_zero = <DTYPE_t>0.0; gray_sum = <DTYPE_t>0.0
                if img[i][j][k] == 0.0:
                    continue

                if i-e>0:
                    gray_e = img[i-e][j][k]
                    if gray_e != 0.0:
                        n_non_zero += 1.0
                        gray_sum += gray_e

                if i+e<z_max:
                    gray_e = img[i+e][j][k]
                    if gray_e != 0.0:
                        n_non_zero += 1.0
                        gray_sum += gray_e

                if j-e>0:
                    gray_e = img[i][j-e][k]
                    if gray_e != 0.0:
                        n_non_zero += 1.0
                        gray_sum += gray_e

                if j+e<y_max:
                    gray_e = img[i][j+e][k]
                    if gray_e != 0.0:
                        n_non_zero += 1.0
                        gray_sum += gray_e

                if k-e>0:
                    gray_e = img[i][j][k-e]
                    if gray_e != 0.0:
                        n_non_zero += 1.0
                        gray_sum += gray_e

                if k+e<x_max:
                    gray_e = img[i][j][k+e]
                    if gray_e != 0.0:
                        n_non_zero += 1.0
                        gray_sum += gray_e

                if n_non_zero != 0:
                    g_r_e_v[i][j][k] = gray_sum/n_non_zero

                else:
                    continue

        return g_r_e
