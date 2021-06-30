#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/*
 * single_type_logit.c
 * This is the C code for creating your own
 * NumPy ufunc for a logit function.
 *
 * In this code we only define the ufunc for
 * a single dtype. The computations that must
 * be replaced to create a ufunc for
 * a different function are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

static PyMethodDef GamerMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */

double h(double kT)
{
    double x;
    x = 2.25 * kT * kT;
    return 2.5 * kT + x / (1.0 + sqrt(x + 1.0));
}

double htilde(double kT, double c)
{
    return h(kT) * c * c;
}

double gamma(double kT)
{
    double x, c_p, c_v;
    x = 2.25 * kT / sqrt(2.25 * kT * kT + 1.0);
    c_p = 2.5 + x;
    c_v = 1.5 + x;
    return c_p / c_v;
}

double _four_vel(double mom, double dens, double kT, double c)
{
    return mom * c * c / (dens * (htilde(kT, c) + c * c));
}

double _lorentz_factor(double momx, double momy, double momz,
                       double dens, double kT, double c)
{
    double u2, c2, vx, vy, vz;

    c2 = c * c;
    vx = _four_vel(momx, dens, kT, c);
    vy = _four_vel(momy, dens, kT, c);
    vz = _four_vel(momz, dens, kT, c);
    u2 = vx * vx + vy * vy + vz * vz;
    return sqrt(1.0 + u2 / c2);
}

static void double_htilde_eos4(char **args, npy_intp *dimensions,
                               npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *cspeed = args[1], *out = args[2];
    npy_intp in_step = steps[0], out_step = steps[2];

    double kT, c;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/

        kT = *(double *)in;
        c = *(double *)cspeed;
        *((double *)out) = htilde(kT, c);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void double_cs_eos4(char **args, npy_intp *dimensions,
                           npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double kT, x, h, cs2;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/

        kT = *(double *)in;
        x = 2.25 * kT * kT;
        h = 2.5 * kT + x / (1.0 + sqrt(x + 1.0)) + 1.0;
        cs2 = kT / (3.0 * h);
        cs2 *= (5.0 * h - 8.0 * kT) / (h - kT);
        *((double *)out) = sqrt(cs2);
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void double_gamma_eos4(char **args, npy_intp *dimensions,
                              npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    npy_intp in_step = steps[0], out_step = steps[1];

    double kT, x;

    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/

        kT = *(double *)in;
        x = 2.25 * kT / sqrt(2.25 * kT * kT + 1.0);
        *((double *)out) = 2.5 + x / 1.5 + x;
        /*END main ufunc computation*/

        in += in_step;
        out += out_step;
    }
}

static void double_density_eos4(char **args, npy_intp *dimensions,
                                npy_intp* steps, void* data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *dens = args[0], *momx = args[1], *momy = args[2],
         *momz = args[3], *ener = args[4], *cspeed = args[5], 
         *out = args[6];
    npy_intp dens_step = steps[0], momx_step = steps[1], momy_step = steps[2],
             momz_step = steps[3], ener_step = steps[4], out_step = steps[6];

    double kT, mx, my, mz, rho, c;

    c = *(double *)cspeed;
    for (i = 0; i < n; i++) {
        /*BEGIN main ufunc computation*/

        rho = *(double *)dens;
        mx = *(double *)momx;
        my = *(double *)momy;
        mz = *(double *)momz;
        kT = *(double *)ener;

        *((double *)out) = rho / _lorentz_factor(mx, my, mz, rho, kT, c);
        
        /*END main ufunc computation*/

        dens += dens_step;
        momx += momx_step;
        momy += momy_step;
        momz += momz_step;
        ener += ener_step;
        out += out_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction htilde_eos4_funcs[1] = {&double_htilde_eos4};
PyUFuncGenericFunction gamma_eos4_funcs[1] = {&double_gamma_eos4};
PyUFuncGenericFunction cs_eos4_funcs[1] = {&double_cs_eos4};
PyUFuncGenericFunction dens_eos4_funcs[1] = {&double_density_eos4};

/* These are the input and return dtypes of logit.*/
static char types[2] = {NPY_DOUBLE, NPY_DOUBLE};
static char dens_types[7] = {NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE,
                             NPY_DOUBLE, NPY_DOUBLE, NPY_DOUBLE};

static void *data[1] = {NULL};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "fields",
    NULL,
    -1,
    GamerMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_fields(void)
{
    PyObject *m, *htilde_eos4, *gamma_eos4, *cs_eos4, *dens_eos4, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    htilde_eos4 = PyUFunc_FromFuncAndData(htilde_eos4_funcs, data, types, 1, 1, 1,
                                          PyUFunc_None, "htilde_eos4",
                                          "htilde_eos4_docstring", 0);
    gamma_eos4 = PyUFunc_FromFuncAndData(gamma_eos4_funcs, data, types, 1, 1, 1,
                                         PyUFunc_None, "gamma_eos4",
                                         "gamma_eos4_docstring", 0);
    cs_eos4 = PyUFunc_FromFuncAndData(cs_eos4_funcs, data, types, 1, 1, 1,
                                      PyUFunc_None, "sound_speed_eos4",
                                      "cs_eos4_docstring", 0);
    dens_eos4 = PyUFunc_FromFuncAndData(dens_eos4_funcs, data, dens_types, 1, 6, 1,
                                        PyUFunc_None, "dens_eos4",
                                        "dens_eos4_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "htilde_eos4", htilde_eos4);
    Py_DECREF(htilde_eos4);
    PyDict_SetItemString(d, "gamma_eos4", gamma_eos4);
    Py_DECREF(gamma_eos4);
    PyDict_SetItemString(d, "sound_speed_eos4", cs_eos4);
    Py_DECREF(cs_eos4);
    PyDict_SetItemString(d, "dens_eos4", dens_eos4);
    Py_DECREF(dens_eos4);

    return m;
}
