#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdlib.h>
#include <stdio.h>

#define BOSS_IMPLEMENTATION

#ifndef BOSS_H_
#include "boss_fly.h"
#endif // BOSS_H_


static PyObject* boss(PyObject* self, PyObject* args, PyObject* kwargs)
{
  (void)self;   // mark 'self' as unused to suppress warning
  PyObject* array_obj;
  float lambda = 1.0;
  int restarts = 1;

  static char *kwlist[] = {"X", "lambda", "restarts", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!|fi", kwlist, 
                                   &PyArray_Type, &array_obj,
                                   &lambda, &restarts)) return NULL;

  PyArrayObject *array = (PyArrayObject*)array_obj;
  if (PyArray_TYPE(array) != NPY_FLOAT32) {
    PyErr_SetString(PyExc_TypeError, "Array must be of type float32.");
    return NULL;
  }

  if (PyArray_NDIM(array) != 2) {
    PyErr_SetString(PyExc_TypeError, "Array must be 2-dimensional.");
    return NULL;
  }

  npy_intp *dims = PyArray_SHAPE(array);
  size_t n = dims[1];
  size_t p = dims[0];

  float *X = (float*)PyArray_DATA(array);

  dims[1] = p;
  PyObject* array_out = PyArray_SimpleNew(2, dims, NPY_UINT8);
  if (!array_out) return PyErr_NoMemory();

  // ACCEPT A RANDOM SEED AS INPUT FROM PYTHON
  srand(time(NULL));

  // DO I NEED TO CAST THIS?
  uint8_t *graph = (uint8_t*)PyArray_DATA((PyArrayObject*)array_out);

  boss_search(X, n, p, lambda, restarts, graph);
  
  return array_out;
}


static PyMethodDef methods[] = {
  { "boss", (PyCFunction)boss, METH_VARARGS | METH_KEYWORDS, "Runs the BOSS algorithm." },
  { NULL, NULL, 0, NULL }
};


static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "boss",
  "a module for the boss causal discovery algorithm",
  -1,
  methods,
  NULL,
  NULL,
  NULL,
  NULL
};


PyMODINIT_FUNC PyInit_boss(void)
{
  import_array();
  return PyModule_Create(&moduledef);
}
