use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::PyResult;

use num_complex::Complex;

use crate::types::*;

pub trait Axpy<T> {
    fn axpy(a: T, x: &Self, y: &mut Self) -> ()
    where
        Self: Sized;
}

impl Axpy<f32> for Vecf {
    fn axpy(a: f32, x: &Self, y: &mut Self) -> () {
        unsafe { blas::saxpy(x.vec.len() as i32, a, &x.vec, 1, &mut y.vec, 1) }
    }
}

impl Axpy<f64> for Vecd {
    fn axpy(a: f64, x: &Self, y: &mut Self) -> () {
        unsafe { blas::daxpy(x.vec.len() as i32, a, &x.vec, 1, &mut y.vec, 1) }
    }
}

impl Axpy<Complex<f32>> for Vecx {
    fn axpy(a: Complex<f32>, x: &Self, y: &mut Self) -> () {
        unsafe {
            blas::caxpy(x.vec.len() as i32, a, &x.vec, 1, &mut y.vec, 1);
        }
    }
}

impl Axpy<Complex<f64>> for Vecz {
    fn axpy(a: Complex<f64>, x: &Self, y: &mut Self) -> () {
        unsafe {
            blas::zaxpy(x.vec.len() as i32, a, &x.vec, 1, &mut y.vec, 1);
        }
    }
}

#[pyfunction]
#[pyo3(name = "axpy")]
pub fn py_axpy(a: &PyAny, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if let Ok(a) = a.extract::<f32>() {
            if let Ok(x) = x.extract::<PyRef<Vecf>>() {
                if let Ok(mut y) = y.extract::<PyRefMut<Vecf>>() {
                    return Ok((Vecf::axpy(a, &x, &mut y)).into_py(py));
                }
            }
        }

        if let Ok(a) = a.extract::<f64>() {
            if let Ok(x) = x.extract::<PyRef<Vecd>>() {
                if let Ok(mut y) = y.extract::<PyRefMut<Vecd>>() {
                    return Ok((Vecd::axpy(a, &x, &mut y)).into_py(py));
                }
            }
        }

        if let Ok(a) = a.extract::<Complex<f32>>() {
            if let Ok(x) = x.extract::<PyRef<Vecx>>() {
                if let Ok(mut y) = y.extract::<PyRefMut<Vecx>>() {
                    return Ok((Vecx::axpy(a, &x, &mut y)).into_py(py));
                }
            }
        }

        if let Ok(a) = a.extract::<Complex<f64>>() {
            if let Ok(x) = x.extract::<PyRef<Vecz>>() {
                if let Ok(mut y) = y.extract::<PyRefMut<Vecz>>() {
                    return Ok((Vecz::axpy(a, &x, &mut y)).into_py(py));
                }
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported argument types for axpy",
        ))
    })
}
