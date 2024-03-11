use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::PyResult;

use num_complex::Complex;

use crate::types::*;

pub trait Copy<T> {
    fn copy(x: &Self, y: &mut Self) -> ()
    where
        Self: Sized;
}

impl Copy<f32> for Vecf {
    fn copy(x: &Self, y: &mut Self) -> () {
        unsafe { blas::scopy(x.vec.len() as i32, &x.vec, 1, &mut y.vec, 1) }
    }
}

impl Copy<f64> for Vecd {
    fn copy(x: &Self, y: &mut Self) -> () {
        unsafe { blas::dcopy(x.vec.len() as i32, &x.vec, 1, &mut y.vec, 1) }
    }
}

impl Copy<Complex<f32>> for Vecx {
    fn copy(x: &Self, y: &mut Self) -> () {
        unsafe {
            blas::ccopy(x.vec.len() as i32, &x.vec, 1, &mut y.vec, 1);
        }
    }
}

impl Copy<Complex<f64>> for Vecz {
    fn copy(x: &Self, y: &mut Self) -> () {
        unsafe {
            blas::zcopy(x.vec.len() as i32, &x.vec, 1, &mut y.vec, 1);
        }
    }
}

#[pyfunction]
#[pyo3(name = "copy")]
pub fn py_copy(x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if let Ok(x) = x.extract::<PyRef<Vecf>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecf>>() {
                return Ok((Vecf::copy(&x, &mut y)).into_py(py));
            }
        }

        if let Ok(x) = x.extract::<PyRef<Vecd>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecd>>() {
                return Ok((Vecd::copy(&x, &mut y)).into_py(py));
            }
        }

        if let Ok(x) = x.extract::<PyRef<Vecx>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecx>>() {
                return Ok((Vecx::copy(&x, &mut y)).into_py(py));
            }
        }

        if let Ok(x) = x.extract::<PyRef<Vecz>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecz>>() {
                return Ok((Vecz::copy(&x, &mut y)).into_py(py));
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported argument types for copy",
        ))
    })
}
