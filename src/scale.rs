use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::PyResult;

use num_complex::Complex;

use crate::types::*;

pub trait Scale<T> {
    fn scale(x: T, y: &mut Self) -> ()
    where
        Self: Sized;
}

impl Scale<f32> for Vecf {
    fn scale(a: f32, y: &mut Self) -> () {
        unsafe {
            blas::sscal(y.vec.len() as i32, a, &mut y.vec, 1);
        }
    }
}

impl Scale<f64> for Vecd {
    fn scale(a: f64, y: &mut Self) -> () {
        unsafe {
            blas::dscal(y.vec.len() as i32, a, &mut y.vec, 1);
        }
    }
}

impl Scale<Complex<f32>> for Vecx {
    fn scale(a: Complex<f32>, y: &mut Self) -> () {
        unsafe {
            blas::cscal(y.vec.len() as i32, a, &mut y.vec, 1);
        }
    }
}

impl Scale<Complex<f64>> for Vecz {
    fn scale(a: Complex<f64>, y: &mut Self) -> () {
        unsafe {
            blas::zscal(y.vec.len() as i32, a, &mut y.vec, 1);
        }
    }
}

#[pyfunction]
#[pyo3(name = "scale")]
pub fn py_scale(a: &PyAny, y: &PyAny) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if let Ok(a) = a.extract::<f32>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecf>>() {
                return Ok((Vecf::scale(a, &mut y)).into_py(py));
            }
        }

        if let Ok(a) = a.extract::<f64>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecd>>() {
                return Ok((Vecd::scale(a, &mut y)).into_py(py));
            }
        }

        if let Ok(a) = a.extract::<Complex<f32>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecx>>() {
                return Ok((Vecx::scale(a, &mut y)).into_py(py));
            }
        }

        if let Ok(a) = a.extract::<Complex<f64>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecz>>() {
                return Ok((Vecz::scale(a, &mut y)).into_py(py));
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported argument types for scale",
        ))
    })
}
