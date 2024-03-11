use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::PyResult;

use num_complex::Complex;

use crate::types::*;

pub trait Swap<T> {
    fn swap(x: &mut Self, y: &mut Self) -> ()
    where
        Self: Sized;
}

impl Swap<f32> for Vecf {
    fn swap(x: &mut Self, y: &mut Self) -> () {
        unsafe { blas::sswap(x.vec.len() as i32, &mut x.vec, 1, &mut y.vec, 1) }
    }
}

impl Swap<f64> for Vecd {
    fn swap(x: &mut Self, y: &mut Self) -> () {
        unsafe { blas::dswap(x.vec.len() as i32, &mut x.vec, 1, &mut y.vec, 1) }
    }
}

impl Swap<Complex<f32>> for Vecx {
    fn swap(x: &mut Self, y: &mut Self) -> () {
        unsafe {
            blas::cswap(x.vec.len() as i32, &mut x.vec, 1, &mut y.vec, 1);
        }
    }
}

impl Swap<Complex<f64>> for Vecz {
    fn swap(x: &mut Self, y: &mut Self) -> () {
        unsafe {
            blas::zswap(x.vec.len() as i32, &mut x.vec, 1, &mut y.vec, 1);
        }
    }
}

#[pyfunction]
#[pyo3(name = "swap")]
pub fn py_swap(x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if let Ok(mut x) = x.extract::<PyRefMut<Vecf>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecf>>() {
                return Ok((Vecf::swap(&mut x, &mut y)).into_py(py));
            }
        }

        if let Ok(mut x) = x.extract::<PyRefMut<Vecd>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecd>>() {
                return Ok((Vecd::swap(&mut x, &mut y)).into_py(py));
            }
        }

        if let Ok(mut x) = x.extract::<PyRefMut<Vecx>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecx>>() {
                return Ok((Vecx::swap(&mut x, &mut y)).into_py(py));
            }
        }

        if let Ok(mut x) = x.extract::<PyRefMut<Vecz>>() {
            if let Ok(mut y) = y.extract::<PyRefMut<Vecz>>() {
                return Ok((Vecz::swap(&mut x, &mut y)).into_py(py));
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported argument types for swap",
        ))
    })
}
