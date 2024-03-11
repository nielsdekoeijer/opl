use crate::types::*;
use num_complex::Complex;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::PyResult;

pub trait Dot<T> {
    fn dot(x: &Self, y: &Self) -> T
    where
        Self: Sized;
}

impl Dot<f32> for Vecf {
    fn dot(x: &Self, y: &Self) -> f32 {
        unsafe { blas::sdot(x.vec.len() as i32, &x.vec, 1, &y.vec, 1) }
    }
}

impl Dot<f64> for Vecd {
    fn dot(x: &Self, y: &Self) -> f64 {
        unsafe { blas::ddot(x.vec.len() as i32, &x.vec, 1, &y.vec, 1) }
    }
}

impl Dot<Complex<f32>> for Vecx {
    fn dot(x: &Self, y: &Self) -> Complex<f32> {
        let mut res = [Complex::<f32>::new(0.0, 0.0)];
        unsafe {
            blas::cdotu(&mut res, x.vec.len() as i32, &x.vec, 1, &y.vec, 1);
        }
        return res[0];
    }
}

impl Dot<Complex<f64>> for Vecz {
    fn dot(x: &Self, y: &Self) -> Complex<f64> {
        let mut res = [Complex::<f64>::new(0.0, 0.0)];
        unsafe {
            blas::zdotu(&mut res, x.vec.len() as i32, &x.vec, 1, &y.vec, 1);
        }
        println!("{}", res[0]);
        return res[0];
    }
}

#[pyfunction]
#[pyo3(name = "dot")]
pub fn py_dot(x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
    Python::with_gil(|py| {
        if let Ok(x) = x.extract::<PyRef<Vecf>>() {
            if let Ok(mut y) = y.extract::<PyRef<Vecf>>() {
                return Ok((Vecf::dot(&x, &mut y)).into_py(py));
            }
        }

        if let Ok(x) = x.extract::<PyRef<Vecd>>() {
            if let Ok(mut y) = y.extract::<PyRef<Vecd>>() {
                return Ok((Vecd::dot(&x, &mut y)).into_py(py));
            }
        }

        if let Ok(x) = x.extract::<PyRef<Vecx>>() {
            if let Ok(mut y) = y.extract::<PyRef<Vecx>>() {
                return Ok((Vecx::dot(&x, &mut y)).into_py(py));
            }
        }

        if let Ok(x) = x.extract::<PyRef<Vecz>>() {
            if let Ok(mut y) = y.extract::<PyRef<Vecz>>() {
                return Ok((Vecz::dot(&x, &mut y)).into_py(py));
            }
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported argument types for dot",
        ))
    })
}
