extern crate blas;
extern crate num_complex;

use pyo3::prelude::*;
use pyo3::PyResult;

mod types;
use types::*;

mod dot;
use dot::*;
mod scale;
use scale::*;
mod swap;
use swap::*;
mod copy;
use copy::*;
mod axpy;
use axpy::*;

#[pymodule]
fn opl(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Vecf>()?;
    m.add_class::<Vecd>()?;
    m.add_class::<Vecx>()?;
    m.add_class::<Vecz>()?;
    m.add_class::<Matf>()?;
    m.add_class::<Matd>()?;
    m.add_class::<Matx>()?;
    m.add_class::<Matz>()?;
    m.add_function(wrap_pyfunction!(py_dot, m)?)?;
    m.add_function(wrap_pyfunction!(py_scale, m)?)?;
    m.add_function(wrap_pyfunction!(py_swap, m)?)?;
    m.add_function(wrap_pyfunction!(py_copy, m)?)?;
    m.add_function(wrap_pyfunction!(py_axpy, m)?)?;
    Ok(())
}
