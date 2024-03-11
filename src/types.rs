use num_complex::Complex;
use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::PyResult;

pub trait FromFloat<T> {
    fn from_float(value: f64) -> T;
}

impl FromFloat<f32> for f32 {
    fn from_float(value: f64) -> f32 {
        value as f32
    }
}

impl FromFloat<f64> for f64 {
    fn from_float(value: f64) -> f64 {
        value as f64
    }
}

impl FromFloat<Complex<f32>> for Complex<f32> {
    fn from_float(value: f64) -> Complex<f32> {
        Complex::<f32>::new(value as f32, 0.0)
    }
}

impl FromFloat<Complex<f64>> for Complex<f64> {
    fn from_float(value: f64) -> Complex<f64> {
        Complex::<f64>::new(value as f64, 0.0)
    }
}

macro_rules! add_vec_pymethods {
    ($struct_name:ident, $s:literal, $t:ty) => {
        #[pyclass]
        #[pyo3(name = $s)]
        #[derive(Clone)]
        pub struct $struct_name {
            pub vec: Vec<$t>,
        }

        impl std::fmt::Display for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{:?}", self.vec)
            }
        }

        #[pymethods]
        impl $struct_name {
            #[new]
            fn new(size: usize) -> Self {
                $struct_name {
                    vec: vec![<$t as FromFloat<$t>>::from_float(0.0); size],
                }
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.to_string())
            }

            fn __getitem__(&self, idx: usize) -> PyResult<$t> {
                if idx >= self.vec.len() {
                    Err(exceptions::PyIndexError::new_err("Index out of bounds"))
                } else {
                    Ok(self.vec[idx])
                }
            }

            fn __setitem__(&mut self, idx: usize, value: $t) -> PyResult<()> {
                if idx >= self.vec.len() {
                    Err(exceptions::PyIndexError::new_err("Index out of bounds"))
                } else {
                    self.vec[idx] = value;
                    Ok(())
                }
            }
        }
    };
}

add_vec_pymethods!(Vecf, "vecf", f32);
add_vec_pymethods!(Vecd, "vecd", f64);
add_vec_pymethods!(Vecx, "vecx", Complex<f32>);
add_vec_pymethods!(Vecz, "vecz", Complex<f64>);

macro_rules! add_mat_pymethods {
    ($struct_name:ident, $s:literal, $t:ty) => {
        #[pyclass]
        #[pyo3(name = $s)]
        #[derive(Clone)]
        pub struct $struct_name {
            pub mat: Vec<$t>,
            stride: usize,
        }

        impl std::fmt::Display for $struct_name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                for i in 0..(self.stride - 1) {
                    let slice = &self.mat[(i * self.stride)..((i + 1) * self.stride)];
                    write!(f, "{:?}\n", slice)?;
                }

                Ok(())
            }
        }

        #[pymethods]
        impl $struct_name {
            #[new]
            fn new(rows: usize, cols: usize) -> Self {
                $struct_name {
                    mat: vec![<$t as FromFloat<$t>>::from_float(0.0); rows * cols],
                    stride: rows,
                }
            }

            fn __repr__(&self) -> PyResult<String> {
                Ok(self.to_string())
            }

            fn __getitem__(&self, idx: (usize, usize)) -> PyResult<$t> {
                // TODO: Maybe we can make this more efficient by having a 2D array, saving us the
                // multiplication? E.g. we make an array of "pointers".
                if (idx.1 + self.stride * idx.0) >= self.mat.len() {
                    Err(exceptions::PyIndexError::new_err("Index out of bounds"))
                } else {
                    Ok(self.mat[idx.1 + self.stride * idx.0])
                }
            }

            fn __setitem__(&mut self, idx: (usize, usize), value: $t) -> PyResult<()> {
                // TODO: Maybe we can make this more efficient by having a 2D array, saving us the
                // multiplication? E.g. we make an array of "pointers".
                if (idx.1 + self.stride * idx.0) >= self.mat.len() {
                    Err(exceptions::PyIndexError::new_err("Index out of bounds"))
                } else {
                    self.mat[idx.1 + self.stride * idx.0] = value;
                    Ok(())
                }
            }
        }
    };
}

add_mat_pymethods!(Matf, "matf", f32);
add_mat_pymethods!(Matd, "matd", f64);
add_mat_pymethods!(Matx, "matx", Complex<f32>);
add_mat_pymethods!(Matz, "matz", Complex<f64>);
