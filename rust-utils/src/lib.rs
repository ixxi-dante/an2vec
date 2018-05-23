#![feature(proc_macro, specialization, pattern)]
extern crate pyo3;
extern crate rayon;
extern crate regex;

use rayon::prelude::*;
use pyo3::prelude::*;
use regex::Regex;

use pyo3::py::modinit as pymodinit;

fn split(string: &str, sep: &Regex) -> Vec<String> {
    sep.split(string).map(|part| part.to_string()).collect::<Vec<_>>()
}

fn array_split(strings: &[String], sep: &str, parallel: bool)
        -> Result<Vec<Vec<String>>, regex::Error> {
    // This has the (important) downside that it must generate new
    // String objects for each of the split parts.
    // To get around this, we might want to convert to embeddings directly
    // in here, or return slices of the parts by pointing to the original data,
    // or investigate how to let python do that by giving it a list of slices
    // directly.
    //
    // Of interest:
    // - https://github.com/bluss/rust-ndarray/issues/137
    // - https://github.com/termoshtt/rust-numpy/issues/23

    let sep = Regex::new(sep)?;
    let splits = if parallel {
        strings.par_iter().map(|string| split(string, &sep)).collect()
    } else {
        strings.iter().map(|string| split(string, &sep)).collect()
    };

    Ok(splits)
}

#[pymodinit(_rust_utils)]
fn init_mod(py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "array_split", parallel= "true")]
    fn array_split_py(py: Python, strings: Vec<String>, sep: String, parallel: bool)
            -> PyResult<Vec<Vec<String>>> {
        let counts = py.allow_threads(|| array_split(&strings, &sep, parallel))
            .map_err(|error| exc::ValueError::new(format!("{:?}", error)))?;

        Ok(counts)
    }

    Ok(())
}
