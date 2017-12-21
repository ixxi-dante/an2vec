#![feature(proc_macro, specialization, pattern)]
extern crate pyo3;
extern crate rayon;
extern crate regex;

use rayon::prelude::*;
use pyo3::prelude::*;
use regex::Regex;

fn split(string: &str, sep: &Regex) -> Vec<String> {
    sep.split(string).map(|part| part.to_string()).collect::<Vec<_>>()
}

fn array_split(strings: &[String], sep: &str, parallel: bool)
        -> Result<Vec<Vec<String>>, regex::Error> {
    let sep = Regex::new(sep)?;
    let splits = if parallel {
        strings.par_iter().map(|string| split(string, &sep)).collect()
    } else {
        strings.iter().map(|string| split(string, &sep)).collect()
    };

    Ok(splits)
}

#[py::modinit(_rust_utils)]
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
