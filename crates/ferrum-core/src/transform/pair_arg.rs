//! The one reader of an ordered-pair constructor argument (`Kde(extent=)`,
//! `Bin2D(extent_x=)`, `Robust(x_range=)`, `ReferenceLine(x=)` and the like).
//!
//! It accepts exactly what the Python reader `ferrum._validate.coerce_range_pair`
//! accepts (a list, tuple, `range` or 1-D numpy array holding real numbers) and
//! refuses everything else without reading it, in the same words: a set would
//! yield its ends in hash order, a generator is consumed by reading it, and a
//! string or bytes object yields characters or integers. Three flavours differ
//! only in what they require of the two ends: a statistic range needs
//! `lo < hi` and both finite, a limit or endpoint pair needs both finite and
//! keeps a reversed order.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyTuple};

/// One end of a pair: the object as the caller wrote it and its float value
/// (`None` when a real number cannot form a float, such as `10**400`).
struct End<'py> {
    obj: Bound<'py, PyAny>,
    value: Option<f64>,
}

impl End<'_> {
    fn is_finite(&self) -> bool {
        self.value.is_some_and(f64::is_finite)
    }

    /// The end as a refusal shows it: a whole number without a fraction,
    /// anything else as Python's float repr (`nan`, `-inf`, `0.5`), and an
    /// end that forms no float as its own repr.
    fn text(&self) -> PyResult<String> {
        let py = self.obj.py();
        match self.value {
            None => Ok(self.obj.repr()?.to_string()),
            Some(v) if v.is_finite() && v.fract() == 0.0 => {
                Ok(PyFloat::new(py, v).call_method0("__int__")?.str()?.to_string())
            }
            Some(v) => Ok(PyFloat::new(py, v).repr()?.to_string()),
        }
    }
}

fn is_builtin_instance(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<bool> {
    let ty = obj.py().import("builtins")?.getattr(name)?;
    obj.is_instance(&ty)
}

/// Whether `obj` is a real number the way `ferrum._validate.is_real_number`
/// reads it: not a `bool`, not `numpy.timedelta64`, otherwise an `int`,
/// `float`, `Decimal` or any `numbers.Real`.
fn is_real_number(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    if obj.is_instance_of::<PyBool>() {
        return Ok(false);
    }
    if obj.is_instance_of::<PyInt>() || obj.is_instance_of::<PyFloat>() {
        return Ok(true);
    }
    let py = obj.py();
    if obj.is_instance(&py.import("decimal")?.getattr("Decimal")?)? {
        return Ok(true);
    }
    if !obj.is_instance(&py.import("numbers")?.getattr("Real")?)? {
        return Ok(false);
    }
    Ok(obj.get_type().name()? != "timedelta64")
}

/// Whether `obj` is a 1-D numpy array. numpy is imported only for an object
/// that is none of the built-in ordered kinds, and an absent numpy means no
/// array can have been passed.
fn is_one_dimensional_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let Ok(numpy) = obj.py().import("numpy") else {
        return Ok(false);
    };
    if !obj.is_instance(&numpy.getattr("ndarray")?)? {
        return Ok(false);
    }
    Ok(obj.getattr("ndim")?.extract::<usize>()? == 1)
}

fn is_ordered_kind(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(obj.is_instance_of::<PyList>()
        || obj.is_instance_of::<PyTuple>()
        || is_builtin_instance(obj, "range")?
        || is_one_dimensional_array(obj)?)
}

/// The shape refusal, in the Python reader's words. `expected` completes
/// `<voice>: <param> must be ` (the sentence for a pair, or for the two pairs
/// of `Kde2D(extent=)`).
fn shape_refusal(voice: &str, param: &str, expected: &str, obj: &Bound<'_, PyAny>) -> PyErr {
    let described = (|| -> PyResult<String> {
        let mut text = format!(
            "{voice}: {param} must be {expected} as a list or tuple, got {} {}",
            obj.get_type().name()?,
            obj.repr()?
        );
        if is_builtin_instance(obj, "set")? || is_builtin_instance(obj, "frozenset")? {
            text.push_str("; a set has no order, so the ends would change from run to run");
        } else if is_builtin_instance(obj, "bytes")?
            || is_builtin_instance(obj, "bytearray")?
            || is_builtin_instance(obj, "memoryview")?
        {
            text.push_str("; iterating it yields integers");
        }
        Ok(text)
    })();
    match described {
        Ok(text) => PyValueError::new_err(text),
        Err(err) => err,
    }
}

pub(crate) const PAIR: &str = "a (lo, hi) pair of numbers";
/// A diverging domain: two ends, or the two ends with a midpoint between.
pub(crate) const PAIR_OR_TRIPLE: &str = "a (lo, hi) pair or (lo, mid, hi) triple of numbers";
const TWO_PAIRS: &str = "two (lo, hi) pairs of numbers, (xmin, xmax, ymin, ymax),";

/// The `count` real-number ends of `obj`, or the shape refusal. Only a list,
/// tuple, `range` or 1-D numpy array of exactly `count` items is read, and its
/// length is checked before any item is.
fn read_ends<'py>(
    voice: &str,
    param: &str,
    expected: &str,
    count: usize,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Vec<End<'py>>> {
    let refuse = || shape_refusal(voice, param, expected, obj);
    if !is_ordered_kind(obj)? || obj.len().ok() != Some(count) {
        return Err(refuse());
    }
    let mut ends = Vec::with_capacity(count);
    for i in 0..count {
        let item = obj.get_item(i).map_err(|_| refuse())?;
        if !is_real_number(&item)? {
            return Err(refuse());
        }
        let value = item.extract::<f64>().ok();
        ends.push(End { obj: item, value });
    }
    Ok(ends)
}

fn range_refusal(voice: &str, param: &str, lo: &End<'_>, hi: &End<'_>) -> PyResult<PyErr> {
    Ok(PyValueError::new_err(format!(
        "{voice}: {param} must be (lo, hi) with lo < hi and both finite; got ({}, {})",
        lo.text()?,
        hi.text()?
    )))
}

fn checked_range(voice: &str, param: &str, lo: &End<'_>, hi: &End<'_>) -> PyResult<(f64, f64)> {
    match (lo.value, hi.value) {
        (Some(a), Some(b)) if lo.is_finite() && hi.is_finite() && a < b => Ok((a, b)),
        _ => Err(range_refusal(voice, param, lo, hi)?),
    }
}

/// `obj` as a statistic range: two finite real numbers with `lo < hi`.
/// `voice` is the public constructor and `param` the parameter the refusals
/// speak as.
pub(crate) fn extract_range(voice: &str, param: &str, obj: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    let ends = read_ends(voice, param, PAIR, 2, obj)?;
    checked_range(voice, param, &ends[0], &ends[1])
}

/// The items of a scale constructor's `domain=` or `range=`, whatever their
/// count: only a list, tuple, `range` or 1-D numpy array is read (so a set,
/// bytes, string, mapping or one-shot iterator is refused unread, in the pair
/// words), and the caller's own length and value checks run on what comes back.
/// A reversed order is kept.
pub(crate) fn read_scale_items<'py>(
    voice: &str,
    param: &str,
    expected: &str,
    obj: &Bound<'py, PyAny>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let refuse = || shape_refusal(voice, param, expected, obj);
    if !is_ordered_kind(obj)? {
        return Err(refuse());
    }
    let count = obj.len().map_err(|_| refuse())?;
    (0..count).map(|i| obj.get_item(i)).collect()
}

/// [`read_scale_items`] with every item read as a float.
pub(crate) fn extract_scale_values(
    voice: &str,
    param: &str,
    expected: &str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Vec<f64>> {
    read_scale_items(voice, param, expected, obj)?.iter().map(|item| item.extract::<f64>()).collect()
}

/// A statistic range read off the wire (a spec field no constructor has
/// checked): `None` and a finite `lo < hi` pass, anything else is refused in
/// the constructor's sentence, in the refusing transform's own `voice`, the
/// slot it read as `param`, and each end spelled as a caller would write it.
pub(crate) fn refuse_malformed_range(voice: &str, param: &str, range: Option<(f64, f64)>) -> PyResult<()> {
    match range {
        Some((lo, hi)) if !(lo.is_finite() && hi.is_finite() && lo < hi) => Err(malformed_range(voice, param, lo, hi)),
        _ => Ok(()),
    }
}

/// The refusal [`refuse_malformed_range`] raises for the ends `lo`, `hi`.
pub(crate) fn malformed_range(voice: &str, param: &str, lo: f64, hi: f64) -> PyErr {
    Python::attach(|py| {
        let end = |v: f64| End { obj: PyFloat::new(py, v).into_any(), value: Some(v) };
        range_refusal(voice, param, &end(lo), &end(hi)).unwrap_or_else(|err| err)
    })
}

/// `obj` as the four values of two statistic ranges, `(xmin, xmax, ymin,
/// ymax)`. A refused pair is named in the refusal (`<param> x pair`,
/// `<param> y pair`).
pub(crate) fn extract_range_quad(
    voice: &str,
    param: &str,
    obj: &Bound<'_, PyAny>,
) -> PyResult<(f64, f64, f64, f64)> {
    let ends = read_ends(voice, param, TWO_PAIRS, 4, obj)?;
    let (x_lo, x_hi) = checked_range(voice, &format!("{param} x pair"), &ends[0], &ends[1])?;
    let (y_lo, y_hi) = checked_range(voice, &format!("{param} y pair"), &ends[2], &ends[3])?;
    Ok((x_lo, x_hi, y_lo, y_hi))
}

/// `obj` as a pair of endpoints or limits: two finite real numbers in either
/// order (a reversed pair is kept).
pub(crate) fn extract_endpoints(voice: &str, param: &str, obj: &Bound<'_, PyAny>) -> PyResult<(f64, f64)> {
    let ends = read_ends(voice, param, PAIR, 2, obj)?;
    match (ends[0].value, ends[1].value) {
        (Some(a), Some(b)) if ends[0].is_finite() && ends[1].is_finite() => Ok((a, b)),
        _ => Err(PyValueError::new_err(format!(
            "{voice}: {param} must be a (lo, hi) pair of finite numbers; got ({}, {})",
            ends[0].text()?,
            ends[1].text()?
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::ffi::c_str;

    const SET_NOTE: &str = "; a set has no order, so the ends would change from run to run";
    const BYTES_NOTE: &str = "; iterating it yields integers";
    const RANGE_PREFIX: &str = "Kde: extent must be (lo, hi) with lo < hi and both finite; got ";

    /// Run `body` with the interpreter started (the crate has no auto-initialize,
    /// so a test run alone must start it) and the test expression evaluator.
    fn with_py<R>(body: impl FnOnce(Python<'_>) -> R) -> R {
        pyo3::Python::initialize();
        Python::attach(body)
    }

    fn eval<'py>(py: Python<'py>, code: &std::ffi::CStr) -> Bound<'py, PyAny> {
        py.eval(code, None, None).expect("test expression evaluates")
    }

    fn refusal(code: &std::ffi::CStr) -> String {
        with_py(|py| {
            let obj = eval(py, code);
            extract_range("Kde", "extent", &obj).unwrap_err().value(py).to_string()
        })
    }

    fn shape_text(type_and_repr: &str, note: &str) -> String {
        format!(
            "Kde: extent must be a (lo, hi) pair of numbers as a list or tuple, got {type_and_repr}{note}"
        )
    }

    #[test]
    fn ordered_kinds_read_equal_to_the_tuple() {
        for code in [
            c_str!("(1, 9)"),
            c_str!("[1, 9]"),
            c_str!("range(1, 10, 8)"),
            c_str!("[1.0, 9.0]"),
            c_str!("(__import__('decimal').Decimal(1), __import__('decimal').Decimal(9))"),
            c_str!("(__import__('fractions').Fraction(1), __import__('fractions').Fraction(9))"),
        ] {
            let read = with_py(|py| extract_range("Kde", "extent", &eval(py, code)).ok());
            assert_eq!(read, Some((1.0, 9.0)), "{code:?}");
        }
    }

    #[test]
    fn unordered_and_byte_kinds_are_refused_in_the_python_words() {
        for (code, expected) in [
            (c_str!("{9, 1}"), shape_text("set {9, 1}", SET_NOTE)),
            (c_str!("frozenset({9, 1})"), shape_text("frozenset frozenset({9, 1})", SET_NOTE)),
            (c_str!("b'\\x01\\x04'"), shape_text("bytes b'\\x01\\x04'", BYTES_NOTE)),
            (c_str!("bytearray(b'\\x01\\x04')"), shape_text("bytearray bytearray(b'\\x01\\x04')", BYTES_NOTE)),
            (c_str!("{1: 2}"), shape_text("dict {1: 2}", "")),
            (c_str!("{1: 2}.keys()"), shape_text("dict_keys dict_keys([1])", "")),
            (c_str!("{1: 2}.values()"), shape_text("dict_values dict_values([2])", "")),
            (c_str!("{1: 2}.items()"), shape_text("dict_items dict_items([(1, 2)])", "")),
            (c_str!("'19'"), shape_text("str '19'", "")),
            (c_str!("(1, 2, 3)"), shape_text("tuple (1, 2, 3)", "")),
            (c_str!("(1,)"), shape_text("tuple (1,)", "")),
            (c_str!("('a', 'b')"), shape_text("tuple ('a', 'b')", "")),
            (c_str!("(True, 9)"), shape_text("tuple (True, 9)", "")),
            (c_str!("(None, 9)"), shape_text("tuple (None, 9)", "")),
        ] {
            assert_eq!(refusal(code), expected, "{code:?}");
        }
    }

    #[test]
    fn memoryview_is_refused_with_the_byte_note() {
        with_py(|py| {
            let view = eval(py, c_str!("memoryview(b'\\x01\\x04')"));
            let expected = shape_text(&format!("memoryview {}", view.repr().unwrap()), BYTES_NOTE);
            let err = extract_range("Kde", "extent", &view).unwrap_err().value(py).to_string();
            assert_eq!(err, expected);
        });
    }

    #[test]
    fn an_iterator_and_a_generator_are_refused_without_being_consumed() {
        with_py(|py| {
            for code in [c_str!("iter([1, 9])"), c_str!("(i for i in (1, 9))")] {
                let obj = eval(py, code);
                let expected =
                    shape_text(&format!("{} {}", obj.get_type().name().unwrap(), obj.repr().unwrap()), "");
                let err = extract_range("Kde", "extent", &obj).unwrap_err().value(py).to_string();
                assert_eq!(err, expected, "{code:?}");
                let first = obj.call_method0("__next__").unwrap().extract::<i64>().unwrap();
                assert_eq!(first, 1, "the refusal must not advance {code:?}");
            }
        });
    }

    #[test]
    fn a_range_refuses_reversed_equal_and_non_finite_ends_with_the_python_number_text() {
        for (code, shown) in [
            (c_str!("(9, 1)"), "(9, 1)"),
            (c_str!("(1, 1)"), "(1, 1)"),
            (c_str!("(1, float('nan'))"), "(1, nan)"),
            (c_str!("(float('-inf'), 1)"), "(-inf, 1)"),
            (c_str!("(0.5, 0.25)"), "(0.5, 0.25)"),
            (c_str!("(1e-07, 1e-08)"), "(1e-07, 1e-08)"),
            (c_str!("(__import__('decimal').Decimal(9), __import__('decimal').Decimal(1))"), "(9, 1)"),
            (c_str!("(__import__('fractions').Fraction(9), __import__('fractions').Fraction(1))"), "(9, 1)"),
        ] {
            assert_eq!(refusal(code), format!("{RANGE_PREFIX}{shown}"), "{code:?}");
        }
    }

    #[test]
    fn an_end_beyond_a_float_shows_as_its_own_repr() {
        let huge = format!("1{}", "0".repeat(400));
        assert_eq!(refusal(c_str!("(1, 10**400)")), format!("{RANGE_PREFIX}(1, {huge})"));
    }

    #[test]
    fn a_decimal_that_forms_no_float_is_a_range_refusal() {
        assert_eq!(
            refusal(c_str!("(1, __import__('decimal').Decimal('sNaN'))")),
            format!("{RANGE_PREFIX}(1, Decimal('sNaN'))")
        );
    }

    #[test]
    fn endpoints_keep_a_reversed_pair_and_refuse_a_non_finite_end() {
        with_py(|py| {
            let reversed = eval(py, c_str!("(1, 0)"));
            assert_eq!(extract_endpoints("ReferenceLine", "x", &reversed).ok(), Some((1.0, 0.0)));
            let nan = eval(py, c_str!("(0, float('nan'))"));
            let err = extract_endpoints("ReferenceLine", "y", &nan).unwrap_err().value(py).to_string();
            assert_eq!(err, "ReferenceLine: y must be a (lo, hi) pair of finite numbers; got (0, nan)");
        });
    }

    #[test]
    fn scale_values_read_ordered_kinds_and_refuse_the_rest_in_the_pair_words() {
        with_py(|py| {
            for code in [c_str!("(9, 1)"), c_str!("[9, 1]"), c_str!("range(9, 0, -8)")] {
                let read = extract_scale_values("LinearScale", "domain", PAIR, &eval(py, code)).ok();
                assert_eq!(read, Some(vec![9.0, 1.0]), "{code:?}");
            }
            let three = eval(py, c_str!("[1, 2, 3]"));
            assert_eq!(extract_scale_values("LinearScale", "domain", PAIR, &three).ok(), Some(vec![1.0, 2.0, 3.0]));
            for (code, expected) in [
                (c_str!("{9, 1}"), "set {9, 1}; a set has no order, so the ends would change from run to run"),
                (c_str!("b'\\x01\\x04'"), "bytes b'\\x01\\x04'; iterating it yields integers"),
                (c_str!("bytearray(b'\\x01')"), "bytearray bytearray(b'\\x01'); iterating it yields integers"),
                (c_str!("'19'"), "str '19'"),
                (c_str!("{1: 2}"), "dict {1: 2}"),
            ] {
                let err = extract_scale_values("LinearScale", "range", PAIR, &eval(py, code))
                    .unwrap_err()
                    .value(py)
                    .to_string();
                assert_eq!(
                    err,
                    format!("LinearScale: range must be a (lo, hi) pair of numbers as a list or tuple, got {expected}")
                );
            }
            let set = eval(py, c_str!("{1, 2, 3}"));
            let err = extract_scale_values("DivergingScale", "domain", PAIR_OR_TRIPLE, &set)
                .unwrap_err()
                .value(py)
                .to_string();
            assert_eq!(
                err,
                "DivergingScale: domain must be a (lo, hi) pair or (lo, mid, hi) triple of numbers as a list or \
                 tuple, got set {1, 2, 3}; a set has no order, so the ends would change from run to run"
            );
            let gen_ = eval(py, c_str!("iter([1, 9])"));
            assert!(extract_scale_values("LinearScale", "domain", PAIR, &gen_).is_err());
            assert_eq!(gen_.call_method0("__next__").unwrap().extract::<i64>().unwrap(), 1);
        });
    }

    #[test]
    fn a_wire_range_refusal_spells_its_ends_as_the_constructor_does() {
        with_py(|py| {
            for (range, shown) in [
                ((9.0, 1.0), "(9, 1)"),
                ((3.0, 3.0), "(3, 3)"),
                ((f64::NAN, 9.0), "(nan, 9)"),
                ((1.0, f64::INFINITY), "(1, inf)"),
                ((1e-7, 1e-8), "(1e-07, 1e-08)"),
                ((0.5, 0.25), "(0.5, 0.25)"),
            ] {
                let err = refuse_malformed_range("stat_bin", "extent", Some(range)).unwrap_err();
                assert_eq!(
                    err.value(py).to_string(),
                    format!("stat_bin: extent must be (lo, hi) with lo < hi and both finite; got {shown}")
                );
            }
            assert!(refuse_malformed_range("stat_bin", "extent", None).is_ok());
            assert!(refuse_malformed_range("stat_bin", "extent", Some((1.0, 2.0))).is_ok());
        });
    }

    #[test]
    fn a_quad_reads_four_values_and_names_the_refused_pair() {
        with_py(|py| {
            let ok = eval(py, c_str!("[0, 1, 2, 3]"));
            assert_eq!(extract_range_quad("Kde2D", "extent", &ok).ok(), Some((0.0, 1.0, 2.0, 3.0)));
            let reversed_x = eval(py, c_str!("(5, 1, 0, 1)"));
            let err = extract_range_quad("Kde2D", "extent", &reversed_x).unwrap_err().value(py).to_string();
            assert_eq!(err, "Kde2D: extent x pair must be (lo, hi) with lo < hi and both finite; got (5, 1)");
            let reversed_y = eval(py, c_str!("(0, 1, 3, 2)"));
            let err = extract_range_quad("Kde2D", "extent", &reversed_y).unwrap_err().value(py).to_string();
            assert_eq!(err, "Kde2D: extent y pair must be (lo, hi) with lo < hi and both finite; got (3, 2)");
            let set = eval(py, c_str!("{0, 1, 2, 3}"));
            let err = extract_range_quad("Kde2D", "extent", &set).unwrap_err().value(py).to_string();
            assert_eq!(
                err,
                format!(
                    "Kde2D: extent must be two (lo, hi) pairs of numbers, (xmin, xmax, ymin, ymax), as a list or tuple, got set {{0, 1, 2, 3}}{SET_NOTE}"
                )
            );
            let pair = eval(py, c_str!("(0, 1)"));
            let err = extract_range_quad("Kde2D", "extent", &pair).unwrap_err().value(py).to_string();
            assert_eq!(
                err,
                "Kde2D: extent must be two (lo, hi) pairs of numbers, (xmin, xmax, ymin, ymax), as a list or tuple, got tuple (0, 1)"
            );
        });
    }

    // The 1-D ndarray accepted and 2-D ndarray refused kinds, numpy scalar ends
    // and `numpy.timedelta64` ends need numpy, which the Python cargo links
    // against does not carry. They are pinned from pytest on every compiled
    // door in tests/test_ordered_pair_params.py (`ndarray_int`, `ndarray_float`,
    // `ndarray_float32`, `ndarray_object`, `numpy_scalars` in `_ACCEPTED`;
    // `array_2d`, `array_text`, `timedelta_ends`, `numpy_bool_end` in
    // `_REFUSED_KINDS`).
}
