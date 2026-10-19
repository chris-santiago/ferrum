//! `DataType`, `EncodingSpec`, and `Encoding` test coverage — everything
//! from the original inline `#[cfg(test)] mod tests` at the bottom of
//! `spec/encoding.rs` EXCEPT the scale wire-key gate cluster, which stays
//! in `gate.rs`'s own inline test module (see that file's doc comment for
//! why). Wave 6A `#158`: relocated wholesale when `mod.rs` split into
//! `mod.rs` + `gate.rs`, mirroring `projection/tests.rs` and
//! `transform/stats/tests.rs`. Pure motion: no assertion, fixture, or test
//! name changed.

use super::*;

#[test]
fn test_linear_scale_domain_param_round_trips() {
    // `domainParam` deserializes from the compact wire form Python emits...
    let scale: ScaleSpec =
        serde_json::from_str(r#"{"type":"linear","domainParam":"d"}"#).unwrap();
    assert_eq!(scale.domain_param(), Some("d"));
    // ...and survives a serialize → deserialize round-trip.
    let back = serde_json::to_string(&scale).unwrap();
    assert!(back.contains(r#""domainParam":"d""#), "missing domainParam: {back}");
    let reparsed: ScaleSpec = serde_json::from_str(&back).unwrap();
    assert_eq!(reparsed, scale);
}

#[test]
fn test_scale_set_domain_clears_domain_param() {
    let mut scale: ScaleSpec =
        serde_json::from_str(r#"{"type":"linear","domainParam":"d"}"#).unwrap();
    scale.set_domain(vec![10.0, 20.0]);
    assert_eq!(scale.domain_param(), None);
    match scale {
        ScaleSpec::Linear { common, .. } => assert_eq!(common.domain, Some(vec![10.0, 20.0])),
        _ => panic!("expected Linear"),
    }
}

#[test]
fn test_data_type_short_and_long_forms() {
    assert_eq!(DataType::from_str("Q").unwrap(), DataType::Quantitative);
    assert_eq!(DataType::from_str("quantitative").unwrap(), DataType::Quantitative);
    assert_eq!(DataType::from_str("N").unwrap(), DataType::Nominal);
    assert_eq!(DataType::from_str("nominal").unwrap(), DataType::Nominal);
}

#[test]
fn test_data_type_unknown() {
    let err = DataType::from_str("Z").unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("'Z'"), "msg: {msg}");
    assert!(msg.contains("quantitative"), "msg: {msg}");
}

#[test]
fn test_data_type_serde_long_form() {
    assert_eq!(serde_json::to_string(&DataType::Quantitative).unwrap(), "\"quantitative\"");
}

#[test]
fn test_encoding_spec_round_trip_no_type() {
    let original = EncodingSpec { field: "price".into(), type_: None, ..Default::default() };
    let json = serde_json::to_string(&original).unwrap();
    assert_eq!(json, r#"{"field":"price"}"#);
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, original);
}

#[test]
fn test_encoding_spec_round_trip_with_type() {
    let original = EncodingSpec {
        field: "weight".into(),
        type_: Some(DataType::Quantitative),
        ..Default::default()
    };
    let json = serde_json::to_string(&original).unwrap();
    assert_eq!(json, r#"{"field":"weight","type":"quantitative"}"#);
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, original);
}

#[test]
fn encoding_spec_unknown_field_is_rejected() {
    // No-silent-drop seam (S4FIX2): a typo'd channel key (e.g. `typ` for
    // `type`) on the EncodingSpec itself must fail loud via
    // `deny_unknown_fields`, not be silently dropped.
    let json = r#"{"field":"a","typ":"quantitative"}"#;
    let err = serde_json::from_str::<EncodingSpec>(json).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("unknown field") && msg.contains("typ"),
        "expected an unknown-field error, got: {msg}"
    );
}

#[test]
fn encoding_spec_all_known_fields_round_trip_under_deny() {
    // Every advertised EncodingSpec field must still deserialize after the
    // deny was added — a known key must never be mistaken for an unknown one.
    let json = r##"{"field":"a","type":"quantitative","scale":{"type":"linear"},"title":"T","axis":{"grid":false},"legend":{"disabled":true},"sort":"ascending","stack":"zero","condition":{},"impute":{"value":0},"scheme":"viridis","format":".2f","formatType":"number"}"##;
    let parsed: EncodingSpec = serde_json::from_str(json).expect("known fields must deserialize");
    assert_eq!(parsed.field, "a");
    assert_eq!(parsed.type_, Some(DataType::Quantitative));
    assert!(parsed.scale.is_some());
}

#[test]
fn test_encoding_round_trip_both_axes() {
    let e = Encoding {
        x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
        y: Some(EncodingSpec {
            field: "weight".into(),
            type_: Some(DataType::Quantitative),
            ..Default::default()
        }),
        color: None,
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert_eq!(
        json,
        r#"{"x":{"field":"price"},"y":{"field":"weight","type":"quantitative"}}"#,
    );
    let parsed: Encoding = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn test_encoding_omits_none_fields() {
    let e = Encoding::default();
    let json = serde_json::to_string(&e).unwrap();
    assert_eq!(json, "{}");
}

#[test]
fn test_encoding_round_trip_with_color() {
    let e = Encoding {
        x: Some(EncodingSpec { field: "price".into(), type_: None, ..Default::default() }),
        y: Some(EncodingSpec { field: "weight".into(), type_: None, ..Default::default() }),
        color: Some(EncodingSpec {
            field: "species".into(),
            type_: Some(DataType::Nominal),
            ..Default::default()
        }),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert_eq!(
        json,
        r#"{"x":{"field":"price"},"y":{"field":"weight"},"color":{"field":"species","type":"nominal"}}"#,
    );
    let parsed: Encoding = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn test_encoding_omits_color_when_none() {
    let e = Encoding {
        x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
        y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
        color: None,
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert_eq!(json, r#"{"x":{"field":"a"},"y":{"field":"b"}}"#);
}

// --- Phase 8a new tests ---

#[test]
fn encoding_spec_round_trips_with_scale() {
    let e = EncodingSpec {
        field: "price".into(),
        type_: Some(DataType::Quantitative),
        scale: Some(ScaleSpec::Log {
            base: 10.0,
            common: ContinuousScaleCommon {
                domain: None,
                range: None,
                clamp: false,
                padding: None,
                scheme: None,
                domain_param: None,
                reverse: false,
            },
            nice: true,
        }),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert!(json.contains(r#""scale":{"type":"log""#));
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn encoding_spec_round_trips_with_title() {
    let e = EncodingSpec {
        field: "x".into(),
        type_: None,
        title: Some("My X Axis".into()),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert!(json.contains(r#""title":"My X Axis""#));
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn encoding_spec_round_trips_with_typed_axis() {
    // A per-channel axis carrying both an honored styling key (`grid` toggle)
    // and an orphan positioning key (`orient`) round-trips through the typed
    // `AxisStyleSpec`.
    use crate::render::chart_config::AxisStyleSpec;
    let e = EncodingSpec {
        field: "x".into(),
        type_: None,
        axis: Some(Box::new(AxisStyleSpec {
            grid: Some(false),
            orient: Some("bottom".into()),
            ..Default::default()
        })),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn encoding_spec_round_trips_with_typed_axis_styling_and_orphan() {
    // Honored styling key `grid_color` + orphan `orient` both survive the
    // round-trip (B5).
    use crate::render::chart_config::AxisStyleSpec;
    let e = EncodingSpec {
        field: "x".into(),
        axis: Some(Box::new(AxisStyleSpec {
            grid_color: Some("#cccccc".into()),
            orient: Some("bottom".into()),
            ..Default::default()
        })),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    let parsed: EncodingSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
    assert_eq!(parsed.axis.as_ref().unwrap().grid_color.as_deref(), Some("#cccccc"));
    assert_eq!(parsed.axis.as_ref().unwrap().orient.as_deref(), Some("bottom"));
}

#[test]
fn encoding_spec_per_channel_axis_unknown_key_fails_loud() {
    // A misspelled per-channel axis key must error (deny_unknown_fields),
    // not silently drop (B5 fail-loud).
    let json = r##"{"field":"x","axis":{"grid_colr":"#f00"}}"##;
    let parsed: Result<EncodingSpec, _> = serde_json::from_str(json);
    assert!(parsed.is_err(), "unknown per-channel axis key must fail to deserialize");
}

#[test]
fn encoding_spec_per_channel_legend_unknown_key_fails_loud() {
    let json = r##"{"field":"c","legend":{"symbol_sze":10}}"##;
    let parsed: Result<EncodingSpec, _> = serde_json::from_str(json);
    assert!(parsed.is_err(), "unknown per-channel legend key must fail to deserialize");
}

#[test]
fn encoding_spec_axis_camel_case_alias_deserializes() {
    // Raw-dict back-compat: camelCase `labelAngle` maps to `label_angle`.
    let json = r##"{"field":"x","axis":{"labelAngle":-30}}"##;
    let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.axis.as_ref().unwrap().label_angle, Some(-30.0));
}

#[test]
fn encoding_spec_axis_false_suppression_round_trips() {
    // `axis=False` serializes (Python side) to the suppression form; the typed
    // struct must accept it and round-trip.
    let json = r##"{"field":"x","axis":{"domain":false,"ticks":false,"labels":false,"title":"","grid":false}}"##;
    let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
    let axis = parsed.axis.as_ref().unwrap();
    assert_eq!(axis.domain, Some(false));
    assert_eq!(axis.ticks, Some(false));
    assert_eq!(axis.labels, Some(false));
    assert_eq!(axis.grid, Some(false));
    assert_eq!(axis.title.as_deref(), Some("")); // empty-string suppress sentinel
    // Re-serialize → re-parse stays equal.
    let back = serde_json::to_string(&parsed).unwrap();
    let reparsed: EncodingSpec = serde_json::from_str(&back).unwrap();
    assert_eq!(reparsed, parsed);
}

#[test]
fn encoding_spec_legend_disabled_suppression_round_trips() {
    // `legend=None`/`False` serializes to `{"disabled": true}`.
    let json = r##"{"field":"c","legend":{"disabled":true}}"##;
    let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.legend.as_ref().unwrap().disabled, Some(true));
}

#[test]
fn encoding_spec_phase_7_canonical_json_byte_identical_when_no_new_fields() {
    let e = EncodingSpec { field: "x".into(), type_: None, ..Default::default() };
    assert_eq!(serde_json::to_string(&e).unwrap(), r#"{"field":"x"}"#);

    let e2 = EncodingSpec {
        field: "y".into(),
        type_: Some(DataType::Quantitative),
        ..Default::default()
    };
    assert_eq!(
        serde_json::to_string(&e2).unwrap(),
        r#"{"field":"y","type":"quantitative"}"#,
    );
}

#[test]
fn encoding_spec_round_trips_pre_phase_8_json() {
    // Existing JSON without any new fields must deserialize.
    let json = r#"{"field":"price","type":"quantitative"}"#;
    let parsed: EncodingSpec = serde_json::from_str(json).unwrap();
    assert_eq!(parsed.field, "price");
    assert_eq!(parsed.type_, Some(DataType::Quantitative));
    assert_eq!(parsed.scale, None);
    assert_eq!(parsed.title, None);
}

// --- Phase 8b Task 22: x2 / y2 channels (ribbon support) ---

#[test]
fn encoding_round_trips_with_y2() {
    let e = Encoding {
        x: Some(EncodingSpec { field: "t".into(), type_: None, ..Default::default() }),
        y: Some(EncodingSpec { field: "lo".into(), type_: None, ..Default::default() }),
        y2: Some(EncodingSpec { field: "hi".into(), type_: None, ..Default::default() }),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert!(json.contains(r#""y2":{"field":"hi"}"#), "json: {json}");
    let parsed: Encoding = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed, e);
}

#[test]
fn encoding_omits_x2_y2_when_none() {
    // Existing 8a JSON without x2/y2 must remain byte-identical.
    let e = Encoding {
        x: Some(EncodingSpec { field: "a".into(), type_: None, ..Default::default() }),
        y: Some(EncodingSpec { field: "b".into(), type_: None, ..Default::default() }),
        ..Default::default()
    };
    let json = serde_json::to_string(&e).unwrap();
    assert_eq!(json, r#"{"x":{"field":"a"},"y":{"field":"b"}}"#);
    assert!(!json.contains("x2"));
    assert!(!json.contains("y2"));
}

#[test]
fn scale_spec_log_default_base_is_10() {
    let json = r#"{"type":"log"}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match parsed {
        ScaleSpec::Log { base, .. } => assert_eq!(base, 10.0),
        _ => panic!("expected Log variant"),
    }
}

#[test]
fn scale_spec_pow_defaults() {
    let json = r#"{"type":"pow"}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match parsed {
        ScaleSpec::Pow { exponent, common } => {
            assert_eq!(exponent, 2.0);
            assert!(!common.clamp);
            assert_eq!(common.padding, None);
        }
        _ => panic!("expected Pow variant"),
    }
}

#[test]
fn scale_spec_sqrt_round_trip() {
    let json = r#"{"type":"sqrt","clamp":true}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::Sqrt { common } => assert!(common.clamp),
        _ => panic!("expected Sqrt variant"),
    }
    let re = serde_json::to_string(&parsed).unwrap();
    assert!(re.contains(r#""type":"sqrt""#));
}

#[test]
fn scale_spec_utc_round_trip() {
    let json = r#"{"type":"utc","nice":true}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::Utc { nice, .. } => assert!(nice),
        _ => panic!("expected Utc variant"),
    }
}

#[test]
fn scale_spec_band_defaults() {
    let json = r#"{"type":"band"}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match parsed {
        ScaleSpec::Band { padding, align, range, .. } => {
            assert_eq!(padding, crate::scale::discrete::BandPadding::default());
            assert_eq!(align, 0.5);
            assert_eq!(range, None);
        }
        _ => panic!("expected Band variant"),
    }
}

#[test]
fn scale_spec_point_defaults() {
    let json = r#"{"type":"point"}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match parsed {
        ScaleSpec::Point { padding, align, reverse, range, .. } => {
            assert_eq!(padding, None);
            assert_eq!(align, 0.5);
            assert!(!reverse);
            assert_eq!(range, None);
        }
        _ => panic!("expected Point variant"),
    }
}

/// Issue #39: `ScaleSpec::Band { range }` round-trips through JSON when
/// present, and the wire form carries the explicit pixel range key.
#[test]
fn scale_spec_band_range_round_trip() {
    let spec = ScaleSpec::Band {
        domain: None,
        padding: crate::scale::discrete::BandPadding::new(Some(0.1), None, None)
            .expect("in-bounds band padding"),
        align: default_band_align(),
        range: Some(vec![40.0, 260.0]),
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""range":[40.0,260.0]"#), "json={json}");
    let re_parsed: ScaleSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(re_parsed, spec);
}

/// Issue #39: with `range: None`, the JSON contains no `"range"` key
/// (byte-identity guard — absent range must not perturb existing wire output).
#[test]
fn scale_spec_band_range_absent_omits_key() {
    let spec = ScaleSpec::Band {
        domain: None,
        padding: crate::scale::discrete::BandPadding::new(Some(0.1), None, None)
            .expect("in-bounds band padding"),
        align: default_band_align(),
        range: None,
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(!json.contains("\"range\""), "json={json}");
}

/// Issue #39: `ScaleSpec::Point { range }` round-trips through JSON when
/// present, and the wire form carries the explicit pixel range key.
#[test]
fn scale_spec_point_range_round_trip() {
    let spec = ScaleSpec::Point {
        domain: None,
        padding: Some(0.5),
        align: default_band_align(),
        reverse: false,
        range: Some(vec![40.0, 260.0]),
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(json.contains(r#""range":[40.0,260.0]"#), "json={json}");
    let re_parsed: ScaleSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(re_parsed, spec);
}

/// Issue #39: with `range: None`, the JSON contains no `"range"` key
/// (byte-identity guard — absent range must not perturb existing wire output).
#[test]
fn scale_spec_point_range_absent_omits_key() {
    let spec = ScaleSpec::Point {
        domain: None,
        padding: None,
        align: default_band_align(),
        reverse: false,
        range: None,
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(!json.contains("\"range\""), "json={json}");
}

#[test]
fn scale_spec_sequential_round_trip() {
    let json = r#"{"type":"sequential","scheme":"viridis","reverse":true}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::Sequential { scheme, reverse, .. } => {
            assert_eq!(scheme.as_deref(), Some("viridis"));
            assert!(reverse);
        }
        _ => panic!("expected Sequential variant"),
    }
}

/// F-L04-02 second revision (spec §4.2, amended 2026-08-28): with
/// `stops: None`, the JSON contains no `"stops"` key — byte-identity
/// guard for every pre-existing Sequential wire form (scheme-name-backed
/// specs never carry stops, so this must not perturb them).
#[test]
fn scale_spec_sequential_stops_absent_omits_key() {
    let spec = ScaleSpec::Sequential {
        scheme: Some("viridis".to_string()),
        domain: None,
        reverse: false,
        stops: None,
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(!json.contains("\"stops\""), "json={json}");
    let re_parsed: ScaleSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(re_parsed, spec);
}

/// F-L04-02 second revision (spec-reviewer cycle 3): a `Gradient`-backed
/// scheme round-trips its `(t, hex)` stop pairs through the wire form
/// byte-identically — carrying real, possibly non-uniform `t` positions,
/// not just an ordered color list.
#[test]
fn scale_spec_sequential_stops_round_trip() {
    let spec = ScaleSpec::Sequential {
        scheme: None,
        domain: None,
        reverse: false,
        stops: Some(vec![
            (0.0, "#ff0000".to_string()),
            (0.9, "#00ff00".to_string()),
            (1.0, "#0000ff".to_string()),
        ]),
    };
    let json = serde_json::to_string(&spec).unwrap();
    assert!(
        json.contains("\"stops\":[[0.0,\"#ff0000\"],[0.9,\"#00ff00\"],[1.0,\"#0000ff\"]]"),
        "json={json}"
    );
    let re_parsed: ScaleSpec = serde_json::from_str(&json).unwrap();
    assert_eq!(re_parsed, spec);
    match &re_parsed {
        ScaleSpec::Sequential { scheme, stops, .. } => {
            assert_eq!(scheme, &None);
            assert_eq!(
                stops.as_deref(),
                Some(
                    &[
                        (0.0, "#ff0000".to_string()),
                        (0.9, "#00ff00".to_string()),
                        (1.0, "#0000ff".to_string()),
                    ][..]
                )
            );
        }
        _ => panic!("expected Sequential variant"),
    }
}

#[test]
fn scale_spec_diverging_round_trip() {
    let json = r#"{"type":"diverging","scheme":"rdbu","domainMid":0.5}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::Diverging { scheme, domain_mid, .. } => {
            assert_eq!(scheme.as_deref(), Some("rdbu"));
            assert_eq!(*domain_mid, Some(0.5));
        }
        _ => panic!("expected Diverging variant"),
    }
}

#[test]
fn scale_spec_quantize_round_trip() {
    let json = r##"{"type":"quantize","domain":[0,100],"range":["#f00","#0f0","#00f"]}"##;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::Quantize { domain, range } => {
            assert_eq!(domain.as_ref().unwrap(), &vec![0.0, 100.0]);
            assert_eq!(range.as_ref().unwrap().len(), 3);
        }
        _ => panic!("expected Quantize variant"),
    }
}

#[test]
fn scale_spec_quantile_round_trip() {
    let json = r#"{"type":"quantile","domain":[0,25,50,75,100],"range":[0,1,2,3]}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::Quantile { domain, range } => {
            assert_eq!(domain.as_ref().unwrap(), &vec![0.0, 25.0, 50.0, 75.0, 100.0]);
            assert_eq!(range.as_ref().unwrap().len(), 4);
        }
        _ => panic!("expected Quantile variant"),
    }
    let re = serde_json::to_string(&parsed).unwrap();
    assert!(re.contains(r#""type":"quantile""#), "json: {re}");
}

#[test]
fn scale_spec_threshold_round_trip() {
    let json = r#"{"type":"threshold","domain":[0,50,100],"range":[0,1,2,3]}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::Threshold { domain, range } => {
            assert_eq!(domain.as_ref().unwrap(), &vec![0.0, 50.0, 100.0]);
            // range.len() == domain.len() + 1
            assert_eq!(range.as_ref().unwrap().len(), 4);
        }
        _ => panic!("expected Threshold variant"),
    }
    let re = serde_json::to_string(&parsed).unwrap();
    assert!(re.contains(r#""type":"threshold""#), "json: {re}");
}

#[test]
fn scale_spec_bin_ordinal_round_trip() {
    let json = r#"{"type":"bin-ordinal","bins":[0,10,20,30],"scheme":"blues"}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match &parsed {
        ScaleSpec::BinOrdinal { bins, scheme, .. } => {
            assert_eq!(bins.as_ref().unwrap(), &vec![0.0, 10.0, 20.0, 30.0]);
            assert_eq!(scheme.as_deref(), Some("blues"));
        }
        _ => panic!("expected BinOrdinal variant"),
    }
    // Verify serialization uses "bin-ordinal" as the type tag
    let re = serde_json::to_string(&parsed).unwrap();
    assert!(re.contains(r#""type":"bin-ordinal""#), "json: {re}");
}

/// A binned color's `bin` parameters ride the `bin-ordinal` wire and round-trip
/// byte for byte, `nice` defaulting to the `Bin` transform's `true`; a scale
/// without `bin` writes no `bin` key, so its bytes are what they were. A
/// misspelled parameter is refused rather than dropped. Rejects a `bin` the
/// wire gate refused as an unknown key, and one serialized as `"bin":null`.
#[test]
fn scale_spec_bin_ordinal_bin_round_trips_and_is_absent_when_unset() {
    let json = r#"{"type":"bin-ordinal","scheme":"blues","bin":{"bin_count":4,"extent":[0.0,10.0],"nice":false}}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).expect("a bin-ordinal scale with bin parses");
    assert_eq!(serde_json::to_string(&parsed).unwrap(), json);

    let defaulted: ScaleSpec = serde_json::from_str(r#"{"type":"bin-ordinal","bin":{"bin_width":2.5}}"#).unwrap();
    assert_eq!(serde_json::to_string(&defaulted).unwrap(), r#"{"type":"bin-ordinal","bin":{"bin_width":2.5,"nice":true}}"#);

    let explicit = r#"{"type":"bin-ordinal","bins":[0.0,10.0],"scheme":"blues"}"#;
    let parsed: ScaleSpec = serde_json::from_str(explicit).unwrap();
    assert_eq!(serde_json::to_string(&parsed).unwrap(), explicit);

    let misspelled = serde_json::from_str::<ScaleSpec>(r#"{"type":"bin-ordinal","bin":{"bins":4}}"#);
    assert!(misspelled.is_err(), "an unknown bin parameter is refused: {misspelled:?}");
}

#[test]
fn scale_spec_band_with_padding_inner_outer() {
    let json = r#"{"type":"band","paddingInner":0.2,"paddingOuter":0.1,"align":0.3}"#;
    let parsed: ScaleSpec = serde_json::from_str(json).unwrap();
    match parsed {
        ScaleSpec::Band { padding, align, .. } => {
            assert_eq!(padding.padding(), None);
            assert_eq!(padding.inner(), Some(0.2));
            assert_eq!(padding.outer(), Some(0.1));
            assert_eq!(align, 0.3);
        }
        _ => panic!("expected Band variant"),
    }
}

#[test]
fn inherit_from_propagates_title_on_same_field() {
    let parent = Encoding {
        x: Some(EncodingSpec {
            field: "fpr".into(),
            title: Some("False Positive Rate".into()),
            ..Default::default()
        }),
        y: Some(EncodingSpec {
            field: "tpr".into(),
            title: Some("True Positive Rate".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut child = Encoding {
        x: Some(EncodingSpec { field: "fpr".into(), ..Default::default() }),
        y: Some(EncodingSpec { field: "tpr".into(), ..Default::default() }),
        ..Default::default()
    };
    child.inherit_from(&parent);
    assert_eq!(child.x.as_ref().unwrap().title.as_deref(), Some("False Positive Rate"));
    assert_eq!(child.y.as_ref().unwrap().title.as_deref(), Some("True Positive Rate"));
}

#[test]
fn inherit_from_propagates_scheme_on_same_field() {
    let parent = Encoding {
        color: Some(EncodingSpec {
            field: "species".into(),
            scheme: Some("paper_ink".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut child = Encoding {
        color: Some(EncodingSpec { field: "species".into(), ..Default::default() }),
        ..Default::default()
    };
    child.inherit_from(&parent);
    assert_eq!(child.color.as_ref().unwrap().scheme.as_deref(), Some("paper_ink"));
}

#[test]
fn inherit_from_does_not_overwrite_child_title() {
    let parent = Encoding {
        x: Some(EncodingSpec {
            field: "fpr".into(),
            title: Some("Parent Title".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut child = Encoding {
        x: Some(EncodingSpec {
            field: "fpr".into(),
            title: Some("Child Title".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    child.inherit_from(&parent);
    assert_eq!(child.x.as_ref().unwrap().title.as_deref(), Some("Child Title"));
}

#[test]
fn inherit_from_does_not_cross_pollinate_different_fields() {
    let parent = Encoding {
        x: Some(EncodingSpec {
            field: "fpr".into(),
            title: Some("False Positive Rate".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut child = Encoding {
        x: Some(EncodingSpec { field: "other_field".into(), ..Default::default() }),
        ..Default::default()
    };
    child.inherit_from(&parent);
    assert_eq!(child.x.as_ref().unwrap().title, None);
}

// --- Encoding inheritance edge-case tests ---

#[test]
fn inherit_from_partial_encoding_only_fills_absent_channels() {
    // Parent has x, y, color, size. Child only has x, color.
    // Expected: child.y and child.size adopt parent values;
    // child.x and child.color remain untouched.
    let parent = Encoding {
        x: Some(EncodingSpec { field: "px".into(), title: Some("Parent X".into()), ..Default::default() }),
        y: Some(EncodingSpec { field: "py".into(), title: Some("Parent Y".into()), ..Default::default() }),
        color: Some(EncodingSpec { field: "pc".into(), scheme: Some("blues".into()), ..Default::default() }),
        size: Some(EncodingSpec { field: "ps".into(), ..Default::default() }),
        ..Default::default()
    };
    let mut child = Encoding {
        x: Some(EncodingSpec { field: "cx".into(), ..Default::default() }),
        color: Some(EncodingSpec { field: "cc".into(), ..Default::default() }),
        ..Default::default()
    };
    child.inherit_from(&parent);
    // x field differs → no metadata propagation
    assert_eq!(child.x.as_ref().unwrap().field, "cx");
    assert_eq!(child.x.as_ref().unwrap().title, None);
    // y was None → adopted from parent
    assert_eq!(child.y.as_ref().unwrap().field, "py");
    assert_eq!(child.y.as_ref().unwrap().title.as_deref(), Some("Parent Y"));
    // color field differs → no metadata propagation
    assert_eq!(child.color.as_ref().unwrap().field, "cc");
    assert_eq!(child.color.as_ref().unwrap().scheme, None);
    // size was None → adopted from parent
    assert_eq!(child.size.as_ref().unwrap().field, "ps");
}

#[test]
fn inherit_from_propagates_all_metadata_on_same_field() {
    // When field matches, scale + title + scheme + type_ + axis + legend + format + format_type
    // all propagate if child has None for that slot.
    let parent = Encoding {
        x: Some(EncodingSpec {
            field: "val".into(),
            type_: Some(DataType::Quantitative),
            scale: Some(ScaleSpec::Log { base: 2.0, common: ContinuousScaleCommon { domain: None, range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false }, nice: true }),
            title: Some("Value (log2)".into()),
            axis: Some(Box::new(crate::render::chart_config::AxisStyleSpec {
                grid: Some(false),
                ..Default::default()
            })),
            legend: Some(Box::new(crate::render::chart_config::LegendStyleSpec::default())),
            format: Some(".2f".into()),
            format_type: Some("number".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut child = Encoding {
        x: Some(EncodingSpec { field: "val".into(), ..Default::default() }),
        ..Default::default()
    };
    child.inherit_from(&parent);
    let cx = child.x.as_ref().unwrap();
    assert!(cx.scale.is_some(), "scale should be inherited");
    assert_eq!(cx.title.as_deref(), Some("Value (log2)"));
    assert_eq!(cx.type_, Some(DataType::Quantitative));
    assert!(cx.axis.is_some(), "axis should be inherited");
    assert!(cx.legend.is_some(), "legend should be inherited");
    assert_eq!(cx.format.as_deref(), Some(".2f"));
    assert_eq!(cx.format_type.as_deref(), Some("number"));
}

#[test]
fn inherit_from_does_not_overwrite_any_child_metadata() {
    // Child has all metadata set; parent has different values for each.
    // None of the child's values should be replaced.
    let parent = Encoding {
        x: Some(EncodingSpec {
            field: "val".into(),
            type_: Some(DataType::Temporal),
            scale: Some(ScaleSpec::Linear { common: ContinuousScaleCommon { domain: None, range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false }, nice: true, zero: false, system_injected: false }),
            title: Some("Parent Title".into()),
            scheme: Some("parent_scheme".into()),
            format: Some("parent_fmt".into()),
            format_type: Some("parent_ft".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut child = Encoding {
        x: Some(EncodingSpec {
            field: "val".into(),
            type_: Some(DataType::Quantitative),
            scale: Some(ScaleSpec::Log { base: 10.0, common: ContinuousScaleCommon { domain: None, range: None, clamp: false, padding: None, scheme: None, domain_param: None, reverse: false }, nice: false }),
            title: Some("Child Title".into()),
            scheme: Some("child_scheme".into()),
            format: Some("child_fmt".into()),
            format_type: Some("child_ft".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    child.inherit_from(&parent);
    let cx = child.x.as_ref().unwrap();
    assert_eq!(cx.type_, Some(DataType::Quantitative));
    assert!(matches!(cx.scale, Some(ScaleSpec::Log { .. })));
    assert_eq!(cx.title.as_deref(), Some("Child Title"));
    assert_eq!(cx.scheme.as_deref(), Some("child_scheme"));
    assert_eq!(cx.format.as_deref(), Some("child_fmt"));
    assert_eq!(cx.format_type.as_deref(), Some("child_ft"));
}

#[test]
fn inherit_non_positional_skips_x_y_x2_y2() {
    // Parent has x, y, x2, y2, color, size. Child has nothing.
    // inherit_non_positional must adopt color and size but NOT x/y/x2/y2.
    let parent = Encoding {
        x: Some(EncodingSpec { field: "px".into(), ..Default::default() }),
        y: Some(EncodingSpec { field: "py".into(), ..Default::default() }),
        x2: Some(EncodingSpec { field: "px2".into(), ..Default::default() }),
        y2: Some(EncodingSpec { field: "py2".into(), ..Default::default() }),
        color: Some(EncodingSpec { field: "pc".into(), scheme: Some("reds".into()), ..Default::default() }),
        size: Some(EncodingSpec { field: "ps".into(), ..Default::default() }),
        opacity: Some(EncodingSpec { field: "po".into(), ..Default::default() }),
        ..Default::default()
    };
    let mut child = Encoding::default();
    child.inherit_non_positional(&parent);
    // Positional channels must remain None
    assert!(child.x.is_none(), "x should not be inherited");
    assert!(child.y.is_none(), "y should not be inherited");
    assert!(child.x2.is_none(), "x2 should not be inherited");
    assert!(child.y2.is_none(), "y2 should not be inherited");
    // Non-positional channels must be inherited
    assert_eq!(child.color.as_ref().unwrap().field, "pc");
    assert_eq!(child.color.as_ref().unwrap().scheme.as_deref(), Some("reds"));
    assert_eq!(child.size.as_ref().unwrap().field, "ps");
    assert_eq!(child.opacity.as_ref().unwrap().field, "po");
}

#[test]
fn inherit_non_positional_propagates_metadata_on_same_field() {
    // Child has color with same field but no scheme; parent has scheme.
    let parent = Encoding {
        color: Some(EncodingSpec {
            field: "species".into(),
            scheme: Some("dark2".into()),
            title: Some("Species Group".into()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let mut child = Encoding {
        color: Some(EncodingSpec { field: "species".into(), ..Default::default() }),
        ..Default::default()
    };
    child.inherit_non_positional(&parent);
    assert_eq!(child.color.as_ref().unwrap().scheme.as_deref(), Some("dark2"));
    assert_eq!(child.color.as_ref().unwrap().title.as_deref(), Some("Species Group"));
}

#[test]
fn inherit_from_propagates_stroke_and_angle_channels() {
    // Verify that the late-addition stroke_width/stroke_opacity/stroke_dash/angle/fill_opacity
    // channels are properly inherited when child is None.
    let parent = Encoding {
        stroke_width: Some(EncodingSpec { field: "sw".into(), ..Default::default() }),
        stroke_opacity: Some(EncodingSpec { field: "so".into(), ..Default::default() }),
        stroke_dash: Some(EncodingSpec { field: "sd".into(), ..Default::default() }),
        angle: Some(EncodingSpec { field: "a".into(), ..Default::default() }),
        fill_opacity: Some(EncodingSpec { field: "fo".into(), ..Default::default() }),
        ..Default::default()
    };
    let mut child = Encoding::default();
    child.inherit_from(&parent);
    assert_eq!(child.stroke_width.as_ref().unwrap().field, "sw");
    assert_eq!(child.stroke_opacity.as_ref().unwrap().field, "so");
    assert_eq!(child.stroke_dash.as_ref().unwrap().field, "sd");
    assert_eq!(child.angle.as_ref().unwrap().field, "a");
    assert_eq!(child.fill_opacity.as_ref().unwrap().field, "fo");
}

#[test]
fn inherit_from_tooltip_fields_only_fills_when_child_is_none() {
    // tooltip_fields inheritance: child None → adopt parent; child Some → keep child.
    let parent = Encoding {
        tooltip_fields: Some(vec![
            EncodingSpec { field: "a".into(), ..Default::default() },
            EncodingSpec { field: "b".into(), ..Default::default() },
        ]),
        ..Default::default()
    };
    // Case 1: child is None → inherits
    let mut child1 = Encoding::default();
    child1.inherit_from(&parent);
    assert_eq!(child1.tooltip_fields.as_ref().unwrap().len(), 2);

    // Case 2: child already has tooltip_fields → keeps own
    let mut child2 = Encoding {
        tooltip_fields: Some(vec![
            EncodingSpec { field: "only_mine".into(), ..Default::default() },
        ]),
        ..Default::default()
    };
    child2.inherit_from(&parent);
    assert_eq!(child2.tooltip_fields.as_ref().unwrap().len(), 1);
    assert_eq!(child2.tooltip_fields.as_ref().unwrap()[0].field, "only_mine");
}

fn tooltip_list(fields: &[&str]) -> Option<Vec<EncodingSpec>> {
    Some(fields.iter().map(|f| EncodingSpec { field: (*f).into(), ..Default::default() }).collect())
}

/// A layer's own single-field `tooltip` is its tooltip: the parent's list is
/// the same channel spelled the other way, so it is not taken. The rejected
/// reading treats the two spellings as two channels and hands the layer the
/// parent's `["v", "y"]` beside its own `z`.
#[test]
fn a_layers_own_tooltip_keeps_the_parents_tooltip_fields_out() {
    let parent = Encoding { tooltip_fields: tooltip_list(&["v", "y"]), ..Default::default() };
    for inherit in [Encoding::inherit_from, Encoding::inherit_non_positional] {
        let mut child = Encoding {
            tooltip: Some(EncodingSpec { field: "z".into(), ..Default::default() }),
            ..Default::default()
        };
        inherit(&mut child, &parent);
        assert_eq!(child.tooltip.as_ref().map(|t| t.field.as_str()), Some("z"));
        assert_eq!(child.tooltip_fields, None);
    }
}

/// A layer's own `tooltip_fields` list is its tooltip: the parent's
/// single-field `tooltip` is not added beside it.
#[test]
fn a_layers_own_tooltip_fields_keep_the_parents_tooltip_out() {
    let parent = Encoding {
        tooltip: Some(EncodingSpec { field: "v".into(), ..Default::default() }),
        ..Default::default()
    };
    for inherit in [Encoding::inherit_from, Encoding::inherit_non_positional] {
        let mut child = Encoding { tooltip_fields: tooltip_list(&["z"]), ..Default::default() };
        inherit(&mut child, &parent);
        assert_eq!(child.tooltip, None);
        assert_eq!(child.tooltip_fields, tooltip_list(&["z"]));
    }
}

/// A layer with no tooltip of its own takes the parent's whole tooltip, both
/// spellings.
#[test]
fn a_layer_without_a_tooltip_takes_both_of_the_parents_spellings() {
    let parent = Encoding {
        tooltip: Some(EncodingSpec { field: "v".into(), ..Default::default() }),
        tooltip_fields: tooltip_list(&["v", "y"]),
        ..Default::default()
    };
    for inherit in [Encoding::inherit_from, Encoding::inherit_non_positional] {
        let mut child = Encoding::default();
        inherit(&mut child, &parent);
        assert_eq!(child.tooltip, parent.tooltip);
        assert_eq!(child.tooltip_fields, parent.tooltip_fields);
    }
}

#[test]
fn inherit_from_completely_empty_parent_is_noop() {
    let parent = Encoding::default();
    let mut child = Encoding {
        x: Some(EncodingSpec { field: "cx".into(), title: Some("My X".into()), ..Default::default() }),
        color: Some(EncodingSpec { field: "cc".into(), ..Default::default() }),
        ..Default::default()
    };
    let child_before = child.clone();
    child.inherit_from(&parent);
    assert_eq!(child, child_before);
}

#[test]
fn inherit_from_completely_empty_child_adopts_everything() {
    let parent = Encoding {
        x: Some(EncodingSpec { field: "px".into(), ..Default::default() }),
        y: Some(EncodingSpec { field: "py".into(), ..Default::default() }),
        color: Some(EncodingSpec { field: "pc".into(), ..Default::default() }),
        size: Some(EncodingSpec { field: "ps".into(), ..Default::default() }),
        shape: Some(EncodingSpec { field: "psh".into(), ..Default::default() }),
        opacity: Some(EncodingSpec { field: "po".into(), ..Default::default() }),
        text: Some(EncodingSpec { field: "pt".into(), ..Default::default() }),
        ..Default::default()
    };
    let mut child = Encoding::default();
    child.inherit_from(&parent);
    assert_eq!(child.x.as_ref().unwrap().field, "px");
    assert_eq!(child.y.as_ref().unwrap().field, "py");
    assert_eq!(child.color.as_ref().unwrap().field, "pc");
    assert_eq!(child.size.as_ref().unwrap().field, "ps");
    assert_eq!(child.shape.as_ref().unwrap().field, "psh");
    assert_eq!(child.opacity.as_ref().unwrap().field, "po");
    assert_eq!(child.text.as_ref().unwrap().field, "pt");
}

#[test]
fn overlay_from_replaces_present_channels() {
    let mut base = Encoding::default();
    base.x = Some(EncodingSpec { field: "base_x".into(), ..Default::default() });
    base.y = Some(EncodingSpec { field: "base_y".into(), ..Default::default() });

    let mut overlay = Encoding::default();
    overlay.x = Some(EncodingSpec { field: "overlay_x".into(), ..Default::default() });
    // overlay.y is None — should NOT replace base.y

    base.overlay_from(&overlay);
    assert_eq!(base.x.as_ref().unwrap().field, "overlay_x");
    assert_eq!(base.y.as_ref().unwrap().field, "base_y");
}

#[test]
fn overlay_from_covers_all_12_channels() {
    let mut base = Encoding {
        x: Some(EncodingSpec { field: "bx".into(), ..Default::default() }),
        y: Some(EncodingSpec { field: "by".into(), ..Default::default() }),
        color: Some(EncodingSpec { field: "bc".into(), ..Default::default() }),
        size: Some(EncodingSpec { field: "bs".into(), ..Default::default() }),
        shape: Some(EncodingSpec { field: "bsh".into(), ..Default::default() }),
        opacity: Some(EncodingSpec { field: "bo".into(), ..Default::default() }),
        x2: Some(EncodingSpec { field: "bx2".into(), ..Default::default() }),
        y2: Some(EncodingSpec { field: "by2".into(), ..Default::default() }),
        text: Some(EncodingSpec { field: "bt".into(), ..Default::default() }),
        tooltip: Some(EncodingSpec { field: "btt".into(), ..Default::default() }),
        tooltip_fields: None,
        href: Some(EncodingSpec { field: "bh".into(), ..Default::default() }),
        description: Some(EncodingSpec { field: "bd".into(), ..Default::default() }),
        key: Some(EncodingSpec { field: "bk".into(), ..Default::default() }),
        url: None,
        stroke_width: None,
        stroke_opacity: None,
        stroke_dash: None,
        angle: None,
        fill_opacity: None,
        group: None,
    };
    // Overlay only tooltip, href, description (the three that were missed
    // by the old inline merge).
    let overlay = Encoding {
        tooltip: Some(EncodingSpec { field: "ott".into(), ..Default::default() }),
        href: Some(EncodingSpec { field: "oh".into(), ..Default::default() }),
        description: Some(EncodingSpec { field: "od".into(), ..Default::default() }),
        ..Default::default()
    };
    base.overlay_from(&overlay);
    // Replaced channels:
    assert_eq!(base.tooltip.as_ref().unwrap().field, "ott");
    assert_eq!(base.href.as_ref().unwrap().field, "oh");
    assert_eq!(base.description.as_ref().unwrap().field, "od");
    // Untouched channels:
    assert_eq!(base.x.as_ref().unwrap().field, "bx");
    assert_eq!(base.y.as_ref().unwrap().field, "by");
    assert_eq!(base.color.as_ref().unwrap().field, "bc");
}

// ── `#176`(11) + N3: `EncodingSpec(sort=)` construction-time gate ──────────

fn attach<F, T>(build: F) -> T
where
    F: for<'py> FnOnce(Python<'py>) -> T,
{
    pyo3::Python::initialize();
    Python::attach(build)
}

/// Build an `EncodingSpec` via its real PyO3 constructor (not the bare
/// struct literal every other test in this file uses), so `sort`'s
/// construction-time validation actually runs. Every parameter besides
/// `field` and `sort` is `None`, matching `EncodingSpec(field="a", sort=...)`.
fn new_encoding_spec_with_sort<'py>(
    py: Python<'py>,
    sort: Option<&pyo3::Bound<'py, pyo3::PyAny>>,
) -> PyResult<EncodingSpec> {
    EncodingSpec::new(py, "a", None, None, None, None, None, sort, None, None, None, None, None, None)
}

/// Matrix control: valid `EncodingSpec(sort=)` forms (a direction string, an
/// explicit list, and a well-formed sort-field object) construct without
/// error, byte-identically to before this gate existed.
#[test]
fn encoding_spec_sort_valid_forms_construct() {
    attach(|py| {
        for sort_json in [
            serde_json::json!("ascending"),
            serde_json::json!("descending"),
            serde_json::json!(["b", "a"]),
            serde_json::json!({"field": "v"}),
            serde_json::json!({"field": "v", "op": "mean", "order": "descending"}),
        ] {
            let json_module = py.import("json").unwrap();
            let s = serde_json::to_string(&sort_json).unwrap();
            let obj = json_module.call_method1("loads", (s,)).unwrap();
            let spec = new_encoding_spec_with_sort(py, Some(&obj))
                .unwrap_or_else(|e| panic!("valid sort {sort_json} must construct: {e}"));
            assert_eq!(spec.sort, Some(sort_json));
        }
    });
}

/// `EncodingSpec(sort=None)` (the common case, no sort declared) constructs
/// with `sort: None` -- unaffected by the gate.
#[test]
fn encoding_spec_sort_absent_constructs() {
    attach(|py| {
        let spec = new_encoding_spec_with_sort(py, None).unwrap();
        assert_eq!(spec.sort, None);
    });
}

/// A sort object missing `"field"` entirely refuses with the exact text,
/// naming the public entry (`EncodingSpec(sort=...)`) and the value
/// received. `serde_json::Map` sorts keys (no `preserve_order` feature), so
/// the echoed value's key order is deterministic.
#[test]
fn encoding_spec_sort_object_missing_field_refuses() {
    attach(|py| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("op", "mean").unwrap();
        let err = new_encoding_spec_with_sort(py, Some(&d.into_any())).unwrap_err();
        let msg = err.value(py).to_string();
        assert_eq!(
            msg,
            "EncodingSpec(sort=...): sort object missing string \"field\"; received {\"op\":\"mean\"}"
        );
        assert!(err.is_instance_of::<PyValueError>(py), "expected ValueError, got {err:?}");
    });
}

/// A sort object whose `"field"` is not a string refuses with the exact text.
#[test]
fn encoding_spec_sort_object_non_string_field_refuses() {
    attach(|py| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("field", 1).unwrap();
        let err = new_encoding_spec_with_sort(py, Some(&d.into_any())).unwrap_err();
        assert!(err.is_instance_of::<PyValueError>(py));
        let msg = err.value(py).to_string();
        assert_eq!(
            msg,
            "EncodingSpec(sort=...): sort \"field\" must be a string, got 1; received {\"field\":1}"
        );
    });
}

/// A sort object naming an unknown `"op"` refuses with the exact text.
#[test]
fn encoding_spec_sort_object_unknown_op_refuses() {
    attach(|py| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("field", "v").unwrap();
        d.set_item("op", "not-a-real-op").unwrap();
        let err = new_encoding_spec_with_sort(py, Some(&d.into_any())).unwrap_err();
        assert!(err.is_instance_of::<PyValueError>(py));
        let msg = err.value(py).to_string();
        assert_eq!(
            msg,
            "EncodingSpec(sort=...): unsupported sort op \"not-a-real-op\"; \
             received {\"field\":\"v\",\"op\":\"not-a-real-op\"}"
        );
    });
}

/// A sort object naming an unknown `"order"` refuses with the exact text --
/// this shape had NO warning arm at all before this gate (any
/// non-`"descending"` order silently became ascending), so this is new
/// refusal behavior, not a relocated warning.
#[test]
fn encoding_spec_sort_object_unknown_order_refuses() {
    attach(|py| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("field", "v").unwrap();
        d.set_item("order", "not-a-real-order").unwrap();
        let err = new_encoding_spec_with_sort(py, Some(&d.into_any())).unwrap_err();
        assert!(err.is_instance_of::<PyValueError>(py));
        let msg = err.value(py).to_string();
        assert_eq!(
            msg,
            "EncodingSpec(sort=...): unsupported sort order \"not-a-real-order\"; \
             received {\"field\":\"v\",\"order\":\"not-a-real-order\"}"
        );
    });
}

/// N3's own reproduction: `sort={"x": 1}` (no `"field"` key at all) refuses,
/// where it previously constructed and only degraded at render.
#[test]
fn encoding_spec_sort_n3_reproduction_refuses() {
    attach(|py| {
        let d = pyo3::types::PyDict::new(py);
        d.set_item("x", 1).unwrap();
        let err = new_encoding_spec_with_sort(py, Some(&d.into_any())).unwrap_err();
        assert!(err.is_instance_of::<PyValueError>(py));
    });
}

/// A non-object malformed sort (e.g. a bare unrecognized string) is
/// untouched by this gate -- it still constructs and degrades at render via
/// `RenderWarning::SortSpecIgnored`'s other, un-deleted arm.
#[test]
fn encoding_spec_sort_non_object_form_is_not_validated_at_construction() {
    attach(|py| {
        let spec = new_encoding_spec_with_sort(py, Some(&"not-a-keyword".into_pyobject(py).unwrap().into_any()))
            .expect("a non-object sort form is out of this gate's scope");
        assert_eq!(spec.sort, Some(serde_json::json!("not-a-keyword")));
    });
}

/// `sort_accepted_vocabulary()` publishes the same `op`/`order` spellings
/// `parse_sort_field_object` accepts -- the two-sided check the
/// `scale_accepted_keys`/`interpolate_accepted_values` precedents both pin,
/// so Python's channels validate against the real vocabulary rather than a
/// hand-copied one.
#[test]
fn sort_accepted_vocabulary_matches_validate_sort_field_object() {
    let vocab = sort_accepted_vocabulary();
    let ops = vocab.get("op").expect("vocabulary must carry an 'op' key");
    let orders = vocab.get("order").expect("vocabulary must carry an 'order' key");
    for op in ops {
        let obj = serde_json::json!({"field": "v", "op": op});
        let serde_json::Value::Object(map) = &obj else { unreachable!() };
        assert!(
            super::sort::parse_sort_field_object(map).is_ok(),
            "published op {op:?} must validate"
        );
    }
    for order in orders {
        let obj = serde_json::json!({"field": "v", "order": order});
        let serde_json::Value::Object(map) = &obj else { unreachable!() };
        assert!(
            super::sort::parse_sort_field_object(map).is_ok(),
            "published order {order:?} must validate"
        );
    }
    // An op/order this vocabulary does NOT list must refuse -- the negative
    // control proving the two sides are actually the same set, not both
    // accidentally permissive. Both keys get their own negative control:
    // the op side alone would not catch an order-side drift (or vice versa).
    let bad_op = serde_json::json!({"field": "v", "op": "not-in-vocabulary"});
    let serde_json::Value::Object(map) = &bad_op else { unreachable!() };
    assert!(super::sort::parse_sort_field_object(map).is_err());

    let bad_order = serde_json::json!({"field": "v", "order": "not-in-vocabulary"});
    let serde_json::Value::Object(map) = &bad_order else { unreachable!() };
    assert!(super::sort::parse_sort_field_object(map).is_err());
}

/// [`Encoding::channels`] on a struct literal populating every field (this
/// literal is itself exhaustive -- no `..Default::default()` -- so a field
/// added to `Encoding` breaks it too, an independent compile-time signal
/// alongside `channels`' own destructuring) returns exactly one entry per
/// single-value channel plus one per `tooltip_fields` entry, in declaration
/// order, each paired with the `EncodingSpec` that channel actually carries
/// (never a copy of some other channel's).
#[test]
fn channels_enumerates_every_populated_field_with_the_right_spec() {
    let spec = |f: &str| Some(EncodingSpec { field: f.into(), ..Default::default() });
    let e = Encoding {
        x: spec("x"),
        y: spec("y"),
        color: spec("color"),
        size: spec("size"),
        shape: spec("shape"),
        opacity: spec("opacity"),
        x2: spec("x2"),
        y2: spec("y2"),
        text: spec("text"),
        tooltip: spec("tooltip"),
        tooltip_fields: Some(vec![
            EncodingSpec { field: "tf0".into(), ..Default::default() },
            EncodingSpec { field: "tf1".into(), ..Default::default() },
        ]),
        href: spec("href"),
        description: spec("description"),
        key: spec("key"),
        url: spec("url"),
        stroke_width: spec("stroke_width"),
        stroke_opacity: spec("stroke_opacity"),
        stroke_dash: spec("stroke_dash"),
        angle: spec("angle"),
        fill_opacity: spec("fill_opacity"),
        // A grouping key, not a binding: no channel lists it.
        group: spec("group"),
    };
    let channels = e.channels();
    let names: Vec<&str> = channels.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(
        names,
        vec![
            "x", "y", "color", "size", "shape", "opacity", "x2", "y2", "text", "tooltip", "href",
            "description", "key", "url", "stroke_width", "stroke_opacity", "stroke_dash", "angle",
            "fill_opacity", "tooltip_fields[0]", "tooltip_fields[1]",
        ]
    );
    // Every returned EncodingSpec is the one this channel actually carries,
    // not a copy of some other channel's.
    for (name, enc) in e.channels() {
        let expected = if let Some(idx) = name
            .strip_prefix("tooltip_fields[")
            .and_then(|s| s.strip_suffix(']'))
        {
            format!("tf{idx}")
        } else {
            name.clone()
        };
        assert_eq!(enc.field, expected, "channel {name} carries the wrong EncodingSpec");
    }
}

/// `retain_bindings` clears each rejected singular binding and drops each
/// rejected `tooltip_fields` entry, keeping an emptied list bound.
#[test]
fn retain_bindings_clears_rejected_bindings_and_tooltip_entries() {
    let spec = |field: &str| Some(EncodingSpec { field: field.into(), ..Default::default() });
    let mut e = Encoding {
        x: spec("x"),
        color: spec("c"),
        shape: spec("s"),
        tooltip_fields: Some(vec![EncodingSpec { field: "a".into(), ..Default::default() }]),
        ..Default::default()
    };
    e.retain_bindings(|wire, binding| wire != "shape" && binding.field != "a");
    assert_eq!(e.x, spec("x"));
    assert_eq!(e.color, spec("c"));
    assert_eq!(e.shape, None);
    assert_eq!(e.tooltip_fields, Some(vec![]));
}
