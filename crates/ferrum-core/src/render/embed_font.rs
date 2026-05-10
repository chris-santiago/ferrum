//! Inter Regular as a base64-embedded @font-face block.
//! Emitted unconditionally for SVG output in Phase 7 (locked decision §11 row 15).

use base64::Engine;

use super::font::INTER_REGULAR;

/// Returns the full `<defs><style>@font-face{...}</style></defs>` block,
/// ready to be appended to the SVG header.
pub fn inter_data_url_block() -> String {
    let b64 = base64::engine::general_purpose::STANDARD.encode(INTER_REGULAR);
    format!(
        "<defs><style>@font-face{{font-family:\"Inter\";src:url(\"data:font/ttf;base64,{}\") format(\"truetype\");}}</style></defs>",
        b64,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_is_well_formed() {
        let s = inter_data_url_block();
        assert!(s.starts_with("<defs><style>@font-face"));
        assert!(s.ends_with("</style></defs>"));
        assert!(s.contains("font-family:\"Inter\""));
        assert!(s.contains("data:font/ttf;base64,"));
    }
}
