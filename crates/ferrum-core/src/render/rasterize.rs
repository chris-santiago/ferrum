//! RGBA grid → PNG bytes. Pinned encoder settings for determinism.

use png::{Encoder, BitDepth, ColorType, Filter, Compression};
use std::io::Cursor;

/// Encode an RGBA pixel buffer as PNG bytes.
/// Pinned: Filter::Sub, Compression::High (level 9 / flate2::best). Required for raster goldens.
pub fn encode_png(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(
        rgba.len(),
        (width * height * 4) as usize,
        "RGBA buffer length mismatch: expected {} bytes, got {}",
        width * height * 4,
        rgba.len()
    );
    let mut out = Vec::with_capacity(rgba.len() / 4);
    {
        let mut encoder = Encoder::new(Cursor::new(&mut out), width, height);
        encoder.set_color(ColorType::Rgba);
        encoder.set_depth(BitDepth::Eight);
        encoder.set_filter(Filter::Sub);
        encoder.set_compression(Compression::High);
        let mut writer = encoder.write_header().expect("png header");
        writer.write_image_data(rgba).expect("png write");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Sha256, Digest};

    fn hash(bytes: &[u8]) -> String {
        let mut h = Sha256::new();
        h.update(bytes);
        format!("{:x}", h.finalize())
    }

    #[test]
    fn encode_png_byte_deterministic_across_calls() {
        let mut rgba = vec![0u8; 4 * 4 * 4];
        for i in 0..16 {
            rgba[i * 4] = i as u8;
            rgba[i * 4 + 3] = 255;
        }
        let a = encode_png(4, 4, &rgba);
        let b = encode_png(4, 4, &rgba);
        assert_eq!(hash(&a), hash(&b), "PNG bytes must be byte-identical across calls");
    }

    #[test]
    fn encode_png_minimal_buffer() {
        let rgba = vec![255, 0, 0, 255];  // single red pixel
        let bytes = encode_png(1, 1, &rgba);
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']), "PNG magic missing");
        assert!(bytes.len() < 200, "1x1 PNG should be small: got {} bytes", bytes.len());
    }

    #[test]
    fn encode_png_large_buffer_succeeds() {
        let rgba = vec![128u8; 4 * 1024 * 1024];  // 1024x1024 grey
        let bytes = encode_png(1024, 1024, &rgba);
        assert!(bytes.len() > 100, "1024x1024 PNG should produce some data");
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    #[should_panic(expected = "RGBA buffer length mismatch")]
    fn encode_png_wrong_length_panics() {
        let rgba = vec![0u8; 10];
        encode_png(2, 2, &rgba);
    }
}
