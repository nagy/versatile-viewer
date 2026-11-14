//! The document bytes that more than one module of this crate reads.
//!
//! `grid` and `loader` test their document paths with a real PDF, and a crate
//! compiles its own `#[cfg(test)]` items alone, so the bytes live here instead
//! of in `vv-source::pdf`.
//! `crates/vv-source/src/pdf.rs` carries the same two strings for the parser
//! tests of that crate: keep the two copies in step.

/// A minimal single-page PDF (8.5x11 in letter, one text object).
///
/// A converted answer of a document is a PDF, so a test of that path needs
/// bytes the parser accepts.
pub(crate) const MINIMAL_PDF: &[u8] = b"%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 60>>stream
BT /F1 24 Tf 72 700 Td (Hello vv) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
trailer<</Root 1 0 R/Size 6>>";

/// A two-page PDF with different page sizes, for the plane-grid tests.
pub(crate) const TWO_PAGE_PDF: &[u8] = b"%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R 6 0 R]/Count 2>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length 60>>stream
BT /F1 24 Tf 72 700 Td (Hello vv) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
6 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 100]/Contents 7 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
7 0 obj<</Length 40>>stream
BT /F1 12 Tf 10 50 Td (page two) Tj ET
endstream
endobj
trailer<</Root 1 0 R/Size 8>>";
