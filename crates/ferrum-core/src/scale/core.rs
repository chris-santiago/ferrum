//! Sealed `Scale` enum that centralises math for every scale variant.
//! Each task in section C/D extends this with a new variant and the
//! corresponding match arms in the dispatch methods.

#![allow(dead_code)] // populated incrementally; suppress until first variant lands
