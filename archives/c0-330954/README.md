# Certified source archive — 330954b

This branch is an archival container for the exact local-only Git objects that
produce certified XOX source commit:

- source commit: `330954b27f146b8a17db2cb353c3e620968bad5e`
- source tree: `d77401fa1913125198da997712641820483f4a29`
- prerequisite already present in the public repository:
  `a0b8ab2c90af70fa8095556033e605477ec186ae`
- bundle SHA-256:
  `c0591f4ad9a1d8193f2d7bc0d21420b29df9043942df64c6c15fdc0b6714a1b9`

The bundle was created from `gpu-precision-g1-g3` with the prerequisite
excluded, verified with `git bundle verify`, base64-encoded, and split into
fixed chunks under this directory.

This archive branch itself does **not** pretend that its branch-tip commit is
the certified source commit. Reconstruct the bundle from the chunks, fetch the
prerequisite from the public repository, then fetch
`refs/heads/gpu-precision-g1-g3` from the bundle. The recovered ref must equal
the source commit above and its tree must equal the source tree above.

Production mutation is not authorized by this archive.
