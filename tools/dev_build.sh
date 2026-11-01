#!/usr/bin/env bash
# Development build in place: cargo build, then put the library next to __init__.py by rename (a new inode), so a
# process that has the old library loaded keeps running (overwriting a loaded library in place crashes it).
set -euo pipefail
cd "$(dirname "$0")/.."
"${CARGO:-cargo}" build --release --offline
ext=so; [ "$(uname)" = Darwin ] && ext=dylib
cp target/release/libtrianglebvh.$ext trianglebvh/.libtrianglebvh.$ext.new
mv -f trianglebvh/.libtrianglebvh.$ext.new trianglebvh/libtrianglebvh.$ext
