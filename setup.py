"""Builds the Rust library (cargo, no crates from the network: the crate has no dependencies) and puts it in the
package. Needs a Rust toolchain: CARGO, or cargo on PATH, or ~/.cargo/bin/cargo."""
import os, shutil, subprocess, sys
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

ROOT = os.path.dirname(os.path.abspath(__file__))
LIB = {"darwin": "libtrianglebvh.dylib", "win32": "trianglebvh.dll"}.get(sys.platform, "libtrianglebvh.so")


def cargo():
    for c in (os.environ.get("CARGO"), shutil.which("cargo"), os.path.expanduser("~/.cargo/bin/cargo")):
        if c and os.path.isfile(c):
            return c
    raise SystemExit("trianglebvh: no Rust toolchain (set CARGO or put cargo on PATH)")


class BuildWithCargo(build_py):
    def run(self):
        target = os.path.join(ROOT, "target")
        subprocess.run([cargo(), "build", "--release", "--offline", "--locked", "--manifest-path", os.path.join(ROOT, "Cargo.toml"),
                        "--target-dir", target], check=True)
        super().run()
        shutil.copy2(os.path.join(target, "release", LIB), os.path.join(self.build_lib, "trianglebvh", LIB))


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(cmdclass={"build_py": BuildWithCargo}, distclass=BinaryDistribution)
