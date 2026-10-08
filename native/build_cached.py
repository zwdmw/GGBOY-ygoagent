"""Offline source build using explicitly versioned xmake package caches."""
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sysconfig
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--packages", type=Path, default=Path.home() / ".xmake/packages")
parser.add_argument("--output", type=Path)
parser.add_argument("--source", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parent
source = (args.source or root / "ygoenv").resolve()
build = root / ("bridge-build-cache" if args.source else "build-cache")
build.mkdir(exist_ok=True)
versions = {"pybind11": "v2.13.6", "fmt": "10.2.1", "glog": "v0.6.0", "gflags": "v2.3.1",
            "concurrentqueue": "v1.0.4", "sqlitecpp": "3.2.1", "sqlite3": "3.53.0+0",
            "unordered_dense": "v4.4.0", "ygopro-core": "0.0.2", "lua": "v5.5.0"}
packages = {}
for name, version in versions.items():
    matches = list((args.packages / name[0] / name / version).glob("*/include"))
    if len(matches) != 1:
        raise ValueError(f"Need exactly one cached {name} {version}; found {matches}. Use build.sh to fetch dependencies.")
    packages[name] = matches[0].parent
suffix = sysconfig.get_config_var("EXT_SUFFIX")
output = (args.output or build / ("ygopro_ygoenv" + suffix)).resolve()
includes = [sysconfig.get_path("include"), str(source)]
for name, path in packages.items():
    includes.append(str(path / "include"))
    if name == "concurrentqueue":
        includes.append(str(path / "include/concurrentqueue/moodycamel"))
    if name == "lua":
        includes.append(str(path / "include/lua"))
compiler = shlex.split(os.environ.get("CXX", "g++"))
object_path = build / "ygopro.cpp.o"
compile_command = [*compiler, "-c", "-m64", "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden",
                   "-O3", "-DNDEBUG", "-std=c++17", "-march=x86-64", "-mtune=generic",
                   *["-I" + path for path in includes], "-o", str(object_path),
                   str(source / "ygoenv/ygopro/ygopro.cpp")]
link_command = [*compiler, "-shared", "-m64", "-fPIC", str(object_path), "-o", str(output),
                *["-L" + str(path / "lib") for path in packages.values()],
                "-lssl", "-lcrypto", "-lfmt", "-lglog", "-lgflags", "-lSQLiteCpp", "-lsqlite3",
                "-lygopro-core", "-llua", "-lutil", "-lpthread", "-ldl", "-lm"]
for command in (compile_command, link_command):
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
with output.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
report = {"output": str(output), "sha256": digest, "packages": {k: str(v) for k, v in packages.items()},
          "versions": versions, "compile": compile_command, "link": link_command}
(build / "build.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"output": str(output), "sha256": digest}))
