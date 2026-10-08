add_rules("mode.debug", "mode.release")
add_repositories("sky-repo repo")
add_requires("ygopro-core 0.0.2", "pybind11 2.13.6", "fmt 10.2.1", "glog 0.6.0",
             "sqlite3 3.53.0+0", "lua 5.5.0", "gflags 2.3.1",
             "concurrentqueue 1.0.4", "unordered_dense 4.4.0", "sqlitecpp 3.2.1")
target("ygopro_ygoenv")
    add_rules("python.library")
    add_files("ygoenv/ygoenv/ygopro/*.cpp")
    add_packages("pybind11", "fmt", "glog", "concurrentqueue", "sqlitecpp", "unordered_dense", "ygopro-core")
    add_includedirs("ygoenv")
    set_languages("c++17")
    if is_mode("release") then
        set_policy("build.optimization.lto", true)
    end
    add_cxxflags("-march=x86-64", "-mtune=generic", {force = true})
    after_build(function (target)
        os.cp(target:targetfile(), "$(projectdir)/ygoenv/ygoenv/ygopro")
    end)
