import pybind11
import skbuild

skbuild.setup(
    name="bytetrack",
    version="1.0.0",
    description="Nguyen Hai Dang dep trai",
    install_requires=["numpy"],
    packages=["bytetrack"],
    package_data={"bytetrack": ["__init__.pyi"]},
    package_dir={"": "src"},
    cmake_install_dir="src",
    cmake_args=[
        "-DBUILD_PYBIND=ON",
        f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
    ],
)
