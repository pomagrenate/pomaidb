from setuptools import setup, find_packages

setup(
    name="pomaidb",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "pomaidb": ["lib/*", "*.dll", "*.so", "*.dylib", "py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
)
