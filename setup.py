from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
import subprocess
import sys
import os

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Read dependencies from pyproject.toml
def get_dependencies():
    try:
        with open('pyproject.toml', 'rb') as f:
            pyproject = tomllib.load(f)
        return pyproject['project']['dependencies']
    except (FileNotFoundError, KeyError):
        return []

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            _ = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        if 'CMAKE_VERBOSE_MAKEFILE' in env:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=1']
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)


setup(
    name='ssapy',
    version='1.1.5',
    ext_modules=[CMakeExtension("ssapy._ssapy")],
    cmdclass={"build_ext": CMakeBuild},
    packages=find_packages(),
    package_data={'ssapy': ['data/*','_ssapy*.so']},
    license='MIT',
    zip_safe=False,
    include_package_data=True,
    install_requires=get_dependencies(),
)

