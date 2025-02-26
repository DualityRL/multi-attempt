from setuptools import setup, find_packages
import pkg_resources

# Read requirements.txt
with open("requirements.txt") as f:
    required_packages = f.read().splitlines()

# Filter out already installed packages to avoid upgrades
installed_packages = {pkg.key for pkg in pkg_resources.working_set}
filtered_requirements = list([pkg for pkg in required_packages if pkg.split("==")[0] not in installed_packages])

setup(
    name="meval",  # Your package name
    version="0.1",
    packages=find_packages(),
    install_requires=filtered_requirements,  # Only install missing packages
    include_package_data=True,
)
