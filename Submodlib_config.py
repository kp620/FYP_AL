import subprocess
import os

# 1. pip install pybind11
# 2. python setup.py build
# 3. pip install .
# 4. pip install "numpy<=1.21"

if os.path.exists("submodlib"):
    print("submodlib already exists!")
    from submodlib.functions.facilityLocation import FacilityLocationFunction
    from submodlib.helper import create_kernel
    print("submodlib imported!")
else:
    command = [
        "git clone https://github.com/decile-team/submodlib.git",
        "pip install -e .",
    ]
    for cmd in command:
        subprocess.run(cmd, shell=True)
    print("submodlib installed!")
    from submodlib.functions.facilityLocation import FacilityLocationFunction
    from submodlib.helper import create_kernel
    print("submodlib imported!")