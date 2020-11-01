# WORK IN PROGRESS

## Installation process:

- Download the Windows64 release version of [Vaa3D v3.447](https://github.com/Vaa3D/release/releases/tag/v3.447)
- Clone the [pyVaa3D](https://github.com/ajkswamy/pyVaa3d) plugin repository
- Activate the Brainlit `conda` environment, `cd` into the pyVaa3D directory, and `pip install -e .`
- The original instructions can be found [here](https://github.com/ajkswamy/pyVaa3d/blob/master/INSTALL.md).

## Initial Run

- Import the method `from pyVaa3d.vaa3dWrapper import runVaa3dPlugin`

- When importing this function, the terminal will prompt the user to enter the path to the executable `start_vaa3d.sh`. Note that on Windows, the file is actually `vaa3d_msvc.exe`, located inside the `C:Users\...\Vaa3D_V3.447_Windows_MSVC_64bit\` directory. This only needs to be done once, and is cached. 

- If it loads correctly, the Python terminal should look something like this:
[![vaa3d-executable-located.png](https://i.postimg.cc/LsrYK2wD/vaa3d-executable-located.png)](https://postimg.cc/mcy27fb1)
  Note that the highlighted portion is a user input.

## Running APP2

- See python script (will update later on some other notes)

- The current version has a problem where the labels are flipped along y=x, will need to fix this.
