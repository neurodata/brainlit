# WSL2 Installation Instructions

For Windows 10 users that prefer Linux functionality without the speed sacrifice of a VM, Brainlit can be installed and run on WSL2.
- OS Specifications: Version 1903, Build 18362 or higher
- [Installation Instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
- Any Linux distribution can be installed. Ubuntu16.04.3 was used for this tutorial.

#### Install python required libraries and build tools. 
Run the below commands to configure the WSL2 environment. See [here](https://stackoverflow.com/questions/8097161/how-would-i-build-python-myself-from-source-code-on-ubuntu/31492697) for more information. 
```
$ sudo apt update && sudo apt install -y build-essential git libexpat1-dev libssl-dev zlib1g-dev
$ libncurses5-dev libbz2-dev liblzma-dev
$ libsqlite3-dev libffi-dev tcl-dev linux-headers-generic libgdbm-dev
$ libreadline-dev tk tk-dev
```

#### Install a python version management tool, and create/activate a virtual environment
- [Pyenv WSL2 Install](https://gist.github.com/monkut/35c2ef098b871144b49f3f9979032cee) (easiest for WSL2)
- [Anaconda WSL2 Install](https://gist.github.com/kauffmanes/5e74916617f9993bc3479f401dfec7da)

#### Install brainlit
- See [installation section](#installation) of README.md

#### Create and save AWS Secrets file
- See AWS Secrets file section of README.md

#### Configure jupyter notebook
`$ sudo apt-get install jupyter notebook`
Add this line to your `~/.bashrc` script: 
``export DISPLAY=`grep -oP "(?<=nameserver ).+" /etc/resolv.conf`:0.0 ``

#### Configure X11 Port Forwarding
- Install [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/) on your Windows host machine
- Let VcXsrv through your Public & Private windows firewall. 
(Control Panel -> System and Security -> Windows Defender Firewall -> Allowed Apps -> Change Settings)
- Run XLaunch on your Windows Host Machine with default settings AND select the "Disable Access Control" option
- To confim X11 Port Forwarding is configured, run `xclock` on the subsystem.  This should launch on your windows machine. 

#### Exceptions
- The Napari viewer cannot be fully launched (only launches a black screen), because [OpenGL versions>1.5 are not currently supported by WSL2](https://discourse.ubuntu.com/t/opengl-on-ubuntu-on-wsl-2-timeline/17599).  This should be resolved in upcoming WSL2 updates.



