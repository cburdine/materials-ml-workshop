# Installing Python and Jupyter Notebook

## Installing Python

In order to write and execute Python code, you will need to install both Python and Jupyter Notebook. Depending on which operating system you are running on your system, the installation procedure may vary.

::::{tab-set}
:::{tab-item} Windows

**Installing Python**

Install the latest version of Python for Windows [here](https://www.python.org/downloads/windows/).

Download the Windows installer based on your operating system (most modern computers use 64-bit). Once the executable file download is complete, open it, click on Run, and the installation process will start.

**Installing Jupyter Notebook**

You can use either pip or Anaconda to install Jupyter Notebook on Windows. It is recommended that you use pip ("Package Installer for Python"), to install Jupyter Notebook, but we provide instructions here for Anaconda, if pip does not work on your system.

**Installing Jupyter Notebook via pip**

* Step 1: Open Command Prompt and execute the command  ```pip --version```  to check whether pip is installed. If not, you will need to install pip through some other method, or try using Anaconda.

* Step 2: If pip is already installed, you need to execute the command  ```python -m pip install jupyter``` to complete the Jupyter Notebook download and installation of all files and packages.

* Step 3: Execute the command  ```jupyter notebook```  to launch Jupyter Notebook.

**Installing Jupyter Notebook via Anaconda:**

* Head over to [anaconda.com](https://www.anaconda.com/download#windows) and download the Anaconda installer file. Run the installer file and follow the wizard to install Anaconda.

![](anacondapromptscreenshot.jpg)

* Search for Anaconda Navigator from the Start Menu. After Anaconda Navigator launches, click the Install button under Jupyter. Wait until the Jupyter Notebook installation is completed. Then, the Install button will turn to a Launch button. Launch it for use.
* When launching Jupyter Notebook from Anaconda, notebooks will be saved to the default folder of Anaconda Powershell Prompt. To save the notebooks to a different folder, follow these steps:
* launch Anaconda Powershell Prompt
* Type  ```cd ```  to move the working directory to the folder of your choice
* Type  ```jupyter notebook```  and it will start the server from your folder

:::

:::{tab-item} Mac OS

Your Mac already shipped with a Python installation, but it is helpful to get the latest Anaconda Mac OS package. This will get you the following:
* a recent Python 3
* jupyter notebook, and
* several other data science packages.

```{note}
As a Mac user, I've had the *most success* with the *command-line* installation, rather than the graphical installation. I've found this to better support the subsequent installation of packages using `conda install <package-name>`. Therefore, these instructions guide you through the command-line installation. --EPB
```

1. Download the command line installer that is appropriate for your Mac's processor.
   1. To get the Command Line installer, visit the Anaconda [download page](https://www.anaconda.com/download#downloads).
   2. Scroll down as needed (you might not need to) toward the bottom (or search the page for `command line`) so that you find this:
   ![](MacOS_cmd_line_installers.png)
   2. Choose the appropriate installer for your Mac processor from the "Mac OS" column.
2. Use your Mac terminal to follow the [Command-line installation instructions](https://docs.anaconda.com/free/anaconda/install/mac-os/#command-line-install).
   ```{note}
   The linked instructions are from 2020, so you'll have to use the updated name of the Anaconda installer (i.e., not `~/Downloads/Anaconda3-2020.05-MacOSX-x86_64.sh`)
   ```
   1. To access your Mac terminal, use the Mac OS search feature ("Spotlight") and enter `terminal`.

:::

:::{tab-item} Linux
Chances are, if you are running Linux, you are already familiar with the command line and your distribution's package manager 😎.

**Installing Python**

Depending on your distribution and package manager, installation of Python3 may vary. For Ubuntu-like distributions, the latset version of Python can be installed with:

```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11
```

Additional steps may be needed to install pip, the Python package manager.

**Installing Jupyter Notebook**

To install Jupyter Notebook, you will need to install it via pip:

```
pip3 install jupyter
```

or on some systems:

```
python -m pip install jupyter
```
:::
::::
