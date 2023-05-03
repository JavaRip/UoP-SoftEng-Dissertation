# UoP-SoftEng-Dissertation

## Installation Instructions

The follow shell will install and run the project code on Ubuntu 22.04.2. 
This has been tested on a fresh Ubuntu 22.04.2 installation in a virtual machine. 
Running this code on a modified Ubuntu system may produce unexpected results.

### Installation Script

```bash
# update system packages
sudo apt update -y
sudo apt upgrade -y

# install system dependencies
sudo apt install git -y
sudo apt install git-lfs -y
sudo apt install nodejs -y
sudo apt install npm -y
sudo apt install libgeos-dev -y
sudo apt install python3-pip -y
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.2/install.sh | bash
export NVM_DIR="$([ -z "${XDG_CONFIG_HOME-}" ] && printf %s "${HOME}/.nvm" || printf %s "${XDG_CONFIG_HOME}/nvm")"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm install --lts
pip install virtualenv
export PATH="$HOME/.local/bin:$PATH"

# download project
git clone https://github.com/JavaRip/UoP-SoftEng-Dissertation
cd UoP-SoftEng-Dissertation

# install project node dependencies
npm i

# create virtual environment
virtualenv venv
source venv/bin/activate

# install project dependencies in virtual environment
pip install -r requirements.txt

# run code
python main.py
```
