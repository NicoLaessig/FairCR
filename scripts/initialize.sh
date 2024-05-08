#!/bin/sh

# creating a python Virtual Environment
echo "Creating Python Virtual Environment"
python -m venv venv

# Activate the Virtual Enviroment
echo "Activating python Virtual Environment"
source venv/bin/activate

# Install the python dependencies
echo "Installing Python dependencies"
pip install -r requirements.txt


# Change to the Angular project directory
echo "Chaning to Angular directory"
cd ./angular-project

# install angular dependencies
echo "Installing Angular dependencies"
npm install

echo "Finished installing all dependencies"
