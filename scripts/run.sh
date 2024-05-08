#!/bin/bash

# Activate Python Virtual Environment
echo "Activating Python Virtual Environment"
source venv/bin/activate

# Starte den Backend-Server
python Flask_Server/FairCR/app.py &

# Starte den Frontend-Entwicklungsserver
cd angular-project
ng serve