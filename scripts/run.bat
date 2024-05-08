ECHO on

rem Activate the Virtual Environment
ECHO Activating Python Virtual Environment
call .\venv\Scripts\activate.bat

rem Starting the Backend Server
start python .\Flask_Server\FALCC_New\app.py

rem Starting Frontend Development Server
cd angular-project
ng serve


