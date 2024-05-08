ECHO on

rem Creating a python Virtual Environment
ECHO Creating Python Virtual Environment
setlocal
%@Try%
  python -m venv venv
%@EndTry%
:@Catch
  python3 -m venv venv
:@EndCatch

rem Install the python dependencies
ECHO Installing Python dependencies
.\venv\Scripts\pip.exe install -r requirements.txt

rem Change to the Angular project directory
ECHO Changing to Angular directory
cd angular-project

rem Install Angular dependencies
ECHO Installing Angular dependencies
npm install

ECHO Finished installing all dependencies
