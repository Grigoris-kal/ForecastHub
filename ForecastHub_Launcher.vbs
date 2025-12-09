Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw ForecastHub.py", 0, True
Set WshShell = Nothing