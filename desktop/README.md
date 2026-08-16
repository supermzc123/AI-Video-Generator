# Desktop shell

The Tauri shell owns the local control-plane process when it starts it. It first checks
`http://127.0.0.1:8000/api/v1/health`; an already-running service is reused and is not
terminated when the app exits.

In a source checkout the executable is discovered at
`<repo>/.venv/Scripts/aivideo-api.exe`. Packaged builds must place the frozen executable at
`resources/control-plane/aivideo-api.exe`, or set `AIVIDEO_CONTROL_PLANE_EXECUTABLE` to an
absolute executable path. The process receives configuration through its environment. The
desktop shell never passes API keys on the command line and discards child stdout/stderr so
credentials cannot enter UI logs.

Closing the main window hides it to the tray. Choosing **退出** stops only the process owned
by this app and then exits. Project settings are revisioned in local storage; task and review
views use the control-plane task API.
