use serde::{Deserialize, Serialize};
use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    AppHandle, Manager, RunEvent, State, WindowEvent,
};

#[derive(Clone, Serialize)]
#[serde(rename_all = "snake_case")]
enum ControlPlanePhase { Starting, Online, Offline }

#[derive(Clone, Serialize)]
struct ControlPlaneStatus {
    phase: ControlPlanePhase,
    managed: bool,
    message: String,
    base_url: String,
}

#[derive(Deserialize)]
struct HealthResponse {
    status: String,
    service_id: String,
    instance_nonce: String,
}

#[derive(Clone)]
struct ControlPlane {
    child: Arc<Mutex<Option<Child>>>,
    status: Arc<Mutex<ControlPlaneStatus>>,
}

impl ControlPlane {
    fn new(base_url: String) -> Self {
        Self {
            child: Arc::new(Mutex::new(None)),
            status: Arc::new(Mutex::new(ControlPlaneStatus {
                phase: ControlPlanePhase::Starting,
                managed: false,
                message: "正在连接本地控制平面".into(),
                base_url,
            })),
        }
    }

    fn set_status(&self, phase: ControlPlanePhase, managed: bool, message: &str) {
        if let Ok(mut status) = self.status.lock() {
            *status = ControlPlaneStatus {
                phase,
                managed,
                message: message.into(),
                base_url: status.base_url.clone(),
            };
        }
    }

    fn stop(&self) {
        if let Ok(mut slot) = self.child.lock() {
            if let Some(child) = slot.as_mut() {
                let _ = child.kill();
                let _ = child.wait();
            }
            *slot = None;
        }
    }
}

#[tauri::command]
fn control_plane_status(control_plane: State<'_, ControlPlane>) -> ControlPlaneStatus {
    control_plane.status.lock().map(|value| value.clone()).unwrap_or(ControlPlaneStatus {
        phase: ControlPlanePhase::Offline,
        managed: false,
        message: "无法读取控制平面状态".into(),
        base_url: String::new(),
    })
}

fn health_is_ok(base_url: &str, instance_nonce: &str) -> bool {
    let Some(address) = base_url.strip_prefix("http://") else { return false; };
    let Ok(mut stream) = TcpStream::connect_timeout(
        &address.parse().expect("validated local health address"),
        Duration::from_secs(2),
    ) else { return false; };
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));
    if stream.write_all(
        b"GET /api/v1/health HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nConnection: close\r\n\r\n",
    ).is_err() { return false; }
    let mut response = Vec::new();
    if stream.read_to_end(&mut response).is_err() { return false; }
    let Ok(text) = String::from_utf8(response) else { return false; };
    if !(text.starts_with("HTTP/1.1 200") || text.starts_with("HTTP/1.0 200")) {
        return false;
    }
    let Some((_, body)) = text.split_once("\r\n\r\n") else { return false; };
    let Ok(health) = serde_json::from_str::<HealthResponse>(body) else { return false; };
    health.status == "ok"
        && health.service_id == "io.github.supermzc123.aivideogenerator.control-plane"
        && health.instance_nonce == instance_nonce
}

fn repo_root_from(start: &Path) -> Option<PathBuf> {
    start.ancestors().find(|path| path.join("pyproject.toml").is_file()).map(Path::to_path_buf)
}

fn executable_and_working_dir(app: &AppHandle) -> (PathBuf, Option<PathBuf>) {
    if let Some(value) = std::env::var_os("AIVIDEO_CONTROL_PLANE_EXECUTABLE") {
        return (PathBuf::from(value), None);
    }
    let current = std::env::current_dir().ok();
    if let Some(root) = current.as_deref().and_then(repo_root_from) {
        let candidate = root.join(".venv").join("Scripts").join("aivideo-api.exe");
        if candidate.is_file() { return (candidate, Some(root)); }
    }
    if let Ok(resources) = app.path().resource_dir() {
        let candidate = resources.join("control-plane").join("aivideo-api.exe");
        if candidate.is_file() { return (candidate, Some(resources)); }
    }
    (PathBuf::from("aivideo-api"), current)
}

fn start_control_plane(
    app: &AppHandle,
    control_plane: &ControlPlane,
    port: u16,
    instance_nonce: &str,
) {
    let (executable, working_dir) = executable_and_working_dir(app);
    let mut command = Command::new(executable);
    command.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null());
    command.env("AIVIDEO_API_PORT", port.to_string());
    command.env("AIVIDEO_INSTANCE_NONCE", instance_nonce);
    if let Ok(data_root) = app.path().app_local_data_dir() {
        let data_root = data_root.join("data");
        let _ = std::fs::create_dir_all(&data_root);
        command.env("AIVIDEO_DATA_ROOT", data_root);
    }
    if let Ok(resources) = app.path().resource_dir() {
        let ffmpeg = resources.join("ffmpeg").join("ffmpeg.exe");
        let ffprobe = resources.join("ffmpeg").join("ffprobe.exe");
        if ffmpeg.is_file() { command.env("AIVIDEO_FFMPEG_BINARY", ffmpeg); }
        if ffprobe.is_file() { command.env("AIVIDEO_FFPROBE_BINARY", ffprobe); }
    }
    if let Some(directory) = working_dir { command.current_dir(directory); }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    match command.spawn() {
        Ok(child) => {
            if let Ok(mut slot) = control_plane.child.lock() { *slot = Some(child); }
            control_plane.set_status(ControlPlanePhase::Starting, true, "控制平面正在启动");
            let state = control_plane.clone();
            let base_url = format!("http://127.0.0.1:{port}");
            let nonce = instance_nonce.to_string();
            thread::spawn(move || {
                for _ in 0..30 {
                    if health_is_ok(&base_url, &nonce) {
                        state.set_status(ControlPlanePhase::Online, true, "本地控制平面在线");
                        return;
                    }
                    thread::sleep(Duration::from_millis(500));
                }
                state.set_status(ControlPlanePhase::Offline, true, "控制平面健康检查超时");
            });
        }
        Err(_) => control_plane.set_status(
            ControlPlanePhase::Offline,
            false,
            "无法启动控制平面；请检查安装目录或 AIVIDEO_CONTROL_PLANE_EXECUTABLE",
        ),
    }
}

fn reserve_local_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .unwrap_or(8000)
}

fn instance_nonce() -> String {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{}-{timestamp}", std::process::id())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let port = reserve_local_port();
    let instance_nonce = instance_nonce();
    let control_plane = ControlPlane::new(format!("http://127.0.0.1:{port}"));
    let cleanup = control_plane.clone();
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_shell::init())
        .manage(control_plane.clone())
        .invoke_handler(tauri::generate_handler![control_plane_status])
        .setup(move |app| {
            start_control_plane(app.handle(), &control_plane, port, &instance_nonce);
            let show = MenuItem::with_id(app, "show", "显示窗口", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;
            TrayIconBuilder::new().menu(&menu).on_menu_event(|app, event| match event.id.as_ref() {
                "show" => {
                    if let Some(window) = app.get_webview_window("main") {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                }
                "quit" => app.exit(0),
                _ => {}
            }).build(app)?;
            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let _ = window.hide();
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building the desktop application")
        .run(move |_app, event| {
            if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) { cleanup.stop(); }
        });
}
