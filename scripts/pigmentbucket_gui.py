#!/usr/bin/env python3
"""Standalone PyQt6 GUI shell for Pigment Bucket."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

try:
    from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, PYQT_VERSION_STR, QT_VERSION_STR
    from PyQt6.QtGui import QAction, QIcon
    from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QStatusBar,
    QTextEdit,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )
except ImportError as import_error:  # pragma: no cover - GUI import guard
    print(
        "PyQt6 is required to run the Pigment Bucket GUI. "
        "Install it with 'pip install PyQt6' and try again.",
        file=sys.stderr,
    )
    raise SystemExit(1) from import_error

SUPPORTED_PYQT_VERSION = "6.6.1"
SUPPORTED_QT_VERSION = "6.6.1"


GUI_DATA_DIR = Path(__file__).resolve().parents[1] / ".cache"
GUI_SETTINGS_PATH = GUI_DATA_DIR / "gui_settings.json"
GUI_TUNING_LOG_PATH = GUI_DATA_DIR / "tuning_log.jsonl"
GUI_STREAM_LOG_PATH = GUI_DATA_DIR / "gui_stream.jsonl"

try:
    from gui_helpers import (
        SENSITIVITY_DEFAULT,
        append_jsonl,
        iso_timestamp,
        load_gui_settings,
        record_auto_profile,
        save_gui_settings,
        sensitivity_to_min_duration,
        sensitivity_to_similarity,
    )
except ImportError:  # pragma: no cover - package-style import
    from .gui_helpers import (  # type: ignore
        SENSITIVITY_DEFAULT,
        append_jsonl,
        iso_timestamp,
        load_gui_settings,
        record_auto_profile,
        save_gui_settings,
        sensitivity_to_min_duration,
        sensitivity_to_similarity,
    )


def ensure_pyqt_compatibility(parent: Optional[QWidget] = None) -> bool:
    """Validate PyQt/Qt versions and surface a friendly hint when mismatched."""

    pyqt_version = PYQT_VERSION_STR
    qt_version = QT_VERSION_STR
    if pyqt_version == SUPPORTED_PYQT_VERSION and qt_version == SUPPORTED_QT_VERSION:
        return True

    message = (
        "Detected PyQt6 "
        f"{pyqt_version} / Qt {qt_version}, but Pigment Bucket expects {SUPPORTED_PYQT_VERSION}.\n"
        "Reinstall dependencies (see docs/run_m0.md — Troubleshooting)")

    if parent is not None:
        QMessageBox.critical(parent, "Pigment Bucket", message)
    else:
        print(message, file=sys.stderr)
    return False


@dataclass
class AnalysisOptions:
    selection_mode: str = "all"
    dry_run: bool = False
    http_timeout: int = 60
    force_min_k: int = 1
    ignore_cache: bool = False
    similarity_threshold: Optional[float] = None
    hysteresis: Optional[bool] = None
    location_min_duration_sec: Optional[float] = None
    location_store: str = "sqlite"
    export_locations_only: bool = False
    add_resolve_markers: bool = False
    report_format: str = "both"
    auto_mode: bool = False
    auto_target_k: Optional[int] = None
    auto_max_iters: Optional[int] = None


@dataclass
class RunnerConfig:
    command: List[str]
    working_dir: Path
    mode: str
    dry_run: bool
    env: Dict[str, str] = field(default_factory=dict)
    http_timeout: int = 30
    allow_mock: bool = False
    force_min_k: int = 1
    ignore_cache: bool = False
    similarity_threshold: Optional[float] = None
    hysteresis: Optional[bool] = None
    location_min_duration_sec: Optional[float] = None
    location_store: str = "sqlite"
    export_locations_only: bool = False
    add_resolve_markers: bool = False
    report_format: str = "both"
    auto_mode: bool = False
    auto_target_k: Optional[int] = None
    auto_max_iters: Optional[int] = None


class RunnerWorker(QObject):
    """Qt-friendly worker that executes the Pigment Bucket runner script."""

    progress = pyqtSignal(str)
    log_line = pyqtSignal(str)
    failed = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, config: RunnerConfig, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self.config = config
        self.stop_flag = False
        self._process: Optional[subprocess.Popen[str]] = None
        self._brace_depth = 0
        self._collecting_json = False
        self._json_lines: List[str] = []
        self._last_summary: Dict[str, object] = {}
        self._job_id: Optional[str] = None
        self._run_id: Optional[str] = None
        self._last_error: Optional[str] = None

    def run(self) -> None:  # pragma: no cover - threaded flow
        if not self.config.command and self.config.allow_mock:
            self._run_mock()
            return

        try:
            self.progress.emit("Launching Pigment Bucket runner…")
            self._process = subprocess.Popen(
                self.config.command,
                cwd=str(self.config.working_dir),
                env=self.config.env or None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as error:
            self.failed.emit(f"Failed to start runner: {error}")
            return

        assert self._process.stdout is not None
        for raw_line in self._process.stdout:
            if self.stop_flag and self._process.poll() is None:
                self._terminate_process()
            line = raw_line.rstrip()
            if not line:
                continue
            self._handle_output_line(line)
            if self.stop_flag and self._process.poll() is None:
                continue

        exit_code = self._process.wait()
        if self.stop_flag:
            self.failed.emit("Runner terminated by user")
            return
        if exit_code != 0:
            message = self._last_error or f"Runner exited with code {exit_code}"
            self.failed.emit(message)
            return

        summary_payload: Dict[str, object] = dict(self._last_summary)
        if self._job_id:
            summary_payload.setdefault("job_id", self._job_id)
        if self._run_id:
            summary_payload.setdefault("run_id", self._run_id)
        summary_payload.setdefault("mode", self.config.mode)
        summary_payload.setdefault("dry_run", self.config.dry_run)
        summary_payload.setdefault("force_min_k", self.config.force_min_k)
        summary_payload.setdefault("ignore_cache", self.config.ignore_cache)
        summary_payload.setdefault("similarity_threshold", self.config.similarity_threshold)
        summary_payload.setdefault("hysteresis", self.config.hysteresis)
        summary_payload.setdefault("location_store", self.config.location_store)
        summary_payload.setdefault("location_min_duration_sec", self.config.location_min_duration_sec)
        summary_payload.setdefault("export_locations_only", self.config.export_locations_only)
        summary_payload.setdefault("add_resolve_markers", self.config.add_resolve_markers)
        summary_payload.setdefault("report_format", self.config.report_format)
        summary_payload.setdefault("auto_mode", self.config.auto_mode)
        summary_payload.setdefault("auto_target_k", self.config.auto_target_k)
        summary_payload.setdefault("auto_max_iters", self.config.auto_max_iters)
        self.finished.emit(summary_payload)

    def stop(self) -> None:
        self.stop_flag = True
        self._terminate_process()

    def _run_mock(self) -> None:
        steps = 6
        for index in range(steps):
            if self.stop_flag:
                self.failed.emit("Mock analysis cancelled by user")
                return
            self.progress.emit(f"Mock analysis step {index + 1}/{steps}")
            QThread.msleep(400)
        summary_payload = {
            "processed": 3,
            "skipped": 0,
            "clusters": 1,
            "feature_dim": 42,
            "chosen_k": 1,
            "silhouette": None,
            "sampled_frames": 3,
            "mode": self.config.mode,
            "dry_run": self.config.dry_run,
        }
        self.finished.emit(summary_payload)

    def _terminate_process(self) -> None:
        if not self._process:
            return
        if self._process.poll() is not None:
            return
        self.log_line.emit("[INFO] Terminating runner process…")
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.log_line.emit("[WARN] Runner did not exit gracefully; killing…")
            self._process.kill()

    def _handle_output_line(self, line: str) -> None:
        lower_line = line.lower()
        if "timeout" in lower_line and "retry" in lower_line:
            self.log_line.emit("[WARN] timeout, retrying…")
        else:
            self.log_line.emit(line)

        if lower_line.startswith("error") or "failed" in lower_line:
            self._record_error(line)

        if "job id:" in lower_line and "run_id" in lower_line:
            job_part, _, run_part = line.partition("run_id:")
            job_token = job_part.split(":")[-1].strip()
            if job_token:
                job_split = job_token.split()
                self._job_id = job_split[-1] if job_split else job_token
            run_token = run_part.strip().split()[0] if run_part.strip() else ""
            if run_token:
                self._run_id = run_token

        stripped = line.strip()
        if not stripped:
            return
        if not self._collecting_json and stripped.startswith("{"):
            self._collecting_json = True
            self._brace_depth = self._brace_delta(stripped)
            self._json_lines = [line]
            if self._brace_depth == 0:
                self._finalize_json_buffer()
            return

        if self._collecting_json:
            self._json_lines.append(line)
            self._brace_depth += self._brace_delta(stripped)
            if self._brace_depth <= 0:
                self._finalize_json_buffer()

    def _brace_delta(self, text: str) -> int:
        return text.count("{") - text.count("}")

    def _finalize_json_buffer(self) -> None:
        payload_text = "\n".join(self._json_lines)
        self._collecting_json = False
        self._json_lines = []
        self._brace_depth = 0
        try:
            data = json.loads(payload_text)
        except json.JSONDecodeError:
            return
        if isinstance(data, dict):
            self._last_summary = data
            if "job_id" in data:
                self._job_id = str(data["job_id"])
            if "run_id" in data:
                self._run_id = str(data["run_id"])
            self.progress.emit("Received summary from runner")

    def _record_error(self, line: str) -> None:
        text = line.strip()
        if text.startswith("[") and "]" in text:
            closing = text.find("]")
            if closing >= 0:
                text = text[closing + 1 :].strip() or text
        if text.lower().startswith("error:"):
            text = text[6:].strip() or text
        if not text:
            text = line.strip() or line
        self._last_error = text


class AnalysisController(QObject):
    """Controller façade abstracting the actual pipeline invocation."""

    log_message = pyqtSignal(str)
    analysis_started = pyqtSignal()
    analysis_finished = pyqtSignal(bool, str, dict)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[RunnerWorker] = None
        self._root_dir = Path(__file__).resolve().parents[1]
        self._http_timeout = int(os.getenv("PIGMENT_HTTP_TIMEOUT", "60"))
        self._current_mode: Optional[str] = None
        self._previous_mode: Optional[str] = None
        self._current_options: Optional[AnalysisOptions] = None
        self._last_run_id: Optional[str] = None
        self._last_run_dry_run = False
        self._undo_run_id: Optional[str] = None
        self._last_summary: Dict[str, object] = {}

    def start_analysis(self, options: AnalysisOptions) -> None:
        if self.is_running:
            self.log_message.emit("Analysis already in progress")
            return
        self._current_mode = "analysis"
        self._current_options = options
        command, env, allow_mock = self._build_analysis_command(options)
        self._http_timeout = options.http_timeout
        config = RunnerConfig(
            command=command,
            working_dir=self._root_dir,
            mode="analysis",
            dry_run=options.dry_run or options.export_locations_only,
            env=env,
            http_timeout=options.http_timeout,
            allow_mock=allow_mock,
            force_min_k=options.force_min_k,
            ignore_cache=options.ignore_cache,
            similarity_threshold=options.similarity_threshold,
            hysteresis=options.hysteresis,
            location_min_duration_sec=options.location_min_duration_sec,
            location_store=(options.location_store or "sqlite").lower(),
            export_locations_only=options.export_locations_only,
            add_resolve_markers=options.add_resolve_markers,
            report_format=options.report_format,
            auto_mode=options.auto_mode,
            auto_target_k=options.auto_target_k,
            auto_max_iters=options.auto_max_iters,
        )
        self._launch_worker(config)

    def stop_analysis(self) -> None:
        if not self._worker:
            self.log_message.emit("No active analysis to stop")
            return
        if self._current_mode != "analysis" and self._current_mode != "undo":
            self.log_message.emit("No active analysis to stop")
            return
        if self._worker and not self._worker.stop_flag:
            self.log_message.emit("Stopping analysis…")
            self._worker.stop()

    def _launch_worker(self, config: RunnerConfig) -> None:
        self._last_summary = {}
        runner_path = self._root_dir / "scripts" / "pigmentbucket_run.py"
        use_mock = os.environ.get("PIGMENT_UI_FORCE_MOCK") == "1" or not runner_path.exists()
        config.allow_mock = config.allow_mock or (use_mock and not config.command)
        if not config.allow_mock and not config.command:
            self._previous_mode = self._current_mode
            self.log_message.emit("Runner command unavailable; aborting request")
            self.analysis_finished.emit(False, "Runner command unavailable", {})
            self._reset_state()
            return
        thread = QThread()
        worker = RunnerWorker(config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        worker.log_line.connect(self._handle_worker_log)
        worker.progress.connect(self._handle_worker_progress)
        worker.finished.connect(self._handle_worker_finished)
        worker.failed.connect(self._handle_worker_failed)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        self._worker = worker
        thread.start()
        self.analysis_started.emit()

    def start_undo(self) -> None:
        if not self._last_run_id:
            self.log_message.emit("No run-id available for undo")
            return
        if self.is_running:
            self.log_message.emit("Operation already in progress")
            return
        command, env, allow_mock = self._build_undo_command()
        if allow_mock:
            return
        if not command:
            return
        config = RunnerConfig(
            command=command,
            working_dir=self._root_dir,
            mode="undo",
            dry_run=False,
            env=env,
            http_timeout=self._http_timeout,
            allow_mock=False,
        )
        self._current_mode = "undo"
        self._launch_worker(config)

    def _build_analysis_command(self, options: AnalysisOptions) -> tuple[List[str], Dict[str, str], bool]:
        runner_path = self._root_dir / "scripts" / "pigmentbucket_run.py"
        use_mock = os.environ.get("PIGMENT_UI_FORCE_MOCK") == "1" or not runner_path.exists()
        if use_mock:
            self.log_message.emit("Runner script not found; using mock mode")
            return [], {}, True
        command: List[str] = [
            sys.executable,
            str(runner_path),
            "--selection",
            options.selection_mode,
            "--report-dir",
            str(self._root_dir / "logs"),
            "--http-timeout",
            str(options.http_timeout),
        ]
        if options.dry_run:
            command.append("--dry-run")
        if options.force_min_k and options.force_min_k > 1:
            command.extend(["--force-min-k", str(options.force_min_k)])
        if options.ignore_cache:
            command.append("--ignore-cache")
        if options.location_min_duration_sec is not None:
            command.extend(["--min-duration-sec", str(options.location_min_duration_sec)])
        if options.similarity_threshold is not None:
            command.extend(["--similarity-threshold", str(options.similarity_threshold)])
        if options.hysteresis is not None:
            command.append("--hysteresis" if options.hysteresis else "--no-hysteresis")
        store_choice = (options.location_store or "sqlite").lower()
        if store_choice:
            command.extend(["--locations-store", store_choice])
        report_choice = (options.report_format or "both").lower()
        if report_choice:
            command.extend(["--report-format", report_choice])
        if options.export_locations_only:
            command.append("--export-locations-only")
        if options.add_resolve_markers:
            command.append("--add-resolve-markers")
        if options.auto_mode:
            command.append("--auto-mode")
        if options.auto_target_k and options.auto_target_k > 1:
            command.extend(["--auto-target-k", str(options.auto_target_k)])
        if options.auto_max_iters and options.auto_max_iters > 0:
            command.extend(["--auto-max-iters", str(options.auto_max_iters)])
        env = self._build_env()
        return command, env, False

    def _build_undo_command(self) -> tuple[List[str], Dict[str, str], bool]:
        runner_path = self._root_dir / "scripts" / "pigmentbucket_run.py"
        use_mock = os.environ.get("PIGMENT_UI_FORCE_MOCK") == "1" or not runner_path.exists()
        if use_mock:
            self.log_message.emit("Undo is unavailable in mock mode")
            return [], {}, True
        command = [
            sys.executable,
            str(runner_path),
            "--undo",
            self._last_run_id or "",
            "--http-timeout",
            str(self._http_timeout),
        ]
        env = self._build_env()
        return command, env, False

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        pythonpath = env.get("PYTHONPATH", "")
        root_str = str(self._root_dir)
        if root_str not in pythonpath.split(os.pathsep):
            env["PYTHONPATH"] = os.pathsep.join([p for p in [root_str, pythonpath] if p])
        return env

    def _handle_worker_log(self, message: str) -> None:
        self.log_message.emit(message)

    def _handle_worker_progress(self, message: str) -> None:
        self.log_message.emit(f"[INFO] {message}")

    def _handle_worker_finished(self, summary: dict) -> None:
        self._previous_mode = self._current_mode
        self._last_summary = summary or {}
        if self._current_mode == "analysis":
            if self._current_options:
                self._last_run_dry_run = bool(
                    self._current_options.dry_run or self._current_options.export_locations_only
                )
            run_id = summary.get("run_id") if isinstance(summary, dict) else None
            if run_id and not self._last_run_dry_run:
                self._last_run_id = str(run_id)
                self._undo_run_id = str(run_id)
            self._latest_run_id = run_id if run_id else None
        else:
            self._latest_run_id = None
        self._latest_summary_payload = dict(summary or {})
        if self._latest_run_id:
            self._record_tuning_entry(self._latest_run_id, self._latest_summary_payload, accepted=None)
        success_message = "Runner completed successfully"
        self.analysis_finished.emit(True, success_message, self._last_summary)
        self._reset_state()

    def _handle_worker_failed(self, message: str) -> None:
        self._previous_mode = self._current_mode
        if self._current_mode == "analysis":
            if self._current_options:
                self._last_run_dry_run = bool(
                    self._current_options.dry_run or self._current_options.export_locations_only
                )
            if not self._last_run_dry_run:
                self._last_run_id = None
                self._undo_run_id = None
        self._last_summary["error"] = message
        self.analysis_finished.emit(False, message, self._last_summary)
        self._reset_state()

    def _on_thread_finished(self) -> None:
        # Ensure references get cleared when the thread stops naturally.
        if self._thread and not self._thread.isRunning():
            self._thread = None

    def _reset_state(self) -> None:
        self._current_mode = None
        self._current_options = None
        self._worker = None
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(100)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.isRunning())

    @property
    def last_run_id(self) -> Optional[str]:
        return self._undo_run_id

    @property
    def last_run_was_dry_run(self) -> bool:
        return self._last_run_dry_run

    @property
    def current_mode(self) -> Optional[str]:
        return self._current_mode

    @property
    def previous_mode(self) -> Optional[str]:
        return self._previous_mode

    @property
    def last_summary(self) -> Dict[str, object]:
        return dict(self._last_summary)

    @property
    def http_timeout(self) -> int:
        return self._http_timeout


class MainWindow(QMainWindow):
    """Main window for Pigment Bucket PyQt UI."""

    def __init__(self, controller: AnalysisController) -> None:
        super().__init__()
        self._controller = controller
        self._latest_summary: Dict[str, object] = {}
        self._gui_settings = load_gui_settings(GUI_SETTINGS_PATH)
        self._current_sensitivity = int(self._gui_settings.get("sensitivity", SENSITIVITY_DEFAULT))
        self._current_sensitivity = max(0, min(100, self._current_sensitivity))
        self._auto_mode_default = bool(self._gui_settings.get("auto_mode", False))
        self._latest_run_id: Optional[str] = None
        self._latest_summary_payload: Dict[str, object] = {}
        self._advanced_similarity_override = False
        self._advanced_duration_override = False
        self._last_tuning_metric: Optional[float] = None
        self._last_tuning_metric_name: Optional[str] = None
        self._auto_last_similarity: Optional[float] = None
        self._auto_last_min_duration: Optional[float] = None
        self._setup_ui()
        self.auto_mode_checkbox.setChecked(self._auto_mode_default)
        self._connect_signals()
        self._update_auto_controls()

    def _setup_ui(self) -> None:
        self.setWindowTitle("Pigment Bucket")
        self.resize(640, 480)

        toolbar = QToolBar("Main Toolbar", self)
        self.addToolBar(toolbar)

        self.analyze_action = QAction(QIcon(), "Analyze", self)
        self.stop_action = QAction(QIcon(), "Stop", self)

        toolbar.addAction(self.analyze_action)
        toolbar.addAction(self.stop_action)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        button_row = QHBoxLayout()
        self.analyze_button = QPushButton("Analyze", central_widget)
        self.stop_button = QPushButton("Stop", central_widget)
        self.stop_button.setEnabled(False)
        self.undo_button = QPushButton("Undo Last", central_widget)
        self.undo_button.setEnabled(False)
        self.dry_run_checkbox = QCheckBox("Dry run", central_widget)
        self.dry_run_checkbox.setChecked(True)
        self.force_min_k_checkbox = QCheckBox("Force min clusters", central_widget)
        self.force_min_k_checkbox.setToolTip("Force at least two clusters when analysis returns a single cluster.")
        self.ignore_cache_checkbox = QCheckBox("Ignore cache", central_widget)
        self.ignore_cache_checkbox.setToolTip("Recompute features even if cached; slower but safer for fresh media.")
        timeout_label = QLabel("HTTP timeout (s):", central_widget)
        self.timeout_spin = QSpinBox(central_widget)
        self.timeout_spin.setRange(5, 600)
        self.timeout_spin.setValue(max(60, self._controller.http_timeout))
        self.timeout_spin.setSingleStep(5)
        button_row.addWidget(self.analyze_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.undo_button)
        button_row.addWidget(self.dry_run_checkbox)
        button_row.addWidget(self.force_min_k_checkbox)
        button_row.addWidget(self.ignore_cache_checkbox)
        button_row.addWidget(timeout_label)
        button_row.addWidget(self.timeout_spin)
        button_row.addStretch()

        slider_row = QHBoxLayout()
        self.preset_combo = QComboBox(central_widget)
        self.preset_combo.addItems(["Custom", "Loose", "Default", "Strict"])
        slider_row.addWidget(QLabel("Preset:", central_widget))
        slider_row.addWidget(self.preset_combo)

        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal, central_widget)
        self.sensitivity_slider.setRange(0, 100)
        self.sensitivity_slider.setValue(self._current_sensitivity)
        self.sensitivity_slider.setToolTip("Lower = merges more, Higher = splits more")
        slider_row.addWidget(QLabel("Sensitivity:", central_widget))
        slider_row.addWidget(self.sensitivity_slider)

        self.sensitivity_value_label = QLabel("", central_widget)
        slider_row.addWidget(self.sensitivity_value_label)
        self.auto_mode_checkbox = QCheckBox("Auto", central_widget)
        self.auto_mode_checkbox.setToolTip("Enable automatic tuning of similarity and min duration")
        slider_row.addWidget(self.auto_mode_checkbox)
        slider_row.addStretch()

        location_row = QHBoxLayout()
        store_label = QLabel("Store:", central_widget)
        self.location_store_combo = QComboBox(central_widget)
        self.location_store_combo.addItems(["sqlite", "json"])
        self.location_store_combo.setCurrentText("sqlite")
        self.location_store_combo.setToolTip("Location persistence backend (sqlite recommended)")
        report_format_label = QLabel("Report:", central_widget)
        self.report_format_combo = QComboBox(central_widget)
        self.report_format_combo.addItems(["both", "json", "csv"])
        self.report_format_combo.setCurrentText("both")
        self.report_format_combo.setToolTip("Server-side report format (json, csv, or both)")
        min_duration_label = QLabel("Min duration (s):", central_widget)
        self.min_duration_spin = QDoubleSpinBox(central_widget)
        self.min_duration_spin.setRange(0.0, 120.0)
        self.min_duration_spin.setDecimals(2)
        self.min_duration_spin.setSingleStep(0.5)
        self.min_duration_spin.setValue(3.0)
        self.min_duration_spin.setToolTip("Minimum length of a standalone location segment")
        similarity_label = QLabel("Similarity:", central_widget)
        self.similarity_spin = QDoubleSpinBox(central_widget)
        self.similarity_spin.setRange(0.0, 1.0)
        self.similarity_spin.setDecimals(2)
        self.similarity_spin.setSingleStep(0.01)
        self.similarity_spin.setValue(0.92)
        self.similarity_spin.setToolTip("Cosine similarity threshold for adjacent locations")
        self.hysteresis_checkbox = QCheckBox("Hysteresis", central_widget)
        self.hysteresis_checkbox.setChecked(True)
        self.hysteresis_checkbox.setToolTip("Absorb short A–B–A segments back into surrounding locations")
        self.export_locations_checkbox = QCheckBox("Export locations only", central_widget)
        self.export_locations_checkbox.setToolTip("Skip Resolve updates; only produce reports")
        self.add_markers_checkbox = QCheckBox("Add markers", central_widget)
        self.add_markers_checkbox.setToolTip("Add timeline markers for each detected location")
        location_row.addWidget(store_label)
        location_row.addWidget(self.location_store_combo)
        location_row.addWidget(report_format_label)
        location_row.addWidget(self.report_format_combo)
        location_row.addWidget(min_duration_label)
        location_row.addWidget(self.min_duration_spin)
        location_row.addWidget(similarity_label)
        location_row.addWidget(self.similarity_spin)
        location_row.addWidget(self.hysteresis_checkbox)
        location_row.addWidget(self.export_locations_checkbox)
        location_row.addWidget(self.add_markers_checkbox)
        location_row.addStretch()

        status_row = QHBoxLayout()
        self.status_label = QLabel("Idle", central_widget)
        status_row.addWidget(QLabel("Status:", central_widget))
        status_row.addWidget(self.status_label)
        status_row.addStretch()

        self.log_view = QTextEdit(central_widget)
        self.log_view.setReadOnly(True)

        action_row = QHBoxLayout()
        self.clear_log_button = QPushButton("Clear log", central_widget)
        self.accept_button = QPushButton("Accept result", central_widget)
        self.reject_button = QPushButton("Reject result", central_widget)
        self.accept_button.setEnabled(False)
        self.reject_button.setEnabled(False)
        action_row.addWidget(self.clear_log_button)
        action_row.addStretch()
        action_row.addWidget(self.accept_button)
        action_row.addWidget(self.reject_button)

        layout.addLayout(button_row)
        layout.addLayout(slider_row)
        layout.addLayout(location_row)
        layout.addLayout(status_row)
        layout.addWidget(self.log_view)
        layout.addLayout(action_row)

        self.setCentralWidget(central_widget)

        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        self._update_sensitivity_labels()
        self._update_preset_from_slider()

    def _connect_signals(self) -> None:
        self.analyze_button.clicked.connect(self._handle_analyze_clicked)
        self.analyze_action.triggered.connect(self._handle_analyze_clicked)
        self.stop_button.clicked.connect(self._handle_stop_clicked)
        self.stop_action.triggered.connect(self._handle_stop_clicked)
        self.undo_button.clicked.connect(self._handle_undo_clicked)
        self.clear_log_button.clicked.connect(self._handle_clear_log_clicked)
        self.accept_button.clicked.connect(lambda: self._handle_feedback_clicked(True))
        self.reject_button.clicked.connect(lambda: self._handle_feedback_clicked(False))
        self.sensitivity_slider.valueChanged.connect(self._handle_sensitivity_changed)
        self.preset_combo.currentIndexChanged.connect(self._handle_preset_changed)
        self.auto_mode_checkbox.stateChanged.connect(self._handle_auto_mode_changed)
        self.similarity_spin.valueChanged.connect(self._handle_similarity_override)
        self.min_duration_spin.valueChanged.connect(self._handle_min_duration_override)

        self._controller.log_message.connect(self._handle_controller_log)
        self._controller.analysis_started.connect(self._on_analysis_started)
        self._controller.analysis_finished.connect(self._on_analysis_finished)

    def _handle_analyze_clicked(self) -> None:
        if self._controller.is_running:
            return
        slider_similarity, slider_min_duration = self._derive_sensitivity_values()
        final_similarity = (
            self.similarity_spin.value()
            if self._advanced_similarity_override
            else slider_similarity
        )
        final_min_duration = (
            self.min_duration_spin.value()
            if self._advanced_duration_override
            else slider_min_duration
        )
        options = AnalysisOptions(
            selection_mode="all",
            dry_run=self.dry_run_checkbox.isChecked(),
            http_timeout=self.timeout_spin.value(),
            force_min_k=2 if self.force_min_k_checkbox.isChecked() else 1,
            ignore_cache=self.ignore_cache_checkbox.isChecked(),
            location_min_duration_sec=final_min_duration,
            similarity_threshold=final_similarity,
            hysteresis=self.hysteresis_checkbox.isChecked(),
            location_store=self.location_store_combo.currentText(),
            export_locations_only=self.export_locations_checkbox.isChecked(),
            add_resolve_markers=self.add_markers_checkbox.isChecked(),
            report_format=self.report_format_combo.currentText(),
            auto_mode=self.auto_mode_checkbox.isChecked(),
        )
        self._controller.start_analysis(options)

    def _handle_stop_clicked(self) -> None:
        self._controller.stop_analysis()

    def _handle_undo_clicked(self) -> None:
        self._controller.start_undo()

    def _on_analysis_started(self) -> None:
        mode = self._controller.current_mode or "analysis"
        self.status_label.setText("Undoing…" if mode == "undo" else "Running…")
        self.analyze_button.setEnabled(False)
        self.analyze_action.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.stop_action.setEnabled(True)
        self.undo_button.setEnabled(False)
        self.dry_run_checkbox.setEnabled(False)
        self.timeout_spin.setEnabled(False)
        self.force_min_k_checkbox.setEnabled(False)
        self.ignore_cache_checkbox.setEnabled(False)
        self.location_store_combo.setEnabled(False)
        self.min_duration_spin.setEnabled(False)
        self.similarity_spin.setEnabled(False)
        self.hysteresis_checkbox.setEnabled(False)
        self.export_locations_checkbox.setEnabled(False)
        self.add_markers_checkbox.setEnabled(False)
        self.report_format_combo.setEnabled(False)
        self.sensitivity_slider.setEnabled(False)
        self.preset_combo.setEnabled(False)
        self.auto_mode_checkbox.setEnabled(False)
        self.accept_button.setEnabled(False)
        self.reject_button.setEnabled(False)
        self._latest_run_id = None
        self._latest_summary_payload = {}
        if self.auto_mode_checkbox.isChecked():
            self._auto_last_similarity = None
            self._auto_last_min_duration = None
        label = "Undo" if mode == "undo" else "Analysis"
        self._append_log_with_level("INFO", f"{label} started")
        self._update_auto_controls()

    def _on_analysis_finished(self, success: bool, message: str, summary: dict) -> None:
        self._latest_summary = summary or {}
        self.status_label.setText("Idle" if success else "Error")
        self.analyze_button.setEnabled(True)
        self.analyze_action.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.stop_action.setEnabled(False)
        self.dry_run_checkbox.setEnabled(True)
        self.timeout_spin.setEnabled(True)
        self.force_min_k_checkbox.setEnabled(True)
        self.ignore_cache_checkbox.setEnabled(True)
        self.location_store_combo.setEnabled(True)
        self.min_duration_spin.setEnabled(True)
        self.similarity_spin.setEnabled(True)
        self.hysteresis_checkbox.setEnabled(True)
        self.export_locations_checkbox.setEnabled(True)
        self.add_markers_checkbox.setEnabled(True)
        self.report_format_combo.setEnabled(True)
        self.sensitivity_slider.setEnabled(True)
        self.preset_combo.setEnabled(True)
        self.auto_mode_checkbox.setEnabled(True)
        self._update_auto_controls()
        allow_feedback = bool(self._latest_run_id)
        self.accept_button.setEnabled(allow_feedback)
        self.reject_button.setEnabled(allow_feedback)
        mode = self._controller.previous_mode
        enable_undo = False
        if mode == "analysis" and self._controller.last_run_id:
            enable_undo = not self._controller.last_run_was_dry_run
        elif mode == "undo" and self._controller.last_run_id:
            enable_undo = True
        self.undo_button.setEnabled(enable_undo)
        label = "Undo" if mode == "undo" else "Analysis"
        hint = ""
        lowered = (message or "").lower()
        is_cancelled = (not success) and "terminated by user" in lowered
        if success or is_cancelled:
            if success:
                self._append_log_with_level("INFO", f"{label} finished: {message}")
                self._display_summary(self._latest_summary)
            else:
                self.status_label.setText("Cancelled")
                self._append_log_with_level("WARN", f"{label} cancelled: {message}")
        else:
            if "timeout" in lowered or "failed to submit analysis" in lowered:
                hint = "\nConsider increasing the HTTP timeout or verifying the service is reachable."
            self._append_log_with_level("ERROR", f"{label} failed: {message}{hint}")
            QMessageBox.critical(self, "Pigment Bucket", f"{label} failed: {message}{hint}")
        if enable_undo and self._controller.last_run_id:
            self.statusBar().showMessage(f"Last run-id: {self._controller.last_run_id}", 5000)

    def _handle_controller_log(self, message: str) -> None:
        self._append_log_entry(message)

    def _handle_sensitivity_changed(self, value: int) -> None:
        self._current_sensitivity = int(value)
        self._advanced_similarity_override = False
        self._advanced_duration_override = False
        self._update_sensitivity_labels()
        self._update_preset_from_slider()
        self._save_gui_settings()

    def _handle_preset_changed(self, index: int) -> None:
        presets = {
            1: 35,
            2: 60,
            3: 85,
        }
        if index in presets:
            self.sensitivity_slider.blockSignals(True)
            self.sensitivity_slider.setValue(presets[index])
            self.sensitivity_slider.blockSignals(False)
            self._handle_sensitivity_changed(presets[index])

    def _handle_auto_mode_changed(self, state: int) -> None:
        if self.auto_mode_checkbox.isChecked():
            self._advanced_similarity_override = False
            self._advanced_duration_override = False
        self._save_gui_settings()
        self._update_auto_controls()

    def _handle_clear_log_clicked(self) -> None:
        self.log_view.clear()

    def _handle_feedback_clicked(self, accepted: bool) -> None:
        if not self._latest_run_id:
            return
        self._record_tuning_entry(self._latest_run_id, self._latest_summary_payload, accepted=accepted)
        self.accept_button.setEnabled(False)
        self.reject_button.setEnabled(False)

    def _handle_similarity_override(self, value: float) -> None:
        slider_similarity, _ = self._derive_sensitivity_values()
        if abs(value - slider_similarity) > 1e-4:
            self._advanced_similarity_override = True
        self._update_override_indicator()

    def _handle_min_duration_override(self, value: float) -> None:
        _, slider_min = self._derive_sensitivity_values()
        if abs(value - slider_min) > 1e-4:
            self._advanced_duration_override = True
        self._update_override_indicator()

    def _derive_sensitivity_values(self) -> tuple[float, float]:
        similarity = sensitivity_to_similarity(self._current_sensitivity)
        min_duration = sensitivity_to_min_duration(self._current_sensitivity)
        return similarity, min_duration

    def _update_sensitivity_labels(self) -> None:
        if self.auto_mode_checkbox.isChecked():
            if (
                self._auto_last_similarity is not None
                and self._auto_last_min_duration is not None
            ):
                self.sensitivity_value_label.setText(
                    f"auto sim={self._auto_last_similarity:.3f}  min_dur={self._auto_last_min_duration:.1f}s"
                )
            else:
                self.sensitivity_value_label.setText("auto sim=—  min_dur=—")
            self._update_override_indicator()
            return
        similarity, min_duration = self._derive_sensitivity_values()
        self.sensitivity_value_label.setText(
            f"sim={similarity:.3f}  min_dur={min_duration:.1f}s"
        )
        self._update_override_indicator()

    def _update_preset_from_slider(self) -> None:
        value = self._current_sensitivity
        preset_mapping = {35: 1, 60: 2, 85: 3}
        index = preset_mapping.get(value, 0)
        if self.preset_combo.currentIndex() != index:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentIndex(index)
            self.preset_combo.blockSignals(False)

    def _save_gui_settings(self) -> None:
        self._gui_settings["sensitivity"] = int(self._current_sensitivity)
        self._gui_settings["auto_mode"] = bool(self.auto_mode_checkbox.isChecked())
        save_gui_settings(GUI_SETTINGS_PATH, self._gui_settings)

    def _update_override_indicator(self) -> None:
        if self.auto_mode_checkbox.isChecked():
            sim = (
                self._auto_last_similarity
                if self._auto_last_similarity is not None
                else sensitivity_to_similarity(self._current_sensitivity)
            )
            min_dur = (
                self._auto_last_min_duration
                if self._auto_last_min_duration is not None
                else sensitivity_to_min_duration(self._current_sensitivity)
            )
            self.statusBar().showMessage(
                f"Location tuning: Auto — sim={sim:.3f}, min_dur={min_dur:.1f}s",
                4000,
            )
            return
        similarity, min_duration = self._derive_sensitivity_values()
        active_similarity = (
            self.similarity_spin.value()
            if self._advanced_similarity_override
            else similarity
        )
        active_min = (
            self.min_duration_spin.value()
            if self._advanced_duration_override
            else min_duration
        )
        status = "Slider"
        if self._advanced_similarity_override or self._advanced_duration_override:
            status = "Advanced"
        self.statusBar().showMessage(
            f"Location tuning: {status} — sim={active_similarity:.3f}, min_dur={active_min:.1f}s",
            4000,
        )

    def _update_auto_controls(self) -> None:
        auto_active = self.auto_mode_checkbox.isChecked()
        manual_enabled = not auto_active and not self._controller.is_running
        self.sensitivity_slider.setEnabled(not auto_active and not self._controller.is_running)
        self.preset_combo.setEnabled(not auto_active and not self._controller.is_running)
        self.similarity_spin.setEnabled(manual_enabled)
        self.min_duration_spin.setEnabled(manual_enabled)
        self._update_sensitivity_labels()

    def _record_tuning_entry(
        self,
        run_id: str,
        summary: Dict[str, object],
        accepted: Optional[bool] = None,
    ) -> None:
        slider_similarity, slider_min_duration = self._derive_sensitivity_values()
        similarity = (
            self.similarity_spin.value()
            if self._advanced_similarity_override
            else slider_similarity
        )
        min_duration = (
            self.min_duration_spin.value()
            if self._advanced_duration_override
            else slider_min_duration
        )
        if summary.get("auto_mode"):
            auto_sim = summary.get("auto_similarity_threshold")
            auto_min = summary.get("auto_min_duration_sec")
            if auto_sim is not None:
                similarity = float(auto_sim)
            if auto_min is not None:
                min_duration = float(auto_min)
        entry = {
            "ts": iso_timestamp(),
            "run_id": run_id,
            "S": int(self._current_sensitivity),
            "similarity": round(float(similarity), 4),
            "min_dur": round(float(min_duration), 2),
            "hysteresis": bool(self.hysteresis_checkbox.isChecked()),
            "clips": int(summary.get("processed", 0) or 0),
            "clusters": int(summary.get("clusters", 0) or 0),
            "cache_hits": summary.get("cache_hits"),
            "quality_metric_name": summary.get("quality_metric_name"),
            "quality_metric_value": summary.get("quality_metric_value"),
            "accepted": accepted,
        }
        entry["auto_mode"] = bool(summary.get("auto_mode"))
        entry["auto_iterations"] = summary.get("auto_iterations")
        append_jsonl(GUI_TUNING_LOG_PATH, entry, max_lines=2000)
        if accepted and summary.get("auto_mode"):
            project_id = summary.get("project_id")
            clip_hash = summary.get("clip_context_hash")
            centroid = summary.get("auto_reference_centroid") or []
            if project_id and clip_hash and centroid:
                record_auto_profile(
                    str(project_id),
                    str(clip_hash),
                    similarity,
                    min_duration,
                    [float(value) for value in centroid if isinstance(value, (int, float))],
                    entry["ts"],
                )

    def _append_log_with_level(self, level: str, message: str) -> None:
        entry = f"[{level.upper()}] {message}"
        self._append_log_entry(entry)

    def _append_log_entry(self, message: str) -> None:
        level, text = self._split_level_and_text(message)
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"{timestamp} {level}: {text}"
        self.log_view.append(formatted)
        self.statusBar().showMessage(text, 2000)
        append_jsonl(
            GUI_STREAM_LOG_PATH,
            {
                "ts": iso_timestamp(),
                "level": level,
                "message": text,
            },
            max_lines=2000,
        )

    def _split_level_and_text(self, message: str) -> tuple[str, str]:
        level = "INFO"
        text = message.strip()
        if message.startswith("[") and "]" in message:
            closing = message.find("]")
            if closing > 1:
                level = message[1:closing].strip().upper() or level
                text = message[closing + 1 :].strip() or text
        return level, text

    def _display_summary(self, summary: Dict[str, object]) -> None:
        if not summary:
            return
        fields = [
            ("job_id", "job_id"),
            ("run_id", "run_id"),
            ("processed", "processed"),
            ("skipped", "skipped"),
            ("clusters", "clusters"),
            ("locations", "locations"),
            ("locations_persisted", "locations_persisted"),
            ("matched_locations", "matched_locations"),
            ("new_locations", "new_locations"),
            ("location_store", "location_store"),
            ("report_format", "report_format"),
            ("quality_metric_name", "quality_metric_name"),
            ("quality_metric_value", "quality_metric_value"),
            ("feature_dim", "feature_dim"),
            ("chosen_k", "chosen_k"),
            ("silhouette", "silhouette"),
            ("sampled_frames", "sampled_frames"),
            ("cache_hits", "cache_hits"),
        ]
        parts = []
        for field_key, label in fields:
            value = summary.get(field_key)
            if value is None:
                value = "—"
            parts.append(f"{label}={value}")
        self._append_log_with_level("INFO", "Summary: " + ", ".join(parts))
        if summary.get("auto_mode"):
            auto_iterations = summary.get("auto_iterations")
            auto_sim = summary.get("auto_similarity_threshold")
            auto_min = summary.get("auto_min_duration_sec")
            auto_source = summary.get("auto_initial_source") or "default"
            sim_text = "—" if auto_sim is None else f"{float(auto_sim):.3f}"
            min_text = "—" if auto_min is None else f"{float(auto_min):.2f}"
            self._append_log_with_level(
                "INFO",
                f"Auto tuning: iterations={auto_iterations}, sim={sim_text}, min_dur={min_text}s, source={auto_source}",
            )
            history = summary.get("auto_history") or []
            for attempt in history:
                iteration = attempt.get("iteration")
                sim_val = attempt.get("similarity")
                min_val = attempt.get("min_duration")
                k_val = attempt.get("chosen_k")
                sil_val = attempt.get("silhouette")
                decision = attempt.get("decision") or ""
                sim_entry = "—" if sim_val is None else f"{float(sim_val):.3f}"
                min_entry = "—" if min_val is None else f"{float(min_val):.2f}"
                sil_entry = "—" if sil_val is None else f"{float(sil_val):.3f}"
                self._append_log_with_level(
                    "INFO",
                    f"Auto attempt {iteration}: sim={sim_entry}, min_dur={min_entry}s, k={k_val}, sil={sil_entry}, decision={decision}",
                )
            if auto_sim is not None:
                self._auto_last_similarity = float(auto_sim)
            if auto_min is not None:
                self._auto_last_min_duration = float(auto_min)
            self._update_sensitivity_labels()
        clusters = self._safe_int(summary.get("clusters"))
        chosen_k = self._safe_int(summary.get("chosen_k"))
        if chosen_k and chosen_k > 1 and clusters == 1:
            self._append_log_with_level(
                "WARN",
                "Homogeneous material: effective single cluster; coloring may be skipped",
            )
        coloring = summary.get("coloring") if isinstance(summary.get("coloring"), dict) else None
        if coloring:
            state = coloring.get("coloring")
            reason = coloring.get("reason") or ""
            message = f"Coloring decision: {state}"
            if reason:
                message += f" ({reason})"
            level = "WARN" if state == "skipped" else "INFO"
            self._append_log_with_level(level, message)
        quality_name = summary.get("quality_metric_name")
        quality_value = summary.get("quality_metric_value")
        if quality_name and quality_value is not None:
            self._last_tuning_metric = float(quality_value)
            self._last_tuning_metric_name = str(quality_name)
            self._append_log_with_level(
                "INFO",
                f"Quality metric: {quality_name}={float(quality_value):.4f}",
            )
        else:
            self._last_tuning_metric = None
            self._last_tuning_metric_name = None
        location_stats = summary.get("locations_stats") if isinstance(summary, dict) else None
        if isinstance(location_stats, dict):
            clip_stats = location_stats.get("clip_count") or {}
            duration_stats = location_stats.get("duration_sec") or {}
            clip_msg = ", ".join(f"{key}={value}" for key, value in clip_stats.items())
            duration_msg = ", ".join(f"{key}={float(value):.2f}" for key, value in duration_stats.items())
            if clip_msg:
                self._append_log_with_level("INFO", f"Location clip counts: {clip_msg}")
            if duration_msg:
                self._append_log_with_level("INFO", f"Location durations (s): {duration_msg}")

    def _safe_int(self, value: object) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self._controller.is_running:
            self._append_log_with_level("INFO", "Stopping active analysis before exit…")
            self._controller.stop_analysis()
            deadline = time.monotonic() + 3.0
            while self._controller.is_running and time.monotonic() < deadline:
                QApplication.processEvents()
                time.sleep(0.05)
            if self._controller.is_running:
                self._append_log_with_level(
                    "WARN",
                    "Analysis still running; close cancelled to avoid abort.",
                )
                event.ignore()
                return
        super().closeEvent(event)


def main(argv: Optional[list[str]] = None) -> int:
    app = QApplication(argv or sys.argv)
    if not ensure_pyqt_compatibility():
        return 1
    controller = AnalysisController()
    window = MainWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
