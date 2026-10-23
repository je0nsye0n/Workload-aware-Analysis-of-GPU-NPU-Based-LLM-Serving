"""vLLM-based performance benchmark implementation."""

import os
import re
import signal
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from furiosa_perf.runner.monitor import HardwareMonitor
from furiosa_perf.utils.config import PerformanceBenchConfig, ScenarioConfig
from furiosa_perf.utils.logger import logger

WORKSPACE = "/dev/shm/sdl2026/bench_space"
RESULTS_DIR = Path(
    os.environ.get(
        "FURIOSA_PERF_RESULTS_DIR",
        Path(__file__).resolve().parents[2] / "results",
    )
)

class VllmPerformanceBenchmark:
    """Run vLLM benchmark scenarios against a running API server.

    Manages an isolated Python venv for the benchmark tool, drives one
    ``vllm bench serve`` subprocess per scenario, parses the output, and writes
    a consolidated CSV summary.
    """

    ENV: dict[str, Any] = {
        "SETUP_COMMANDS": [
            ("create_venv", ["uv", "venv", "--python", "3.12", "{venv_dir}"]),
            ("install_package", ["uv", "pip", "install", "--python", "{python_exe}", "vllm[bench]"]),
        ],
    }

    COMMAND: list[str] = [
        "{exe}",
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--percentile-metrics=ttft,tpot,itl,e2el",
        "--metric-percentiles=25,50,75,90,95,99",
        "--max-concurrency={max_concurrency}",
        "--num-prompts={num_prompts}",
        "--request-rate={request_rate}",
        "--model={model}",
        "--ignore-eos",
        "--ready-check-timeout-sec=0",
        "--trust-remote-code",
        "--temperature=0",
    ]

    VLLM_COMMANDS: dict[str, list[str]] = {
        "offline": [
            "--backend=vllm",
            "--dataset-name=random",
            "--random-input-len={input_tokens}",
            "--random-output-len={output_tokens}",
            "--random-range-ratio={random_range_ratio}",
        ],
        "vl-offline": [
            "--backend=openai-chat",
            "--dataset-name=random-mm",
            "--random-input-len={input_tokens}",
            "--random-output-len={output_tokens}",
            "--random-range-ratio={random_range_ratio}",
            "--random-mm-base-items-per-request={random_mm_base_items_per_request}",
            "--random-mm-bucket-config={random_mm_bucket_config}",
            "--random-mm-limit-mm-per-prompt={random_mm_limit_mm_per_prompt}",
            "--endpoint=/v1/chat/completions",
        ],
        "reranker": [
            "--backend=vllm-rerank",
            "--dataset-name=random-rerank",
            "--random-input-len={input_tokens}",
            "--random-batch-size={random_batch_size}",
            "--endpoint=/v1/rerank",
        ],
        "embeddings": [
            "--backend=openai-embeddings",
            "--dataset-name=random",
            "--random-input-len={input_tokens}",
            "--endpoint=/v1/embeddings",
        ],
        "custom-datasets": [
            "--backend=vllm",
            "--dataset-name=custom",
            "--dataset-path={dataset_path}",
            "--custom-output-len={output_tokens}",
            "--skip-chat-template",
            "--disable-shuffle",
        ],
    }

    def __init__(
        self,
        config: PerformanceBenchConfig,
        backend: str,
        host: str | None = None,
        port: int | None = None,
        env: dict[str, Any] | None = None,
    ) -> None:
        """Initialise benchmark paths, commands, and environment.

        Args:
            config: Benchmark configuration (name, task, scenarios, …).
            backend: Serving framework name used in result-directory paths.
            host: API server hostname (forwarded to the benchmark command).
            port: API server port (forwarded to the benchmark command).
            env: Environment variables passed to the benchmark subprocess
                (must include ``HF_TOKEN`` if model download is needed).
        """
        self.name = config.name
        self.model = config.model
        self.device_name = config.device_name
        self.used_device_num = config.used_device_num
        self.task = config.task
        self.scenarios = config.scenarios
        self.backend = backend
        self.total_results: dict[str, Any] = {}
        self._wikitext_datasets: dict[int, Path] = {}
        self.host = host
        self.port = port
        self.bench_process: subprocess.Popen[str] | None = None
        self.env: dict[str, Any] = env or {}

        self.base_dir = self.get_base_dir()
        self.venv_dir = self.base_dir / f".{self.name}"
        self.python_exe = (self.venv_dir / "bin" / "python").absolute()

        self.SETUP_COMMANDS = self._build_setup_commands()
        self.BASE_COMMAND = [arg.replace("{exe}", str(self.python_exe)) for arg in self.COMMAND]
        self.log_init()

    def log_init(self) -> None:
        """Log resolved paths at startup for traceability."""
        logger.info(f"base_dir={self.base_dir}")
        logger.info(f"venv_dir={self.venv_dir}")
        logger.info(f"python_exe={self.python_exe}")
        logger.info("=" * 100)

    def _build_setup_commands(self) -> list[tuple[str, list[str]]]:
        """Expand format placeholders in ENV setup commands.

        Returns:
            A list of ``(step_name, command_argv)`` tuples ready to execute.
        """
        fmt = {
            "python_exe": self.python_exe,
            "venv_dir": str(self.venv_dir),
        }
        return [
            (key, [arg.format(**fmt) for arg in arg_list])
            for key, arg_list in self.ENV["SETUP_COMMANDS"]
        ]

    def start_info(self) -> None:
        """Log a human-readable summary of what is about to run."""
        logger.info(
            f"Starting performance test — {self.name} + {self.task} | "
            f"Model: {self.model} | "
            f"Device: {self.device_name} x {self.used_device_num}"
        )

    def setup(self) -> None:
        """Create the venv and install the benchmark tool inside it."""
        logger.info(f"Setting up {self.name} benchmark environment…")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._execute_setup_commands()
        logger.info(f"{self.name} setup completed successfully.")

    def _execute_setup_commands(self) -> None:
        """Run each setup step sequentially, raising on failure."""
        for step_name, command in self.SETUP_COMMANDS:
            logger.info(f"Setup step: {step_name}")
            if step_name == "create_venv" and self.venv_dir.exists():
                logger.info(f"Skipping {step_name} — venv already exists at {self.venv_dir}.")
                continue
            if step_name == "install_package" and (self.venv_dir / "bin" / "vllm").exists():
                check = subprocess.run(
                    [str(self.python_exe), "-c", "import pandas, matplotlib, seaborn, datasets, scipy, plotly"],
                    capture_output=True,
                    text=True,
                )
                if check.returncode == 0:
                    logger.info(f"Skipping {step_name} — vllm benchmark dependencies are installed.")
                    continue
                version = subprocess.run(
                    [str(self.python_exe), "-c", "import importlib.metadata; print(importlib.metadata.version('vllm'))"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
                command[-1] = f"vllm[bench]=={version}"
            p = subprocess.run(command, capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"Setup step {step_name!r} failed: {p.stderr}")

    @staticmethod
    def _scenario_output_tokens(scenario: ScenarioConfig) -> int:
        """Output-token count for a scenario, or ``0`` for tasks that don't generate.

        ``reranker`` and ``embeddings`` produce no output tokens, so their scenario
        configs omit the field entirely; treat those as ``0``.
        """
        return getattr(scenario, "output_tokens", 0)

    def _scenario_result_name(self, scenario: ScenarioConfig) -> str:
        """Unique per-scenario result sub-directory name.

        The generative tasks vary output length, reranker varies batch size, and
        embeddings vary neither — so the middle component is task-dependent to keep
        every scenario's directory (and monitoring log) distinct:

        * ``offline`` / ``vl-offline`` → ``"<isl>.<osl>.<concurrency>"``
        * ``reranker`` → ``"<isl>.b<batch>.<concurrency>"``
        * ``embeddings`` → ``"<isl>.<concurrency>"``
        """
        if self.task == "wikitext-2-full":
            return "full-dataset"
        parts = [str(scenario.input_tokens)]
        if hasattr(scenario, "output_tokens"):
            parts.append(str(scenario.output_tokens))
        elif hasattr(scenario, "random_batch_size"):
            parts.append(f"b{scenario.random_batch_size}")
        parts.append(str(scenario.max_concurrency))
        return ".".join(parts)

    def _get_format_args(self, scenario: ScenarioConfig) -> dict[str, Any]:
        """Build the format-arg dict for filling command placeholders.

        Args:
            scenario: The scenario whose fields are merged into the dict.

        Returns:
            A dict suitable for ``str.format(**args)`` on command templates.
        """
        return {
            **asdict(scenario),
            "model": self.model,
        }

    def _prepare_wikitext_dataset(self, input_tokens: int) -> Path:
        """Prepare each input length once per run, using the benchmark tokenizer."""
        if input_tokens not in self._wikitext_datasets:
            destination = self.base_dir / "datasets" / f"input_{input_tokens}.jsonl"
            env = os.environ.copy()
            if hf_token := self.env.get("HF_TOKEN"):
                env["HF_TOKEN"] = hf_token
            subprocess.run(
                [
                    str(self.python_exe),
                    # Avoid shadowing the installed vllm package with this module.
                    "-P",
                    str(Path(__file__).with_name("prepare_wikitext.py").resolve()),
                    "--source", env.get("WIKITEXT2_SOURCE", "/dev/shm/sdl2026/bench_data/wikitext2.jsonl"),
                    "--destination", str(destination),
                    "--model", self.model,
                    "--input-tokens", str(input_tokens),
                ],
                check=True,
                env=env,
            )
            self._wikitext_datasets[input_tokens] = destination
        return self._wikitext_datasets[input_tokens]

    def _get_command_for_scenario(self, scenario: ScenarioConfig) -> list[str]:
        """Build the full benchmark argv for a single scenario.

        Args:
            scenario: Scenario whose parameters are embedded in the command.

        Returns:
            List of strings ready to pass to :class:`subprocess.Popen`.
        """
        fmt = self._get_format_args(scenario)
        if self.task == "custom-datasets":
            dataset_path = (
                fmt.get("dataset_path")
                or os.environ.get("CUSTOM_DATASET_SOURCE")
                or os.environ.get("WIKITEXT2_SOURCE")
            )
            if not dataset_path:
                raise ValueError(
                    "Set dataset_path in the scenario or CUSTOM_DATASET_SOURCE "
                    "(WIKITEXT2_SOURCE is also supported)."
                )
            source = Path(dataset_path).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Custom dataset source not found: {source}")
            fmt["dataset_path"] = source
        if self.task == "wikitext-2":
            fmt["dataset_path"] = self._prepare_wikitext_dataset(scenario.input_tokens)
        if self.task == "wikitext-2-full":
            source = Path(os.environ["WIKITEXT2_SOURCE"]).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"WikiText-2 source not found: {source}")
            fmt["dataset_path"] = source
        base_command = self.BASE_COMMAND
        if self.task in {"wikitext-2-full", "custom-datasets"}:
            base_command = [arg for arg in base_command if arg != "--ignore-eos"]
        command = [arg.format(**fmt) for arg in (base_command + self.VLLM_COMMANDS[self.task])]
        if self.host:
            command.append(f"--host={self.host}")
        if self.port:
            command.append(f"--port={self.port}")
        return command

    def run(self) -> None:
        """Run all scenarios sequentially and store results in :attr:`total_results`."""
        cwd: Path = self.base_dir.resolve()
        self.total_results = {
            "model": self.model,
            "task": self.task,
            "device_name": self.device_name,
            "used_device_num": self.used_device_num,
            "results": [],
        }
        for scenario in self.scenarios:
            self.total_results["results"].append(self._run_scenario(scenario, cwd))

        assert len(self.total_results["results"]) > 0, "No results collected from benchmarks"
        logger.info(f"Total results collected: {len(self.total_results['results'])}")

    def _run_scenario(self, scenario: ScenarioConfig, cwd: Path) -> dict[str, Any]:
        """Execute one benchmark scenario and return the parsed result dict.

        Args:
            scenario: Scenario configuration to run.
            cwd: Working directory for the benchmark subprocess.

        Returns:
            Dict containing parsed metrics and hardware power summary.

        Raises:
            RuntimeError: If the benchmark subprocess exits with a non-zero code.
        """
        command = self._get_command_for_scenario(scenario)
        logger.info(f"Executing: {' '.join(command)} | cwd: {cwd}")

        start_timestamp = datetime.now(UTC).isoformat()
        env = os.environ.copy()
        if hf_token := self.env.get("HF_TOKEN"):
             env["HF_TOKEN"] = hf_token
        self.bench_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=cwd,
            bufsize=1,
            start_new_session=True,
            env=env,
        )

        output_lines: list[str] = []
        if self.bench_process.stdout is not None:
            for line in self.bench_process.stdout:
                logger.info(line.strip())
                output_lines.append(line)

        self.bench_process.wait()
        if self.bench_process.returncode != 0:
            raise RuntimeError(
                f"Benchmark failed — subprocess exited with code {self.bench_process.returncode}"
            )
        self.bench_process = None
        end_timestamp = datetime.now(UTC).isoformat()

        result = self._parse_results("".join(output_lines), scenario)
        result.update(
            HardwareMonitor.get_benchmark_power_summary(
                csv_file_path=self.base_dir / f"{self.device_name}_{self.used_device_num}_monitoring_log.csv",
                start_dt=start_timestamp,
                end_dt=end_timestamp,
            )
        )
        logger.info(f"Parsed result: {result}")
        return result

    def _parse_results(self, stdout: str, scenario: ScenarioConfig) -> dict[str, Any]:
        """Parse vLLM bench stdout into a structured result dict.

        Args:
            stdout: Full standard output captured from the benchmark subprocess.
            scenario: Scenario whose token counts seed the result dict.

        Returns:
            Dict of metric name → value, keyed with display-friendly names.
        """
        results: dict[str, Any] = {
            "Input Tokens": None if self.task in {"wikitext-2", "wikitext-2-full"} else scenario.input_tokens,
            "Output Tokens": None if self.task == "wikitext-2-full" else self._scenario_output_tokens(scenario),
            "Concurrent": scenario.max_concurrency,
        }
        completed_requests: int | None = None
        total_input_tokens: int | None = None
        total_output_tokens: int | None = None
        for line in stdout.strip().split("\n"):
            if self.task in {"wikitext-2", "wikitext-2-full"}:
                if match := re.search(r"Successful requests:\s+(\d+)", line):
                    completed_requests = int(match.group(1))
                if match := re.search(r"Total input tokens:\s+(\d+)", line):
                    total_input_tokens = int(match.group(1))
                if match := re.search(r"Total generated tokens:\s+(\d+)", line):
                    total_output_tokens = int(match.group(1))
            try:
                metric, *_, score = line.split()
            except ValueError as e:
                logger.debug(f"Skipping unparseable line: {e}")
                continue
            if "Total token throughput (tok/s):" in line or "Total Token throughput (tok/s):" in line:
                results["Total Throughput(tok/s)"] = float(score)
            if "Output token throughput (tok/s):" in line:
                results["Output Throughput(tok/s)"] = float(score)
            if "TTFT" in line:
                results[f"{metric.title()} TTFT(s)"] = round(float(score) / 1000.0, 3)
            if "TPOT" in line:
                results[f"{metric.title()} TPOT(ms)"] = float(score)
            if "ITL" in line:
                results[f"{metric.title()} ITL(ms)"] = float(score)
            if "E2EL" in line:
                results[f"{metric.title()} E2EL(s)"] = round(float(score) / 1000.0, 2)
        if self.task in {"wikitext-2", "wikitext-2-full"} and completed_requests and total_input_tokens is not None:
            results["Input Tokens"] = round(total_input_tokens / completed_requests, 1)
        if self.task == "wikitext-2-full":
            if completed_requests and total_output_tokens is not None:
                results["Output Tokens"] = round(total_output_tokens / completed_requests, 1)
            results["Successful Requests"] = completed_requests
            results["Total Input Tokens"] = total_input_tokens
            results["Total Output Tokens"] = total_output_tokens
            results["Output Token Cap"] = scenario.output_tokens
        return results

    def finish_info(self, desc: str = "") -> None:
        """Write one CSV summary file for all completed scenario results.

        Args:
            desc: Prefix text prepended to the auto-generated report description.
        """
        scenario_results = self.total_results["results"]
        report_desc = self._build_report_desc(desc)
        headers = self._prepare_result_headers(scenario_results)

        rows = [list(result.values()) for result in scenario_results]
        self._write_summary_csv(self.get_vllm_result_dir(), headers, rows, report_desc)

    def _build_report_desc(self, prefix: str) -> str:
        """Build the human-readable description block prepended to summary files.

        Args:
            prefix: Text to place before the auto-generated metadata lines.

        Returns:
            Multi-line description string.
        """
        return (
            f"{prefix}\n"
            f"* Performance Benchmark - {self.name} for {self.task}\n"
            f"* Model Info: {self.model}\n"
            f"* Device Info: {self.device_name} x {self.used_device_num}\n"
            f"* date: {datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}\n"
            f"* Scenario Results:"
        )

    def _prepare_result_headers(self, scenario_results: list[dict[str, Any]]) -> list[str]:
        """Optionally strip P95/P99 keys and return the column header list.

        Args:
            scenario_results: List of result dicts.

        Returns:
            List of column header strings derived from the first result dict.
        """
        return list(scenario_results[0].keys()) if scenario_results else []

    def _write_summary_csv(
        self,
        result_dir: Path,
        headers: list[str],
        rows: list[Any],
        desc: str,
    ) -> None:
        """Write a CSV summary file for *rows*.

        Args:
            result_dir: Directory to write the model-specific CSV summary.
            headers: Column header strings.
            rows: List of row value lists (one per scenario result).
            desc: Description block prepended before the CSV table.
        """
        csv_lines = [",".join(headers)] + [",".join(str(v) for v in row) for row in rows]
        model_name = self.model.rstrip("/").split("/")[-1]
        benchmark_name = os.environ.get("FURIOSA_PERF_BENCHMARK_NAME", self.name)
        with Path.open(result_dir / f"{model_name}-{benchmark_name}-summary.csv", "w") as f:
            f.write(desc + "\n" + "\n".join(csv_lines) + "\n")

    def _to_isl_osl_result(self, input_tokens: int, output_tokens: int) -> str:
        """Return a compact ``"<ISL>_<OSL>"`` key for grouping results.

        Args:
            input_tokens: Input sequence length in tokens.
            output_tokens: Output sequence length in tokens.

        Returns:
            String like ``"1k_1k"`` or ``"512_256"``.
        """
        isl = f"{input_tokens // 1024}k" if input_tokens >= 1024 else str(input_tokens)
        osl = f"{output_tokens // 1024}k" if output_tokens >= 1024 else str(output_tokens)
        return f"{isl}_{osl}"

    def get_base_dir(self) -> Path:
        """Return (and create) the top-level workspace directory.

        Returns:
            Resolved :class:`~pathlib.Path` to ``./bench_space/``.
        """
        workspace = Path(WORKSPACE)
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def get_vllm_result_dir(self) -> Path:
        """Return (and create) the model result directory under ``results/``.

        Layout: ``results/<DEVICE>_<NUM>_<BACKEND>/<name>/<task>/<model>``.
        """
        result_dir = (
            RESULTS_DIR
            / f"{self.device_name}_{self.used_device_num}_{self.backend}"
            / self.name
            / self.task
            / self.model.split("/")[-1]
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir.resolve()

    def stop(self) -> None:
        """Terminate the running benchmark subprocess if one is active."""
        if self.bench_process is None:
            return

        if self.bench_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.bench_process.pid), signal.SIGTERM)
                self.bench_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.bench_process.pid), signal.SIGKILL)

        self.bench_process = None
