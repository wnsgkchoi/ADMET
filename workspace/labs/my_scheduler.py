import threading
import queue
import subprocess
import os
import time
import argparse
import signal
import shutil
from collections import deque
from datetime import datetime, timedelta

# UI Library
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.console import Group
from rich.text import Text

# === 전역 변수 (초기화는 main에서) ===
job_queue = queue.Queue()
gpu_status = {}  # {gpu_id: [job_idx, ...]}
running_processes = {} # {job_idx: Popen_object}
log_buffer = deque(maxlen=200)
lock = threading.Lock()

# 통계 변수
total_jobs = 0
completed_jobs = 0
failed_jobs = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Custom GPU Job Scheduler with Dashboard")
    
    # 필수 인자: 명령어 파일
    parser.add_argument("command_file", type=str, help="Path to the file containing commands (line by line)")
    
    # 선택 인자: GPU 설정
    parser.add_argument(
        "--gpus", 
        nargs="+", 
        default=["0", "1", "2", "3"],
        help="List of GPUs to use. Format: 'ID' or 'ID:SLOTS'. (e.g., --gpus 0:6 1:4 2 3)"
    )
    
    parser.add_argument("--default-slots", type=int, default=6, help="Default number of slots per GPU if not specified")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory to save individual job logs")
    parser.add_argument("--error-file", type=str, default="error_report.txt", help="File to save error summaries")
    
    return parser.parse_args()

def parse_gpu_config(gpu_args, default_slots):
    """
    CLI 입력값을 파싱하여 {gpu_id: slots} 딕셔너리로 변환
    입력 예시: ['0:6', '1:4', '2'] -> {0: 6, 1: 4, 2: default_slots}
    """
    config = {}
    for item in gpu_args:
        if ":" in item:
            try:
                gpu_id_str, slots_str = item.split(":")
                gpu_id = int(gpu_id_str)
                slots = int(slots_str)
                config[gpu_id] = slots
            except ValueError:
                print(f"[Error] Invalid GPU format: {item}. Use 'ID:SLOTS' or just 'ID'.")
                exit(1)
        else:
            try:
                gpu_id = int(item)
                config[gpu_id] = default_slots
            except ValueError:
                print(f"[Error] Invalid GPU ID: {item}")
                exit(1)
    return config

def add_log(message, style="white"):
    # UTC+9 (Korea Standard Time)
    timestamp = (datetime.utcnow() + timedelta(hours=9)).strftime("%H:%M:%S")
    log_buffer.append(Text(f"[{timestamp}] {message}", style=style))

def worker(gpu_id, log_dir, error_file, progress, task_id):
    global completed_jobs, failed_jobs
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    while True:
        try:
            job_index, command = job_queue.get_nowait()
        except queue.Empty:
            break

        with lock:
            gpu_status[gpu_id].append(job_index)
            add_log(f"Align job #{job_index} to GPU {gpu_id}", style="cyan")

        log_path = os.path.join(log_dir, f"job_{job_index}.log")
        start_time = time.time()
        
        try:
            with open(log_path, "w") as f_out:
                process = subprocess.Popen(
                    command, 
                    shell=True, 
                    env=env, 
                    stdout=f_out, 
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
                
                with lock:
                    running_processes[job_index] = process

                returncode = process.wait()
                
                with lock:
                    if job_index in running_processes:
                        del running_processes[job_index]
            
            duration = time.time() - start_time
            
            with lock:
                gpu_status[gpu_id].remove(job_index)
                progress.update(task_id, advance=1)
                
                if returncode == 0:
                    completed_jobs += 1
                    add_log(f"Job #{job_index} completed ({duration:.1f}s)", style="green")
                else:
                    # If killed by signal (negative return code), don't count as failure if we are shutting down
                    if returncode == -signal.SIGTERM:
                        add_log(f"Job #{job_index} Terminated", style="yellow")
                    else:
                        failed_jobs += 1
                        add_log(f"Error: Job #{job_index} failed! See logs.", style="bold red")
                        with open(error_file, "a") as err_f:
                            err_f.write(f"[{datetime.now()}] Job #{job_index} Failed (GPU {gpu_id})\n")
                            err_f.write(f"Command: {command}\n")
                            err_f.write(f"Log: {log_path}\n\n")

        except Exception as e:
            with lock:
                if job_index in gpu_status[gpu_id]:
                    gpu_status[gpu_id].remove(job_index)
                if job_index in running_processes:
                    del running_processes[job_index]
                failed_jobs += 1
                progress.update(task_id, advance=1)
                add_log(f"Critical Error on Job #{job_index}: {str(e)}", style="bold red")

        job_queue.task_done()

def generate_layout(progress_obj, sorted_gpu_ids, gpu_config):
    # 터미널 크기에 맞춰 레이아웃 동적 조정
    term_width, term_height = shutil.get_terminal_size()
    
    header_height = 3
    # GPU Table 높이 계산: Panel(2) + Table(Header(2) + Rows(N) + Bottom(1)) approx N + 6
    gpu_count = len(sorted_gpu_ids)
    body_height = gpu_count + 6
    
    # 남은 공간을 Footer(로그)에 할당
    footer_height = term_height - header_height - body_height
    if footer_height < 5:
        footer_height = 5

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=header_height),
        Layout(name="body", size=body_height),
        Layout(name="footer", size=footer_height)
    )

    layout["header"].update(Panel(progress_obj, title="Global Progress", border_style="blue"))

    table = Table(title="GPU Status", expand=True, border_style="dim")
    table.add_column("GPU ID (Run/Max)", justify="center", style="cyan", no_wrap=True, width=20)
    table.add_column("Running Jobs (Index)", style="yellow")

    for gpu in sorted_gpu_ids:
        current_jobs = len(gpu_status[gpu])
        max_slots = gpu_config[gpu]
        
        running_jobs = ", ".join([f"#{idx}" for idx in gpu_status[gpu]])
        if not running_jobs:
            running_jobs = "[dim]Idle[/dim]"
        
        gpu_label = f"GPU {gpu} ({current_jobs} / {max_slots})"
        table.add_row(gpu_label, running_jobs)

    layout["body"].update(Panel(table, border_style="white"))
    
    # Footer 높이에 맞춰 로그 개수 조절 (Panel border 2줄 제외)
    lines_for_logs = max(0, footer_height - 2)
    visible_logs = list(log_buffer)[-lines_for_logs:]
    
    log_content = Group(*visible_logs)
    layout["footer"].update(Panel(log_content, title="Real-time Logs", border_style="green"))

    return layout

def main():
    args = parse_args()
    
    # 설정 초기화
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    
    # GPU 설정 파싱
    gpu_config = parse_gpu_config(args.gpus, args.default_slots)
    sorted_gpu_ids = sorted(gpu_config.keys())
    
    # 전역 상태 초기화
    for gpu in sorted_gpu_ids:
        gpu_status[gpu] = []

    # 명령어 로드
    commands = []
    if os.path.exists(args.command_file):
        with open(args.command_file, 'r') as f:
            commands = [line.strip() for line in f if line.strip()]
    else:
        print(f"[Error] Command file '{args.command_file}' not found.")
        return

    global total_jobs
    total_jobs = len(commands)
    
    # 큐에 작업 넣기 (1-based index)
    for idx, cmd in enumerate(commands):
        job_queue.put((idx + 1, cmd))

    print(f"Loaded {total_jobs} jobs.")
    print(f"GPU Configuration: {gpu_config}")
    time.sleep(1)

    # Rich Progress
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
    )
    task_id = progress.add_task("Processing...", total=total_jobs)

    # 스레드 생성 (설정된 슬롯 수만큼)
    threads = []
    for gpu_id, slots in gpu_config.items():
        for _ in range(slots):
            t = threading.Thread(
                target=worker, 
                args=(gpu_id, args.log_dir, args.error_file, progress, task_id), 
                daemon=True
            )
            t.start()
            threads.append(t)

    # Live UI 실행
    try:
        with Live(generate_layout(progress, sorted_gpu_ids, gpu_config), refresh_per_second=4, screen=False) as live:
            while not job_queue.empty() or any(len(jobs) > 0 for jobs in gpu_status.values()):
                live.update(generate_layout(progress, sorted_gpu_ids, gpu_config))
                time.sleep(0.2)
            live.update(generate_layout(progress, sorted_gpu_ids, gpu_config))
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user. Terminating all jobs...")
        with lock:
            for idx, proc in running_processes.items():
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    pass
        print("All jobs terminated.")
        return

    print("\nAll jobs finished!")
    print(f"Completed: {completed_jobs}, Failed: {failed_jobs}")
    print(f"Check '{args.error_file}' for details if failures exist.")

if __name__ == "__main__":
    main()