#!/usr/bin/env python3
"""Run only the nine missing matched-seed jobs on GPUs 0 and 1, after evaluating reused weights."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"runs/detect/targeted-mcgi-isolated"
SCRIPT=ROOT/"tools/run_mcgi_targeted.py"
LOCK=threading.Lock()
STOP=threading.Event()
TASKS=queue.Queue()
STATE={"pid":os.getpid(),"started":time.time(),"seeds":[0,1,2],"jobs":[]}


def status():
    path=OUTPUT/"queue.json"
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(STATE,indent=2)+"\n")
    temporary.replace(path)


def worker(device):
    while not STOP.is_set():
        try: job=TASKS.get_nowait()
        except queue.Empty: return
        with LOCK:
            job.update(state="running",device=device,started=time.time());status()
        command=[sys.executable,str(SCRIPT),"--variant",job["variant"],"--seeds",str(job["seed"]),"--device",str(device)]
        with (OUTPUT/"logs"/f'{job["variant"]}-{job["seed"]}.log').open("w") as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        with LOCK:
            job.update(state="completed" if result.returncode==0 else "failed",returncode=result.returncode,finished=time.time())
            if result.returncode: STOP.set()
            status()
        TASKS.task_done()


if __name__=="__main__":
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/"logs").mkdir(exist_ok=True)
    if (OUTPUT/"queue.json").exists():
        old=json.loads((OUTPUT/"queue.json").read_text())
        try: os.kill(old["pid"],0)
        except ProcessLookupError: pass
        else: raise RuntimeError("A recorded queue process is still active.")
    status()
    with (OUTPUT/"logs/references.log").open("w") as log:
        subprocess.run([sys.executable,str(SCRIPT),"--variant","references","--seeds","0","1","2","--device","0"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    for seed in (0,1,2):
        for variant in ("ca","bilinear","carafe"):
            job={"variant":variant,"seed":seed,"state":"pending"}
            STATE["jobs"].append(job);TASKS.put(job)
    status()
    workers=[threading.Thread(target=worker,args=(device,)) for device in (0,1)]
    for thread in workers:thread.start()
    for thread in workers:thread.join()
    subprocess.run([sys.executable,str(SCRIPT),"--variant","summarize"],cwd=ROOT,check=True)
    STATE.update(finished=time.time(),state="failed" if STOP.is_set() else "completed");status()
    sys.exit(1 if STOP.is_set() else 0)
