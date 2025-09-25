# Pigment Bucket M0 Runbook

This guide explains how to run the mock analysis service and the Resolve adapter for the M0 skeleton build.

## 1. Prepare a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Start the FastAPI mock service

```bash
bash scripts/run_service.sh
```

The script sets `PYTHONPATH` to the project root and launches `uvicorn` on `127.0.0.1:8765`.

## 3. Make the Resolve scripting API available

Ensure the DaVinci Resolve scripting modules are on `PYTHONPATH`. On macOS this usually means:

```bash
export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export PYTHONPATH="$RESOLVE_SCRIPT_API/Modules:$PYTHONPATH"
```

Consult `docs/resolve_api_reference.md` for the authoritative API reference.

## 4. Run the Resolve adapter

Launch the script from the Resolve scripting console or via the command line:

```bash
python3 scripts/pigmentbucket_run.py
```

### Dry run mode

Add `--dry-run` to collect the service response without calling `TimelineItem.SetClipColor`.

```bash
python3 scripts/pigmentbucket_run.py --dry-run
```

Both modes set a completion marker at `00:00:00:00` and print summary counts to the console.

