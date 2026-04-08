# AI Pong Training Runner

This is the private laptop-side training workflow for the public AI Pong human training site. It is intentionally separate from the website so a scheduled task can pull gameplay chunks, train a supervised model, evaluate it, and promote a browser-ready policy.

## What It Does

- Pulls `.aipong` binary trajectory chunks from local disk or Cloudflare R2.
- Reads the 10 Hz human observations and human actions.
- Trains the same small browser policy shape used by the Pong game.
- Runs simple promotion gates so a collapsed model does not replace the public bot.
- Uses class-balanced loss by default so the common `stay` action does not swamp `up` and `down`.
- Writes each run under `runs/YYYYMMDD-HHMMSS/`.
- Optionally promotes a passing model to the website's `web/models/current-policy.json`.

The runner exits successfully when there is not enough data yet. That makes it safe for scheduled intervals.

## Setup

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Edit the included `config.json` before the first real R2 run. `config.example.json` is there as a clean reset template.

For local collector storage, leave `r2.enabled` as `false` and point `data_dir` at the website's `data/trajectories` folder.

For Cloudflare R2, install the AWS CLI on the laptop, configure these environment variables, and set `r2.enabled` to `true`:

```powershell
$env:AWS_ACCESS_KEY_ID = "your-r2-access-key-id"
$env:AWS_SECRET_ACCESS_KEY = "your-r2-secret-access-key"
```

The runner uses:

```powershell
aws s3 sync s3://your-r2-bucket-name/ai-pong-training/trajectories .\downloaded-trajectories --endpoint-url https://your-account-id.r2.cloudflarestorage.com
```

## Manual Run

```powershell
.\run_once.ps1
```

Or directly:

```powershell
.\.venv\Scripts\python.exe .\run_once.py --config .\config.json
```

To force promotion of a passing candidate for a test run:

```powershell
.\.venv\Scripts\python.exe .\run_once.py --config .\config.json --promote
```

## Scheduled Run

Install a Windows scheduled task that runs every 12 hours:

```powershell
.\install_windows_task.ps1 -EveryMinutes 720
```

Install it and start immediately:

```powershell
.\install_windows_task.ps1 -EveryMinutes 720 -StartNow
```

## Promotion Rules

The included `config.json` promotes passing candidates to the website's `web/models/current-policy.json`. Set `promotion.enabled` to `false` if you want scheduled runs to only produce candidates under `runs/`.

The default gates are conservative starter checks:

- at least `min_files` trajectory chunks
- at least `min_samples` usable human action samples
- validation accuracy above `min_validation_accuracy`
- prediction entropy above `min_prediction_entropy`
- not too many predictions as only `stay`
- not too many predictions as any single action

You should tune these after you see real data. The point is not to prove the model is amazing; it is to stop obviously broken candidates from replacing the public bot.
