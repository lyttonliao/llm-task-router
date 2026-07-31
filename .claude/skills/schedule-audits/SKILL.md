---
name: schedule-audits
description: Schedule drift/shadow audits as daily recurring tasks (cron, launchd, or Windows Task Scheduler) with portable recipes for all platforms.
---

`audit_tier2.py` and `report_shadow_divergence.py` are now installable
console scripts — `llm-route-audit-tier2` and `llm-route-shadow-report` —
and can be scheduled to run on a daily cadence. Neither script writes
anything or auto-corrects anything — they only print a report a human
reads. Neither has an explicit exit-code contract: a genuine failure
(unset `DATABASE_URL`, DB connection error) surfaces as an uncaught
exception and a nonzero exit already, for free; a "consider raising"
verdict or a high divergence rate still exits 0, since that's a judgment
call for the human reading the log — a scheduler should alert on "job
crashed," never on "job ran and found something worth reading."

## Linux/macOS cron

```cron
# crontab -e
0 6 * * * DATABASE_URL=postgresql:///llm_task_router /path/to/venv/bin/llm-route-audit-tier2 >> /path/to/logs/audit_tier2.log 2>&1
15 6 * * * DATABASE_URL=postgresql:///llm_task_router /path/to/venv/bin/llm-route-shadow-report >> /path/to/logs/shadow_report.log 2>&1
```

Daily at 06:00 and 06:15 UTC (or your cron server's timezone). Adapt
paths/venv for your own install.

## macOS launchd

Create two plist files under `~/Library/LaunchAgents/`:

```xml
<!-- com.llm-task-router.audit-tier2.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.llm-task-router.audit-tier2</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/venv/bin/llm-route-audit-tier2</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>DATABASE_URL</key><string>postgresql:///llm_task_router</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>0</integer></dict>
    <key>StandardOutPath</key><string>/path/to/logs/audit_tier2.log</string>
    <key>StandardErrorPath</key><string>/path/to/logs/audit_tier2.log</string>
</dict>
</plist>
```

```xml
<!-- com.llm-task-router.shadow-report.plist -->
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>com.llm-task-router.shadow-report</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/venv/bin/llm-route-shadow-report</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>DATABASE_URL</key><string>postgresql:///llm_task_router</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>15</integer></dict>
    <key>StandardOutPath</key><string>/path/to/logs/shadow_report.log</string>
    <key>StandardErrorPath</key><string>/path/to/logs/shadow_report.log</string>
</dict>
</plist>
```

Load both with `launchctl load ~/Library/LaunchAgents/com.llm-task-router.*.plist`,
or verify with `launchctl start com.llm-task-router.audit-tier2` (etc.) before
trusting the schedule.

## Windows Task Scheduler

```bat
schtasks /create /sc daily /st 06:00 /tn "llm-route-audit-tier2" /tr "C:\path\to\logs\run_audit_tier2.bat"
schtasks /create /sc daily /st 06:15 /tn "llm-route-shadow-report" /tr "C:\path\to\logs\run_shadow_report.bat"
```

`schtasks`' `/tr` argument doesn't reliably support shell redirection
operators (`>>`) inline, so each `/tr` target should be a small wrapper
`.bat` file that sets `DATABASE_URL` and redirects output itself:

```bat
@echo off
set DATABASE_URL=postgresql:///llm_task_router
"C:\path\to\venv\Scripts\llm-route-audit-tier2.exe" >> "C:\path\to\logs\audit_tier2.log" 2>&1
```

```bat
@echo off
set DATABASE_URL=postgresql:///llm_task_router
"C:\path\to\venv\Scripts\llm-route-shadow-report.exe" >> "C:\path\to\logs\shadow_report.log" 2>&1
```

## Why daily, not weekly?

Daily was chosen on all three platforms to track drift/divergence trends
while `routing_decisions` is still accumulating. These are portable recipes
— the point is a pattern any installer can adopt on whichever OS they're on,
with no machine-specific paths hardcoded into the repo itself.
