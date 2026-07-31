# Scheduling audits

Built 2026-07-28. `audit_tier2.py` and `report_shadow_divergence.py`
originally lived under `scripts/` as loose dev-only files — confirmed via
`uv build --wheel` that `scripts/` never shipped in the built wheel, so
anyone installing this as a published package had no way to run either one.
Both moved into the `llm_task_router` package proper with real
`[project.scripts]` entries: `llm-route-audit-tier2` and
`llm-route-shadow-report`, alongside the existing `llm-route`/`llm-chat`
pattern (see `cli-entrypoints.md`). `scripts/seed_vector_store.py`
deliberately stayed put — a one-time cold-start operation, not something
meant to run on a schedule.

Neither script writes anything or auto-corrects anything — they only print a
report a human reads. Neither has an explicit exit-code contract: a genuine
failure (unset `DATABASE_URL`, DB connection error) surfaces as an uncaught
exception and a nonzero exit already, for free; a "consider raising" verdict
or a high divergence rate still exits 0, since that's a judgment call for
the human reading the log — a scheduler should alert on "job crashed," never
on "job ran and found something worth reading."

Example recipes for all three major platforms (generic — adapt paths/venv
for your own install, not tied to any one machine). See the
`schedule-audits` skill for the automated version of this.

```cron
# crontab -e
0 6 * * * DATABASE_URL=postgresql:///llm_task_router /path/to/venv/bin/llm-route-audit-tier2 >> /path/to/logs/audit_tier2.log 2>&1
15 6 * * * DATABASE_URL=postgresql:///llm_task_router /path/to/venv/bin/llm-route-shadow-report >> /path/to/logs/shadow_report.log 2>&1
```

```xml
<!-- ~/Library/LaunchAgents/com.llm-task-router.audit-tier2.plist (macOS) -->
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

(`report_shadow_divergence.py`'s launchd job is the same shape, swapping in
`llm-route-shadow-report` and a different log path/label.)

```bat
:: Windows, via Task Scheduler (schtasks) - staggered 06:00 / 06:15 to match
schtasks /create /sc daily /st 06:00 /tn "llm-route-audit-tier2" /tr "C:\path\to\logs\run_audit_tier2.bat"
schtasks /create /sc daily /st 06:15 /tn "llm-route-shadow-report" /tr "C:\path\to\logs\run_shadow_report.bat"
```

`schtasks`' `/tr` argument doesn't reliably support shell redirection
operators (`>>`) inline, so each `/tr` target should be a small wrapper
`.bat` file that sets `DATABASE_URL` and redirects output itself, e.g.
`run_audit_tier2.bat`:

```bat
@echo off
set DATABASE_URL=postgresql:///llm_task_router
"C:\path\to\venv\Scripts\llm-route-audit-tier2.exe" >> "C:\path\to\logs\audit_tier2.log" 2>&1
```

Daily was chosen over weekly, on all three platforms, to track
drift/divergence trends while `routing_decisions` is still accumulating.
None of these three recipes is tied to any one machine — the point is a
portable pattern any installer can adopt on whichever OS they're on. The
launchd variant has since been instantiated on this dev machine — see
CLAUDE.md's "Next step" for the real paths/schedule used; that instantiation
lives outside the repo (`~/Library/LaunchAgents/`), so this section stays
generic/unmodified.
