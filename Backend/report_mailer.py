"""
report_mailer.py — PackerVision AI
====================================
Sends two kinds of automated emails:

  1. shift_report_email(shift, packer_id, summary)
     Called by ShiftScheduler._stop_session() when a shift ends.
     Sends a per-packer HTML summary for the finished shift.

  2. daily_digest_email()
     Called at midnight by the DailyDigestScheduler.
     Aggregates all three shifts for the calendar day and mails one
     combined HTML report to the client.

Both helpers require a live Flask app-context (already present in the
scheduler threads) and use the Flask-Mail instance configured in app.py.

Recipients
----------
Configure CLIENT_EMAILS below (or set the REPORT_EMAILS env-var as a
comma-separated list).  A BCC list can be added via BCC_EMAILS.

    REPORT_EMAILS="client@example.com,manager@example.com" python app.py
"""

import os
import threading
import time
from datetime import datetime, timedelta
from typing import Optional

# ── Recipients ────────────────────────────────────────────────────────────────
# Fallback hard-coded list — override via env-var in production.
_default_recipients = ["sakshitandon1193@gmail.com","sakshi.tandon@amzbizsol.in"]   # ← replace with client emails

CLIENT_EMAILS: list[str] = [
    e.strip()
    for e in os.environ.get("REPORT_EMAILS", "").split(",")
    if e.strip()
] or _default_recipients

BCC_EMAILS: list[str] = [
    e.strip()
    for e in os.environ.get("REPORT_BCC", "").split(",")
    if e.strip()
]

# ── Shift labels ──────────────────────────────────────────────────────────────
SHIFT_LABELS = {
    "A": "Shift A  (6:00 AM – 2:00 PM)",
    "B": "Shift B  (2:00 PM – 10:00 PM)",
    "C": "Shift C  (10:00 PM – 6:00 AM)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eff_color(value: float) -> str:
    """Return a traffic-light hex color for an efficiency percentage."""
    if value >= 85:
        return "#27ae60"   # green
    elif value >= 65:
        return "#f39c12"   # amber
    return "#e74c3c"       # red

def _fmt(value, suffix="", decimals=1) -> str:
    """Safe number formatter — shows '–' for None."""
    if value is None:
        return "–"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def _shift_summary_html(shift: str, rows: list[dict]) -> str:
    """
    Build an HTML table section for one shift.

    rows — list of dicts, each with keys:
        packer_name, bags_placed, bags_missed, stuck_bags,
        packer_efficiency, manual_efficiency, elapsed_time
    """
    label = SHIFT_LABELS.get(shift, f"Shift {shift}")

    if not rows:
        return f"""
        <h3 style="color:#555;margin-top:28px">{label}</h3>
        <p style="color:#888;font-style:italic">No sessions recorded for this shift.</p>
        """

    # Aggregate totals
    total_placed  = sum(r.get("bags_placed", 0) or 0 for r in rows)
    total_missed  = sum(r.get("bags_missed", 0) or 0 for r in rows)
    total_stuck   = sum(r.get("stuck_bags",  0) or 0 for r in rows)
    avg_eff       = sum(r.get("packer_efficiency", 0) or 0 for r in rows) / len(rows)
    avg_manual    = sum(r.get("manual_efficiency", 0) or 0 for r in rows) / len(rows)

    tbody = ""
    for r in rows:
        eff = r.get("packer_efficiency", 0) or 0
        tbody += f"""
        <tr>
          <td style="padding:8px 12px;border-bottom:1px solid #eee">{r.get('packer_name','–')}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center">{r.get('bags_placed','–')}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center">{r.get('bags_missed','–')}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center">{r.get('stuck_bags','–')}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center;
                     font-weight:bold;color:{_eff_color(eff)}">{_fmt(eff,'%')}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center">{_fmt(r.get('manual_efficiency'),'%')}</td>
          <td style="padding:8px 12px;border-bottom:1px solid #eee;text-align:center">{_fmt(r.get('elapsed_time'),' min')}</td>
        </tr>"""

    eff_color = _eff_color(avg_eff)
    return f"""
    <h3 style="color:#2c3e50;margin-top:32px;border-left:4px solid {eff_color};
               padding-left:10px">{label}</h3>

    <table style="width:100%;border-collapse:collapse;font-size:14px;margin-top:8px">
      <thead>
        <tr style="background:#f4f6f8">
          <th style="padding:10px 12px;text-align:left;border-bottom:2px solid #ddd">Packer</th>
          <th style="padding:10px 12px;border-bottom:2px solid #ddd">Bags Placed</th>
          <th style="padding:10px 12px;border-bottom:2px solid #ddd">Bags Missed</th>
          <th style="padding:10px 12px;border-bottom:2px solid #ddd">Stuck Bags</th>
          <th style="padding:10px 12px;border-bottom:2px solid #ddd">Packer Eff.</th>
          <th style="padding:10px 12px;border-bottom:2px solid #ddd">Manual Eff.</th>
          <th style="padding:10px 12px;border-bottom:2px solid #ddd">Duration</th>
        </tr>
      </thead>
      <tbody>{tbody}</tbody>
      <tfoot>
        <tr style="background:#f9fafb;font-weight:bold">
          <td style="padding:10px 12px">Totals / Avg</td>
          <td style="padding:10px 12px;text-align:center">{total_placed}</td>
          <td style="padding:10px 12px;text-align:center">{total_missed}</td>
          <td style="padding:10px 12px;text-align:center">{total_stuck}</td>
          <td style="padding:10px 12px;text-align:center;color:{eff_color}">{_fmt(avg_eff,'%')}</td>
          <td style="padding:10px 12px;text-align:center">{_fmt(avg_manual,'%')}</td>
          <td style="padding:10px 12px;text-align:center">–</td>
        </tr>
      </tfoot>
    </table>
    """

def _wrap_html(title: str, body: str, subtitle: str = "") -> str:
    """Wrap body content in a clean branded HTML email shell."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title></head>
<body style="margin:0;padding:0;background:#f0f2f5;font-family:Arial,sans-serif">
  <table width="100%" cellpadding="0" cellspacing="0">
    <tr><td align="center" style="padding:30px 10px">
      <table width="640" cellpadding="0" cellspacing="0"
             style="background:#fff;border-radius:8px;overflow:hidden;
                    box-shadow:0 2px 8px rgba(0,0,0,.1)">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#1a237e,#283593);
                     padding:28px 32px;text-align:center">
            <h1 style="color:#fff;margin:0;font-size:22px;letter-spacing:.5px">
              📦 PackerVision AI
            </h1>
            <p style="color:#c5cae9;margin:6px 0 0;font-size:14px">{title}</p>
            {f'<p style="color:#9fa8da;margin:4px 0 0;font-size:12px">{subtitle}</p>' if subtitle else ''}
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:28px 32px">
            {body}
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#f4f6f8;padding:16px 32px;text-align:center;
                     color:#aaa;font-size:11px;border-top:1px solid #eee">
            PackerVision AI — Automated Report &nbsp;·&nbsp;
            Generated {datetime.now().strftime("%d %b %Y, %I:%M %p")}
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API — called by the scheduler
# ─────────────────────────────────────────────────────────────────────────────

def shift_report_email(shift: str, packer_id: str, summary: dict,
                       packer_name: str = "Unknown") -> bool:
    """
    Send a single-packer shift-end report email.

    Called from ShiftScheduler._stop_session() just after save_report_to_db().
    Runs in the background thread that owns the app-context, so Flask-Mail
    works without any extra context juggling.

    Parameters
    ----------
    shift       : 'A', 'B', or 'C'
    packer_id   : packer identifier (used in subject only)
    summary     : dict from PackerEfficiencyMonitor.get_summary()
    packer_name : human-readable packer name

    Returns True if mail was sent, False on any error.
    """
    try:
        from app import mail   # imported here to avoid circular import at module load

        label    = SHIFT_LABELS.get(shift, f"Shift {shift}")
        date_str = datetime.now().strftime("%d %b %Y")
        eff      = float(summary.get("packer_efficiency", 0) or 0)

        rows = [{
            "packer_name":       packer_name,
            "bags_placed":       summary.get("bags_placed", 0),
            "bags_missed":       summary.get("bags_missed", 0),
            "stuck_bags":        summary.get("stuck_bags", 0),
            "packer_efficiency": eff,
            "manual_efficiency": summary.get("manual_efficiency", 0),
            "elapsed_time":      round(float(summary.get("elapsed_time", 0) or 0), 1),
        }]

        table_html = _shift_summary_html(shift, rows)

        kpi_color = _eff_color(eff)
        body = f"""
        <p style="color:#555;font-size:15px">
          Hi,<br><br>
          The <strong>{label}</strong> session for <strong>{packer_name}</strong>
          has ended.  Here is the performance summary:
        </p>

        <!-- KPI badge -->
        <div style="text-align:center;margin:24px 0">
          <div style="display:inline-block;background:{kpi_color};color:#fff;
                      border-radius:50%;width:80px;height:80px;line-height:80px;
                      font-size:22px;font-weight:bold">
            {_fmt(eff,'%',0)}
          </div>
          <p style="color:#555;margin:8px 0 0;font-size:13px">Packer Efficiency</p>
        </div>

        {table_html}

        <p style="color:#aaa;font-size:12px;margin-top:24px">
          You can view the full report in the PackerVision dashboard.
        </p>
        """

        from flask_mail import Message
        msg = Message(
            subject=f"[PackerVision] {label} Report — {packer_name} ({date_str})",
            recipients=CLIENT_EMAILS,
            bcc=BCC_EMAILS or None,
            html=_wrap_html(
                title=f"{label} Report",
                subtitle=f"{packer_name}  ·  {date_str}",
                body=body,
            ),
        )
        mail.send(msg)
        print(f"[MAIL] Shift-end email sent for {packer_name} ({shift})")
        return True

    except Exception as exc:
        print(f"[MAIL] shift_report_email error: {exc}")
        return False


def daily_digest_email(date: Optional[datetime] = None) -> bool:
    """
    Send one daily digest that covers all three shifts for `date`
    (defaults to yesterday so the full day is always complete).

    Called by DailyDigestScheduler at midnight.

    Returns True if mail was sent, False on any error.
    """
    try:
        from app import mail
        from flask_mail import Message
        from database import get_db_connection

        if date is None:
            date = datetime.now() - timedelta(days=1)

        date_str  = date.strftime("%Y-%m-%d")
        date_nice = date.strftime("%d %b %Y")

        conn = get_db_connection()
        rows_db = conn.execute("""
            SELECT r.shift, r.packer_name, p.name as p_name,
                   r.bags_placed, r.bags_missed, r.stuck_bags,
                   r.packer_efficiency, r.manual_efficiency, r.elapsed_time
            FROM reports r
            LEFT JOIN packers p ON r.packer_id = p.id
            WHERE date(r.timestamp) = ?
            ORDER BY r.shift, r.timestamp
        """, (date_str,)).fetchall()
        conn.close()

        # Group rows by shift
        by_shift: dict[str, list] = {"A": [], "B": [], "C": []}
        for r in rows_db:
            shift = r["shift"] or "Manual"
            if shift in by_shift:
                by_shift[shift].append({
                    "packer_name":       r["p_name"] or r["packer_name"] or "Unknown",
                    "bags_placed":       r["bags_placed"],
                    "bags_missed":       r["bags_missed"],
                    "stuck_bags":        r["stuck_bags"],
                    "packer_efficiency": r["packer_efficiency"],
                    "manual_efficiency": r["manual_efficiency"],
                    "elapsed_time":      round(float(r["elapsed_time"] or 0), 1),
                })

        # Overall day KPIs
        total_placed = sum(r["bags_placed"] or 0 for r in rows_db)
        total_missed = sum(r["bags_missed"] or 0 for r in rows_db)
        total_stuck  = sum(r["stuck_bags"]  or 0 for r in rows_db)
        avg_eff      = (
            sum(float(r["packer_efficiency"] or 0) for r in rows_db) / len(rows_db)
            if rows_db else 0
        )

        kpi_html = f"""
        <table width="100%" cellpadding="0" cellspacing="0" style="margin:16px 0">
          <tr>
            {"".join(f'''
            <td style="text-align:center;padding:12px">
              <div style="background:#f4f6f8;border-radius:8px;padding:14px">
                <div style="font-size:26px;font-weight:bold;color:{_eff_color(avg_eff) if label=='Avg. Eff.' else '#2c3e50'}">{val}</div>
                <div style="font-size:11px;color:#888;margin-top:4px">{label}</div>
              </div>
            </td>'''
            for val, label in [
                (total_placed,         "Bags Placed"),
                (total_missed,         "Bags Missed"),
                (total_stuck,          "Stuck Bags"),
                (f"{_fmt(avg_eff,'%')}", "Avg. Eff."),
            ])}
          </tr>
        </table>
        """

        shifts_html = "".join(
            _shift_summary_html(s, by_shift[s]) for s in ("A", "B", "C")
        )

        body = f"""
        <p style="color:#555;font-size:15px">Hi,<br><br>
          Here is the <strong>Daily Performance Digest</strong> for
          <strong>{date_nice}</strong>.
        </p>
        {kpi_html}
        {shifts_html}
        <p style="color:#aaa;font-size:12px;margin-top:24px">
          Full reports and evidence are available in the PackerVision dashboard.
        </p>
        """

        msg = Message(
            subject=f"[PackerVision] Daily Digest — {date_nice}",
            recipients=CLIENT_EMAILS,
            bcc=BCC_EMAILS or None,
            html=_wrap_html(
                title="Daily Performance Digest",
                subtitle=date_nice,
                body=body,
            ),
        )
        mail.send(msg)
        print(f"[MAIL] Daily digest sent for {date_str}")
        return True

    except Exception as exc:
        print(f"[MAIL] daily_digest_email error: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Daily Digest Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class DailyDigestScheduler:
    """
    Fires daily_digest_email() every night at SEND_HOUR:SEND_MINUTE.

    Usage (in app.py, after mail = Mail(app)):
        from report_mailer import DailyDigestScheduler
        _digest = DailyDigestScheduler(app)
        _digest.start()

    The thread is a daemon so it never blocks clean process exit.
    Calling start() a second time is a no-op.
    """

    SEND_HOUR   = 0    # midnight local time
    SEND_MINUTE = 5    # :05 — gives the DB a few seconds to finish the last report

    def __init__(self, app=None):
        self._app     = app
        self._thread  = None
        self._started = False

    def start(self):
        if self._started:
            return
        self._started = True
        self._thread  = threading.Thread(
            target=self._loop, daemon=True, name="DailyDigestScheduler"
        )
        self._thread.start()
        print(f"[MAIL] Daily digest scheduler started — fires at "
              f"{self.SEND_HOUR:02d}:{self.SEND_MINUTE:02d} local time")

    def _loop(self):
        if self._app:
            self._app.app_context().push()

        last_sent_date: Optional[str] = None

        while True:
            now = datetime.now()
            target_date = now.strftime("%Y-%m-%d")

            if (now.hour == self.SEND_HOUR
                    and now.minute == self.SEND_MINUTE
                    and last_sent_date != target_date):
                daily_digest_email()          # sends for yesterday by default
                last_sent_date = target_date

            time.sleep(30)   # wake every 30 s to check the clock