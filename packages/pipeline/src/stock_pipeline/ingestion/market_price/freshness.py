"""EOD freshness measured against completed XNYS sessions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import exchange_calendars as xcals
import pandas as pd


EXCHANGE_TIMEZONE = "America/New_York"


def expected_eod_session(now: datetime, *, grace: timedelta = timedelta(hours=3)):
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now_utc = pd.Timestamp(now).tz_convert("UTC")
    local_date = now_utc.tz_convert(EXCHANGE_TIMEZONE).date()
    calendar = xcals.get_calendar("XNYS")
    session = calendar.date_to_session(pd.Timestamp(local_date), direction="previous")
    if session.date() == local_date and now_utc < calendar.session_close(session) + grace:
        session = calendar.previous_session(session)
    return session.date()


def eod_is_stale(event_time: datetime, *, now: datetime | None = None) -> bool:
    if event_time.tzinfo is None:
        raise ValueError("event_time must be timezone-aware")
    now = now or datetime.now(timezone.utc)
    bar_date = pd.Timestamp(event_time).tz_convert(EXCHANGE_TIMEZONE).date()
    return bar_date < expected_eod_session(now)
