"""Cold start on Cloud Run: the one latency number a first visitor actually feels.

This number went stale once already, and the way it went stale is the point. It
was measured by hand, written into the README, and then the image lost 130 MB and
four fifths of its vector store — so the README carried a figure that described a
container that no longer existed, with a note admitting it. A number nobody can
re-derive is a number that will eventually be wrong, which is the same argument
`record_demo.py` makes about the screenshots.

"Cold" means no container was running when the request arrived. Cloud Run
reclaims an idle instance after roughly fifteen minutes and there is no API to
force scale-to-zero, so **the wait is the method** — this script sleeps out that
window and then times the first request. It prints what it is waiting for rather
than sleeping silently, because a script that looks hung is a script nobody runs
twice.

Any other traffic during the idle window — a visitor, a bot, a Gradio tab
reconnecting in the background — warms the instance, and then "cold" silently
means "warm". The first version of this script called that undetectable and told
the reader to decide whether the service had been quiet. **That was wrong, and it
produced a wrong number on the first run**: three samples came back at 0.34, 0.14
and 0.18 s — one of them faster than its own warm follow-up, which is the tell —
because a Stackdriver uptime check was hitting the service every ~2 minutes from
several probe regions. Cloud Logging knew that the whole time.

So the window is now *verified* rather than assumed: after waiting, this reads the
request log for the host it is about to measure and refuses to call a sample cold
if anything else touched it. A silent wrong number is worse than a loud failure,
which is the same reason `record_demo.py` will not write a GIF it caught on the
cache path.

The other limit is real and stays: cold start varies with what the platform is
doing (image pull, host placement). One sample is an anecdote; `--repeat` takes
more, each preceded by its own idle window, which is why three samples cost an
hour of waiting and not three minutes.

To measure a service that is deliberately kept warm, deploy the same image as a
revision nothing routes to and measure that instead — it has its own instance
pool, so the live demo keeps its uptime check and its 0.09 s:

    gcloud run deploy leasehound --source . --region us-west1 --tag cold --no-traffic

Costs nothing: it fetches the landing page, which makes no model calls.

    python -m scripts.measure_cold_start
    python -m scripts.measure_cold_start --repeat 3
    python -m scripts.measure_cold_start --url https://cold---leasehound-….run.app
    python -m scripts.measure_cold_start --no-wait   # times the CURRENT state, warm or not
"""

import argparse
import json
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from evaluation.provenance import stamp

# The URL the README sends visitors to. Measuring anything else would measure a
# different door into the same house.
DEFAULT_URL = "https://leasehound-sbiact24ca-uw.a.run.app"

# Cloud Run's documented idle window is ~15 minutes; 17 buys margin without
# turning a single sample into half an hour.
DEFAULT_IDLE_MINUTES = 17

RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "cold_start_results.json"

# The Cloud Run service the log filter asks about. Cold start is a property of a
# revision, but the log is keyed by service, and the host filter below is what
# narrows it to the one URL being measured — which is how a no-traffic tagged
# revision can be measured while the live one is being kept warm.
SERVICE = "leasehound"

# urllib stamps this on our own requests, and nothing else in this project talks
# to the deployed service, so it is enough to tell our traffic from everyone
# else's in the log.
OWN_USER_AGENT = "Python-urllib"

# A cold container has to pull the image, boot Python, import gradio and chromadb
# and open the store, so a generous ceiling is not optimism — it is the point of
# measuring.
REQUEST_TIMEOUT_S = 180


def fetch_seconds(url: str) -> tuple[float, int, int]:
    """Time a full GET of the landing page. Returns (seconds, status, bytes)."""
    start = time.monotonic()
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_S) as response:
        body = response.read()
        status = response.status
    return time.monotonic() - start, status, len(body)


def foreign_requests(service: str, host: str, since: datetime) -> int | None:
    """Requests OTHER than ours that reached `host` since `since`.

    `None` means the question could not be asked — no gcloud, no credentials, no
    log permission — which is emphatically not the same as "none happened", and is
    reported as its own state rather than quietly passing for zero.
    """
    query = " AND ".join([
        'resource.type="cloud_run_revision"',
        f'resource.labels.service_name="{service}"',
        # Startup probes and container logs are not requests; only real traffic
        # keeps an instance alive.
        'httpRequest.requestMethod!=""',
        f'httpRequest.requestUrl:"{host}"',
        f'timestamp>="{since.strftime("%Y-%m-%dT%H:%M:%SZ")}"',
    ])
    try:
        found = subprocess.run(
            ["gcloud", "logging", "read", query, "--limit=200",
             "--format=value(httpRequest.userAgent)"],
            capture_output=True, text=True, timeout=180, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return sum(1 for line in found.stdout.splitlines()
               if line.strip() and OWN_USER_AGENT not in line)


def wait_for_idle(minutes: float) -> None:
    """Sleep out the scale-to-zero window, saying how much is left as it goes."""
    deadline = time.monotonic() + minutes * 60
    print(f"  waiting {minutes:g} min for the instance to be reclaimed "
          f"(any other traffic in this window invalidates the sample)")
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return
        print(f"    {left / 60:4.1f} min left", flush=True)
        time.sleep(min(60, left))


def measure(url: str, wait_minutes: float, repeat: int, service: str) -> dict:
    host = urllib.parse.urlparse(url).hostname
    samples = []
    for run in range(1, repeat + 1):
        print(f"sample {run} of {repeat}")
        window_opened = datetime.now(timezone.utc)
        if wait_minutes > 0:
            wait_for_idle(wait_minutes)
        # Asked AFTER the wait and BEFORE the request, so it covers exactly the
        # window whose emptiness the word "cold" is claiming. With --no-wait there
        # is no window, so the answer is None (unverifiable) and not 0 (verified
        # empty) — the difference is the whole point of this check.
        intruders = foreign_requests(service, host, window_opened) if wait_minutes else None

        cold, status, size = fetch_seconds(url)
        # Immediately again: the same request against a now-running container is
        # the baseline that says how much of the cold number was the cold start
        # rather than the network and the page itself.
        warm, warm_status, _ = fetch_seconds(url)
        print(f"  cold {cold:.2f}s (HTTP {status}, {size / 1024:.0f} KB)"
              f" · warm {warm:.2f}s (HTTP {warm_status})")
        if intruders is None:
            print("    ! could not read the request log, so the window is "
                  "unverified — treat this sample as warm unless you know better")
        elif intruders:
            print(f"    ! {intruders} other request(s) hit {host} during the wait, "
                  f"so this instance was never reclaimed and this is NOT a cold "
                  f"start")
        samples.append({"cold_seconds": round(cold, 2),
                        "warm_seconds": round(warm, 2),
                        "status": status,
                        "waited_minutes": wait_minutes,
                        "foreign_requests_during_wait": intruders,
                        "window_verified_empty": intruders == 0})

    trusted = [s["cold_seconds"] for s in samples if s["window_verified_empty"]]
    result = {
        "url": url,
        # models=False: this fetches a static page and calls nothing, so naming
        # three models would imply they had something to do with the number. The
        # commit is the field that matters here — the whole reason the old figure
        # went stale is that the image changed underneath it.
        "provenance": stamp(models=False),
        "idle_wait_minutes": wait_minutes,
        "samples": samples,
        "cold_samples_trusted": len(trusted),
        "warm_median": sorted(s["warm_seconds"] for s in samples)[len(samples) // 2],
    }
    # No cold_* keys at all when nothing earned them: a reader who greps this file
    # for a cold start should come away empty rather than come away wrong.
    if trusted:
        result |= {"cold_min": min(trusted), "cold_max": max(trusted),
                   "cold_median": sorted(trusted)[len(trusted) // 2]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--repeat", type=int, default=1,
                        help="samples, each preceded by its own idle window")
    parser.add_argument("--no-wait", action="store_true",
                        help="skip the idle window and time whatever state the "
                             "service is in — useful for a warm baseline, "
                             "dishonest as a cold-start number")
    parser.add_argument("--service", default=SERVICE,
                        help="Cloud Run service name, for the request-log check")
    parser.add_argument("--write", action="store_true",
                        help=f"also write {RESULTS_PATH.name}")
    args = parser.parse_args()

    wait_minutes = 0 if args.no_wait else DEFAULT_IDLE_MINUTES
    if args.no_wait:
        print("--no-wait: this measures the CURRENT state, which may be warm\n")

    result = measure(args.url, wait_minutes, args.repeat, args.service)
    print()
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
    if not result["cold_samples_trusted"]:
        print("\nNo sample survived the idle-window check, so this run has no cold "
              "start in it. Measure a revision nothing routes to (see the module "
              "docstring) if the service is deliberately kept warm.")

    if args.write:
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        RESULTS_PATH.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {RESULTS_PATH.relative_to(RESULTS_PATH.parent.parent)}")


if __name__ == "__main__":
    main()
