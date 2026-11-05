"""Parse raw HDFS logs into event templates and group the lines into per-block traces.

Each line is split into header and message, variable fields (block ids, IPs, paths,
numbers) are masked, and the masked message goes through Drain to get a template id.
Most masked messages repeat many times, so Drain only sees each distinct one once.

    python -m logad.parse --log data/raw/HDFS.log --out data/parsed
"""

import argparse
import csv
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

# 081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-160899 ...
LINE_RE = re.compile(r"^(\d{6}) (\d{6}) (\d+) (\w+) ([^:]+): (.*)$")
BLOCK_RE = re.compile(r"blk_-?\d+")

MASKS = [
    (BLOCK_RE, "<BLK>"),
    (re.compile(r"/?\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?"), "<IP>"),
    (re.compile(r"(?:/[\w.\-]+)+"), "<PATH>"),
    (re.compile(r"\b\d+\b"), "<NUM>"),
]


def mask(message: str) -> str:
    for pattern, token in MASKS:
        message = pattern.sub(token, message)
    return message


def parse_line(line: str):
    """Return (unix_ts, message) or None if the line does not match the HDFS format."""
    m = LINE_RE.match(line)
    if m is None:
        return None
    date, clock, _pid, _level, _component, message = m.groups()
    return to_unix(date + clock), message


@lru_cache(maxsize=4096)
def to_unix(stamp: str) -> int:
    """'081109203518' -> seconds since epoch, read as UTC so results don't depend on the machine."""
    dt = datetime.strptime(stamp, "%y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


class Parser:
    def __init__(self, sim_th: float = 0.9, depth: int = 4):
        config = TemplateMinerConfig()
        config.drain_sim_th = sim_th
        config.drain_depth = depth
        config.profiling_enabled = False
        self.miner = TemplateMinerQuiet(config=config)
        self.cache: dict[str, int] = {}
        self.line_counts: dict[int, int] = defaultdict(int)

    def event_id(self, message: str) -> int:
        masked = mask(message)
        cid = self.cache.get(masked)
        if cid is None:
            cid = self.miner.add_log_message(masked)["cluster_id"]
            self.cache[masked] = cid
        self.line_counts[cid] += 1
        return cid

    def templates(self) -> dict[int, tuple[str, int]]:
        """event id -> (template, number of lines)."""
        return {c.cluster_id: (c.get_template(), self.line_counts[c.cluster_id])
                for c in self.miner.drain.clusters}


class TemplateMinerQuiet(TemplateMiner):
    """TemplateMiner without its per-message logging."""

    def add_log_message(self, log_message: str) -> dict:
        cluster, change_type = self.drain.add_log_message(log_message)
        return {"cluster_id": cluster.cluster_id, "change_type": change_type}


def parse_file(path: Path, parser: Parser):
    """Read a log file and return (traces, skipped). traces maps block id to
    [first_ts, last_ts, [event ids in order]]. A line naming several blocks is added to each."""
    traces: dict[str, list] = {}
    skipped = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_line(line.rstrip("\n"))
            if parsed is None:
                skipped += 1
                continue
            ts, message = parsed
            blocks = BLOCK_RE.findall(message)
            if not blocks:
                continue
            eid = parser.event_id(message)
            for blk in dict.fromkeys(blocks):
                t = traces.get(blk)
                if t is None:
                    traces[blk] = [ts, ts, [eid]]
                else:
                    t[1] = ts
                    t[2].append(eid)
    return traces, skipped


def write_outputs(out: Path, parser: Parser, traces: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    templates = parser.templates()
    with open(out / "templates.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "lines", "template"])
        for cid in sorted(templates):
            template, size = templates[cid]
            w.writerow([f"E{cid}", size, template])
    with open(out / "traces.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["block_id", "start", "end", "events"])
        for blk, (start, end, events) in traces.items():
            w.writerow([blk, start, end, " ".join(f"E{e}" for e in events)])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", type=Path, default=Path("data/raw/HDFS.log"))
    ap.add_argument("--out", type=Path, default=Path("data/parsed"))
    # 0.5 (the usual Drain default) merges "... terminating" with "... Interrupted." once
    # the variables are masked, and those two mean opposite things here.
    ap.add_argument("--sim-th", type=float, default=0.9)
    args = ap.parse_args(argv)

    t0 = time.time()
    parser = Parser(sim_th=args.sim_th)
    traces, skipped = parse_file(args.log, parser)
    write_outputs(args.out, parser, traces)
    print(
        f"{len(traces)} blocks, {len(parser.templates())} templates, "
        f"{len(parser.cache)} distinct masked messages, {skipped} unparsed lines, "
        f"{time.time() - t0:.0f}s"
    )


if __name__ == "__main__":
    main()
