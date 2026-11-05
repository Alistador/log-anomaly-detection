from logad.parse import Parser, mask, parse_file, parse_line, to_unix

LINES = """\
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-1608999687919862906 src: /10.250.19.102:54106 dest: /10.250.19.102:50010
081109 203518 35 INFO dfs.FSNamesystem: BLOCK* NameSystem.allocateBlock: /mnt/hadoop/mapred/system/job_200811092030_0001/job.jar. blk_-1608999687919862906
081109 203519 145 INFO dfs.DataNode$PacketResponder: PacketResponder 1 for block blk_-1608999687919862906 terminating
081109 203520 147 INFO dfs.DataNode$PacketResponder: PacketResponder 2 for block blk_7503483334202473044 Interrupted.
081109 203521 19 INFO dfs.FSNamesystem: BLOCK* ask 10.250.14.38:50010 to delete  blk_7503483334202473044 blk_-1608999687919862906
this line is not an hdfs log line
081109 203522 13 INFO dfs.DataBlockScanner: Verification succeeded for blk_7503483334202473044
"""


def test_parse_line_splits_header():
    ts, msg = parse_line(LINES.splitlines()[0])
    assert ts == to_unix("081109203518")
    assert msg.startswith("Receiving block blk_-1608999687919862906")
    assert parse_line("garbage") is None


def test_timestamps_are_utc():
    assert to_unix("700101000000") == 0
    assert to_unix("081109203519") - to_unix("081109203518") == 1


def test_mask_replaces_variables():
    _, msg = parse_line(LINES.splitlines()[0])
    assert mask(msg) == "Receiving block <BLK> src: <IP> dest: <IP>"
    _, msg = parse_line(LINES.splitlines()[1])
    assert mask(msg) == "BLOCK* NameSystem.allocateBlock: <PATH> <BLK>"


def test_terminating_and_interrupted_stay_separate():
    p = Parser()
    a = p.event_id("PacketResponder 1 for block blk_1 terminating")
    b = p.event_id("PacketResponder 2 for block blk_2 Interrupted.")
    c = p.event_id("PacketResponder 0 for block blk_3 terminating")
    assert a == c != b


def test_parse_file_groups_by_block(tmp_path):
    log = tmp_path / "x.log"
    log.write_text(LINES)
    p = Parser()
    traces, skipped = parse_file(log, p)
    assert skipped == 1
    assert set(traces) == {"blk_-1608999687919862906", "blk_7503483334202473044"}
    first, last, events = traces["blk_-1608999687919862906"]
    assert len(events) == 4  # three own lines plus the shared delete line
    assert last - first == 3
    assert len(traces["blk_7503483334202473044"][2]) == 3
    assert sum(n for _, n in p.templates().values()) == 6
