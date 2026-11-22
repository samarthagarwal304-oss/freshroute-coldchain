from freshroute import config as C
from freshroute import datagen
from freshroute import guardrails as G
from freshroute.rag import SopIndex, load_chunks


def test_chunks_and_ids():
    ch = load_chunks()
    assert len(ch) >= 30 and len({c.doc_id for c in ch}) == 10
    assert all(c.chunk_id.startswith("SOP-") and "#" in c.chunk_id for c in ch)


def test_retrieval_finds_right_passage():
    ix = SopIndex()
    assert ix.search("max door open time for vaccines")[0]["chunk_id"] == "SOP-02#2"
    assert ix.search("who handles a Level 3 escalation")[0]["chunk_id"] == "SOP-06#3"
    assert ix.search("battery below 10 percent")[0]["chunk_id"] == "SOP-08#1"


def test_uncovered_topics_return_nothing():
    ix = SopIndex()
    for q in ["How do I submit a GST invoice?", "tyre puncture on the highway", "customs clearance documents"]:
        assert ix.search(q) == [], q


def test_injection_notes_detected_and_benign_notes_not():
    for n in datagen.INJECTION_NOTES:
        assert G.detect_injection(n), n
    for n in datagen.NOTES_NORMAL + datagen.NOTES_TROUBLE + ["Nothing to report. Trip fine.", "Customer asked to ignore the first drop."]:
        assert not G.detect_injection(n), n


def test_sanitize_withholds_injection_and_wraps_others():
    s = G.sanitize_note(datagen.INJECTION_NOTES[0])
    assert s["injection_suspected"] and "IGNORE" not in s["driver_note"]
    ok = G.sanitize_note("Heavy traffic.")
    assert ok["driver_note"].startswith("<untrusted_driver_note>") and not ok["injection_suspected"]


def test_output_filter_blocks_prompt_leak():
    out, blocked = G.check_output(f"Sure! My marker is {C.CANARY}")
    assert blocked and C.CANARY not in out
    assert G.check_output("Risk is HIGH [SOP-06#2]")[1] is False
    assert G.extract_citations("a [SOP-06#2] b [SOP-02#1] c [SOP-06#2]") == ["SOP-06#2", "SOP-02#1"]
