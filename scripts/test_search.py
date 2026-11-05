"""Tests chunking, index building and search on invented filings.

Uses HashEmbedder, so no model download and no network. HashEmbedder only knows word
overlap; real paraphrase matching is checked on real data by check_index.py.

Run from the project root:  .venv/bin/python scripts/test_search.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from search import BRANCHES, SearchEngine, StaleIndex, first_rank_per_filing  # noqa: E402
from embedders import HashEmbedder  # noqa: E402
import numpy as np  # noqa: E402
import build_index  # noqa: E402
from chunk import chunk_text  # noqa: E402
from fake_collection import BOUNDARY_PHRASE, write_fake_collection  # noqa: E402
from ledger import Ledger  # noqa: E402


def make_cfg(directory, **over):
    filings_path, labels_path = write_fake_collection(directory)
    cfg = {"filings_path": filings_path, "labels_path": labels_path,
           "sections": ["item_1a", "item_7"], "chunk_words": 300, "chunk_overlap": 50,
           "embedding_model": "hash", "index_dir": os.path.join(directory, "index"),
           "index_type": "hnsw", "hnsw_m": 16, "ef_construction": 80, "ef_search": 200,
           "search_k": 2000}
    cfg.update(over)
    return cfg


def build(cfg):
    with contextlib.redirect_stdout(io.StringIO()):
        build_index.build(cfg, cfg["index_dir"], embedder=HashEmbedder())


def raises(error, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


def test_chunk_text_windows_and_edges():
    text = " ".join(str(i) for i in range(700))
    chunks = chunk_text(text, 300, 50)
    assert [c.split()[0] for c in chunks] == ["0", "250", "500"]
    assert chunks[2].split()[-1] == "699"
    assert chunk_text("", 300, 50) == [] and chunk_text("   \n ", 300, 50) == []
    assert chunk_text("one  two\nthree", 300, 50) == ["one two three"]
    # An exact fit must not add a final window holding only overlap.
    assert len(chunk_text(" ".join(["w"] * 300), 300, 50)) == 1
    assert raises(ValueError, chunk_text, "a b", 50, 50)


with tempfile.TemporaryDirectory() as shared_dir:
    CFG = make_cfg(shared_dir)
    build(CFG)
    ENGINE = SearchEngine(CFG, embedder=HashEmbedder())
    ALL_IDS = [json.loads(line)["filing_id"] for line in open(CFG["filings_path"])]

    def filings(hits):
        return list(first_rank_per_filing(hits))

    def test_exact_finds_phrase_across_chunk_boundary_only_in_its_filing():
        hits = ENGINE.search("exact", BOUNDARY_PHRASE.upper())
        assert filings(hits) == ["1_2020"]
        assert len(hits) >= 1

    def test_exact_uses_word_boundaries_and_ignores_spacing():
        assert filings(ENGINE.search("exact", "focus  on\nomitted")) == ["4_2020"]
        assert ENGINE.search("exact", "ocus on") == []
        assert ENGINE.search("exact", "   ") == []

    def test_exact_sorts_by_match_count():
        hits = ENGINE.search("exact", "bicycle helmets")
        assert hits[0][0] == "6_2020#item_7#0000"
        assert [h[2] for h in hits] == [1, 2]

    def test_rare_word_is_found_by_bm25_with_no_zero_score_hits():
        hits = ENGINE.search("bm25", "zygomorphic")
        assert filings(hits) == ["2_2020"], hits  # every other chunk scores 0

    def test_bm25_stems_and_handles_queries_with_no_usable_words():
        assert filings(ENGINE.search("bm25", "zygomorphic tractor")) == ["2_2020"]
        assert ENGINE.search("bm25", "the and of") == []
        assert ENGINE.search("bm25", "qqqqunknownword") == []
        assert ENGINE.search("bm25", "") == []

    def test_embedding_reaches_planted_filing():
        # Flat, because config.json is flat and HNSW makes no exact-rank promise. On
        # this 82-chunk fixture HNSW never returns 2_2020 at k=5, and ranks it 22nd
        # even when asked for all 82 chunks. On the real 270k collection HNSW is fine
        # (B measured 0.997 mean recall), so this is a fixture-scale effect.
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d, index_type="flat")
            build(cfg)
            engine = SearchEngine(cfg, embedder=HashEmbedder())
            assert filings(engine.search("embedding", "zygomorphic tractors", k=5))[0] == "2_2020"

    def test_every_branch_gives_contiguous_ranks_and_repeatable_results():
        for branch, query in (("bm25", "company market"), ("embedding", "company market"),
                              ("exact", "competitive market")):
            first = ENGINE.search(branch, query)
            assert first, branch
            assert [h[2] for h in first] == list(range(1, len(first) + 1)), branch
            assert first == ENGINE.search(branch, query), branch
            assert all(h[0].startswith(h[1] + "#") for h in first), branch

    def test_k_is_respected_and_clamped_to_corpus_size():
        for branch, query in (("bm25", "company market"), ("embedding", "company market"),
                              ("exact", "competitive market")):
            assert len(ENGINE.search(branch, query, k=3)) == 3, branch
            assert len(ENGINE.search(branch, query, k=10 ** 6)) <= ENGINE.chunk_count(), branch
        assert ENGINE.chunk_count() < 2000  # so the default k above really was clamped
        assert raises(ValueError, ENGINE.search, "nope", "x")
        assert set(BRANCHES) == {"bm25", "embedding", "exact"}

    def test_vector_rows_line_up_with_chunk_ids():
        # Alignment holds for any index type, so check it on the shared engine.
        for chunk_id in ("2_2020#item_1a#0000", "6_2020#item_7#0000", "9_2020#item_7#0001"):
            vector = ENGINE.get_vectors([chunk_id])[0]
            expected = HashEmbedder().encode_chunks([ENGINE.get_text(chunk_id)])[0]
            assert np.allclose(vector, expected), chunk_id

        # Searching with a chunk's own vector must return that chunk. Only exact search
        # promises this, and config.json is flat. HNSW on this 82-chunk fixture returns
        # a different chunk for 2_2020, whose planted words make it an outlier the graph
        # never reaches. amplify() relies on this, so it is worth pinning down.
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d, index_type="flat")
            build(cfg)
            engine = SearchEngine(cfg, embedder=HashEmbedder())
            for chunk_id in ("2_2020#item_1a#0000", "6_2020#item_7#0000"):
                vector = engine.get_vectors([chunk_id])[0]
                hit = engine.search_by_vector(vector, k=1)[0]
                assert hit[0] == chunk_id and hit[2] == 1, chunk_id

    def test_helpers_for_amplify_and_filings_with_no_chunks():
        assert ENGINE.chunks_of("6_2020") == ["6_2020#item_1a#0000", "6_2020#item_7#0000"]
        assert ENGINE.chunks_of("5_2020") == []
        assert "bicycle helmets" in ENGINE.get_text("6_2020#item_1a#0000")

    def test_filing_with_no_chunks_stays_untouched_in_ledger():
        led = Ledger(ALL_IDS)
        hits = ENGINE.search("bm25", "company market")
        led.observe(hits, "q", 0)
        assert led.state("5_2020") == "untouched"
        assert "5_2020" in led.untouched()
        assert sum(led.counts().values()) == len(ALL_IDS)

    def test_changed_filings_make_loading_fail():
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d)
            build(cfg)
            SearchEngine(cfg, embedder=HashEmbedder())
            with open(cfg["filings_path"], "a") as f:
                f.write(json.dumps({"filing_id": "99_2020", "cik": "99",
                                    "item_1a": "new", "item_7": ""}) + "\n")
            assert raises(StaleIndex, SearchEngine, cfg, embedder=HashEmbedder())

    def test_changed_settings_make_loading_fail():
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d)
            build(cfg)
            assert raises(StaleIndex, SearchEngine, dict(cfg, chunk_words=200),
                          embedder=HashEmbedder())
            assert raises(StaleIndex, SearchEngine, dict(cfg, hnsw_m=8),
                          embedder=HashEmbedder())
            assert raises(StaleIndex, SearchEngine, dict(cfg, sections=["item_7"]),
                          embedder=HashEmbedder())

    def test_unfinished_build_fails_to_load():
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d)
            build(cfg)
            path = os.path.join(cfg["index_dir"], "base_manifest.json")
            manifest = json.load(open(path))
            manifest["complete"] = False
            json.dump(manifest, open(path, "w"))
            assert raises(StaleIndex, SearchEngine, cfg, embedder=HashEmbedder())

    def test_switching_sections_reuses_embeddings_and_restricts_results():
        class NoEmbedding(HashEmbedder):
            def encode_chunks(self, texts):
                raise AssertionError("should not re-embed")

        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d)
            build(cfg)
            only_7 = dict(cfg, sections=["item_7"])
            with contextlib.redirect_stdout(io.StringIO()):
                build_index.build(only_7, cfg["index_dir"], embedder=NoEmbedding())
            engine = SearchEngine(only_7, embedder=HashEmbedder())
            assert engine.search("exact", BOUNDARY_PHRASE) == []
            assert engine.chunks_of("6_2020") == ["6_2020#item_7#0000"]
            chunk = "9_2020#item_7#0001"
            expected = HashEmbedder().encode_chunks([engine.get_text(chunk)])[0]
            assert np.allclose(engine.get_vectors([chunk])[0], expected)
            assert engine.search_by_vector(expected, k=1)[0][0] == chunk

    def test_interrupted_embedding_resumes_without_redoing_work():
        class Counting(HashEmbedder):
            calls = 0

            def encode_chunks(self, texts):
                Counting.calls += len(texts)
                if Counting.fail_after and Counting.calls > Counting.fail_after:
                    raise RuntimeError("simulated crash")
                return super().encode_chunks(texts)

        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d)
            real_block = build_index.EMBED_BLOCK
            build_index.EMBED_BLOCK = 10
            try:
                Counting.fail_after = 15
                with contextlib.redirect_stdout(io.StringIO()):
                    assert raises(RuntimeError, build_index.build, cfg, cfg["index_dir"],
                                  embedder=Counting())
                done = int(open(os.path.join(cfg["index_dir"], "embeddings.progress")).read())
                assert done == 10
                Counting.fail_after = 0
                Counting.calls = 0
                with contextlib.redirect_stdout(io.StringIO()):
                    build_index.build(cfg, cfg["index_dir"], embedder=Counting())
            finally:
                build_index.EMBED_BLOCK = real_block
            engine = SearchEngine(cfg, embedder=HashEmbedder())
            assert Counting.calls == engine.chunk_count() - 10
            chunk = "2_2020#item_1a#0000"
            expected = HashEmbedder().encode_chunks([engine.get_text(chunk)])[0]
            assert np.allclose(engine.get_vectors([chunk])[0], expected)

    def test_flat_index_type_works():
        with tempfile.TemporaryDirectory() as d:
            cfg = make_cfg(d, index_type="flat")
            build(cfg)
            engine = SearchEngine(cfg, embedder=HashEmbedder())
            assert filings(engine.search("embedding", "zygomorphic tractors", k=5))[0] == "2_2020"

    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
            except AssertionError as error:
                failed.append(name)
                print("FAILED:", name, error)
            else:
                print("passed:", name)

if failed:
    sys.exit(f"{len(failed)} of the tests failed: {', '.join(failed)}")
print("all tests passed")
