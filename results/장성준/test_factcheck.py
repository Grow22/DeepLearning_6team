"""factcheck.py 시험 (진짜 API 대신 가짜 응답으로). 실행: python -m pytest -q results"""
import csv
import json

import pytest

import factcheck as fc

anthropic = pytest.importorskip("anthropic")


def message(texts, stop="end_turn", inp=3000, out=400, searches=1, cache_read=0):
    """Claude API 응답과 같은 모양의 Message (검색 1번 + 결과 + 인용 + 답)."""
    content = [{"type": "thinking", "thinking": "...", "signature": "sig"},
               {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "corn Paris"}},
               {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                   {"type": "web_search_result", "url": "https://news.example/a", "title": "A", "page_age": None,
                    "encrypted_content": "x"}]},
               {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_2", "content": {
                   "type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}},
               {"type": "text", "text": "Found the original story.", "citations": [{
                   "type": "web_search_result_location", "url": "https://news.example/a", "title": "A",
                   "encrypted_index": "y", "cited_text": "Activists dumped corn"}]}]
    content += [{"type": "text", "text": t, "citations": None} for t in texts]
    return anthropic.types.Message.model_validate({
        "id": "msg", "type": "message", "role": "assistant", "model": fc.MODEL, "content": content,
        "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": inp, "output_tokens": out, "cache_creation_input_tokens": 0,
                  "cache_read_input_tokens": cache_read,
                  "server_tool_use": {"web_search_requests": searches, "web_fetch_requests": 0}}})


def verdict(v, p):
    return json.dumps({"verdict": v, "p_false": p, "reason": "checked"})


class FakeClient:
    """messages.create(**kwargs)를 흉내 낸다. reply(kwargs)가 Message를 돌려주거나 예외를 던진다."""

    def __init__(self, reply):
        self.messages, self.reply, self.calls = self, reply, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply(kwargs)


def captions(n):
    return [{"id": f"t{i}", "text": f"Caption number {i}.", "label_fake": i % 2, "source": "NYT"} for i in range(n)]


def by_label(kwargs):
    """거짓 글(홀수 번호)은 거짓, 사실 글은 사실이라고 답하는 가짜 Claude. 요청마다 $0.0205."""
    i = int(kwargs["messages"][0]["content"].split()[-1].rstrip("."))
    return message([verdict("FALSE", 0.9) if i % 2 else verdict("TRUE", 0.1)], searches=2)


def test_parse_verdict():
    assert fc.parse_verdict('blah {"x": 1}\n' + verdict("FALSE", 1.7)) == \
        {"verdict": "FALSE", "p_false": 1.0, "reason": "checked"}  # 확률은 0~1로 자름
    assert fc.parse_verdict("```json\n" + verdict("insufficient", 0.4) + "\n```")["verdict"] == "INSUFFICIENT"
    assert fc.parse_verdict(verdict("TRUE", 0.2) + " then " + verdict("MAYBE", 0.5))["verdict"] == "TRUE"
    assert fc.parse_verdict("no json here") is None
    assert fc.parse_verdict('{"verdict": "TRUE", "p_false": "high"}') is None


def test_quotes_inside_reason_are_still_read():
    text = 'blah\n{"verdict": "TRUE", "p_false": 0.1, "reason": "confirmed, though "Maria" is not."}'
    assert fc.parse_verdict(text) == {"verdict": "TRUE", "p_false": 0.1, "reason": 'confirmed, though "Maria" is not.'}


def test_unparsed_answers_are_repaired_without_new_calls(tmp_path):
    out = tmp_path / "r.jsonl"
    bad = {"id": "t1", "text": "x", "label_fake": 0, "source": "NYT", "cost": 0.02, "searches": 2,
           "error": fc.UNPARSED + " (stop_reason=end_turn)",
           "answer": 'Checked.\n{"verdict": "TRUE", "p_false": 0.2, "reason": "a "b" c"}'}
    out.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    assert fc.repair_unparsed(out) == 1
    rec = fc.load_records(out)["t1"]
    assert (rec["verdict"], rec["p_false"], rec["repaired"]) == ("TRUE", 0.2, True) and "error" not in rec
    assert fc.usage_so_far(out) == (0.02, 2)  # 다시 읽은 줄은 비용·검색을 두 번 세지 않음
    assert fc.repair_unparsed(out) == 0


def test_request_cost():
    usage = message([], inp=3000, out=400, searches=2).usage
    assert fc.request_cost(fc.MODEL, usage) == pytest.approx((3000 * 0.10 + 400 * 0.50) / 1e6 + 0.02)
    big = message([], inp=150_000, out=1000, searches=0).usage  # 입력이 10만 토큰을 넘으면 비싼 값
    assert fc.request_cost(fc.MODEL, big) == pytest.approx((150_000 * 0.50 + 1000 * 2.50) / 1e6)
    cached = message([], inp=1000, out=0, searches=0, cache_read=10_000).usage
    assert fc.request_cost(fc.MODEL, cached) == pytest.approx((1000 * 0.10 + 10_000 * 0.10 * 0.1) / 1e6)


def test_check_caption_collects_evidence_and_cost():
    client = FakeClient(lambda kw: message(["Let me compare.", verdict("FALSE", 0.85)]))
    rec = fc.check_caption(client, fc.MODEL, "Activists were seen wasting corn.", effort="high")
    assert (rec["verdict"], rec["p_false"], rec["calls"], rec["searches"]) == ("FALSE", 0.85, 1, 1)
    assert rec["queries"] == ["corn Paris"] and rec["urls"] == ["https://news.example/a"]
    assert rec["citations"][0]["cited_text"] == "Activists dumped corn"
    assert rec["search_errors"] == ["max_uses_exceeded"] and "error" not in rec
    sent = client.calls[0]
    assert sent["model"] == fc.MODEL and sent["output_config"] == {"effort": "high"}
    assert sent["tools"][0]["max_uses"] == fc.MAX_SEARCHES and "Caption:" in sent["messages"][0]["content"]
    assert "label" not in json.dumps(sent["messages"])  # 정답은 보내지 않음


def test_pause_turn_is_continued():
    replies = [message(["Searching..."], stop="pause_turn"), message([verdict("TRUE", 0.05)])]
    client = FakeClient(lambda kw: replies.pop(0))
    rec = fc.check_caption(client, fc.MODEL, "A caption.")
    assert rec["verdict"] == "TRUE" and rec["calls"] == 2 and rec["searches"] == 2
    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant" and second[1]["content"][0].type == "thinking"  # 받은 그대로 돌려보냄
    assert rec["cost"] == pytest.approx(2 * fc.request_cost(fc.MODEL, message([]).usage))


def test_unreadable_answer_is_an_error():
    rec = fc.check_caption(FakeClient(lambda kw: message(["I am not sure."], stop="max_tokens")), fc.MODEL, "x")
    assert "error" in rec and "max_tokens" in rec["error"] and rec["cost"] > 0


def test_run_resumes_and_stops_at_budget(tmp_path):
    out = tmp_path / "r.jsonl"
    client = FakeClient(by_label)
    state = fc.run(captions(10), client, fc.MODEL, out, budget=0.08, workers=1)
    assert "예산" in state["stopped"] and len(client.calls) == 4  # 4번째에 $0.082가 되어 멈춤
    state = fc.run(captions(10), client, fc.MODEL, out, budget=1.0, workers=2)
    assert len(client.calls) == 10 and not state["stopped"]  # 끝난 4개는 건너뜀
    records = fc.load_records(out)
    assert len(records) == 10 and all(r["verdict"] == ("FALSE" if r["label_fake"] else "TRUE")
                                       for r in records.values())


def test_errors_are_retried_and_many_in_a_row_stop(tmp_path):
    out = tmp_path / "r.jsonl"

    def broken(kwargs):
        raise RuntimeError("overloaded")
    state = fc.run(captions(10), FakeClient(broken), fc.MODEL, out, budget=1.0, workers=1, max_errors_in_a_row=3)
    assert "오류" in state["stopped"] and len(fc.load_records(out)) == 3
    client = FakeClient(by_label)
    fc.run(captions(10), client, fc.MODEL, out, budget=1.0, workers=1)
    records = fc.load_records(out)
    assert len(client.calls) == 10 and not any(r.get("error") for r in records.values())  # 오류 난 3개도 다시 함


def write_cross(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["item_id", "split", "image_path", "text", "is_ai", "text_fake"])
        writer.writeheader()
        writer.writerows(rows)


def test_test_captions_are_unique_with_one_answer(tmp_path):
    rows = [{"item_id": f"c{i}", "split": s, "image_path": "x.jpg", "text": t, "is_ai": i % 2, "text_fake": f}
            for i, (s, t, f) in enumerate([("test1_nyt_mj", "Real  caption one.", 0),
                                            ("test1_nyt_mj", "Real caption one.", 0),
                                            ("test2_bbc_dalle", "Fake caption.", 1),
                                            ("test4_bbc_sdxl", "Fake caption.", 1)])]
    write_cross(tmp_path / "ai_photos" / "cross_test.csv", rows)
    caps = sorted(fc.load_test_captions(tmp_path), key=lambda c: c["text"])
    assert [(c["text"], c["label_fake"], c["source"]) for c in caps] == \
        [("Fake caption.", 1, "BBC"), ("Real caption one.", 0, "NYT")]
    rows.append({**rows[2], "item_id": "c9", "text_fake": 0})  # 같은 글에 다른 정답
    write_cross(tmp_path / "ai_photos" / "cross_test.csv", rows)
    with pytest.raises(SystemExit):
        fc.load_test_captions(tmp_path)


def test_validation_captions_are_balanced():
    rows = [(f"Real {i}.", 0) for i in range(20)] + [(f"Fake {i}.", 1) for i in range(20)]
    caps = fc.load_validation_captions(10, seed=1, rows=rows)
    assert len(caps) == 10 and sum(c["label_fake"] for c in caps) == 5
    assert caps == fc.load_validation_captions(10, seed=1, rows=rows)


def test_main_end_to_end_with_fake_client(tmp_path):
    rows = [{"item_id": f"c{i}", "split": "test1_nyt_mj" if i < 6 else "test3_cnn_dalle", "image_path": "x.jpg",
             "text": f"Caption number {i}.", "is_ai": 0, "text_fake": i % 2} for i in range(10)]
    write_cross(tmp_path / "ai_photos" / "cross_test.csv", rows)
    client = FakeClient(by_label)
    assert fc.main(["--data", str(tmp_path), "--yes"], client=client) == 0
    table = (tmp_path / "factcheck" / f"{fc.MODEL}-test.md").read_text(encoding="utf-8")
    assert "| **전체** | 10 | 100.0% | 1.000 | 100.0% | 100.0% |" in table
    assert "| CNN | 4 |" in table and "| 거짓 글 | 0 | 5 | 0 |" in table
    assert fc.main(["--data", str(tmp_path), "--report"], client=client) == 0 and len(client.calls) == 10
