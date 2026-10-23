import json

from byosm.cloud.aws import template
from byosm.cloud.sizing import size


def _bundle(tmp_path, mb, ms):
    (tmp_path / "model.int8.onnx").write_bytes(b"0" * int(mb * 1e6))
    manifest = {"params": 1000, "precision": "int8", "quantization": {"tried": {"ops": {"ms": ms}}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def test_a_small_fast_model_gets_the_smallest_fargate_size(tmp_path):
    s = size(_bundle(tmp_path, 20, 4), rps=10)
    assert (s["cpu"], s["memory"]) == (256, 512) and "20 MB of ONNX" in s["why"]


def test_a_big_slow_model_gets_more_cpu_and_memory(tmp_path):
    s = size(_bundle(tmp_path, 1500, 120), rps=20)
    assert s["cpu"] >= 4096 and s["memory"] >= s["need_mb"]


def test_the_template_keeps_the_model_read_only_and_reads_its_token_from_a_secret(tmp_path):
    t = template("intent", size(_bundle(tmp_path, 20, 4), rps=10))
    task = t["Resources"]["Task"]["Properties"]
    box = task["ContainerDefinitions"][0]
    assert box["ReadonlyRootFilesystem"] and box["Secrets"][0]["ValueFrom"] == {"Ref": "TokenSecretArn"}
    assert (task["Cpu"], task["Memory"]) == ("256", "512")
    assert "TokenSecretArn" in t["Parameters"] and "Password" not in json.dumps(t)
    ingress = t["Resources"]["TaskGroup"]["Properties"]["SecurityGroupIngress"][0]
    assert ingress["SourceSecurityGroupId"] == {"Ref": "BalancerGroup"}


def test_the_runtime_asks_for_its_token_but_not_on_the_health_check(monkeypatch):
    import io

    from byosm import rt

    monkeypatch.setenv(rt.TOKEN_ENV, "s3cret")
    handler = rt._handler([], {}, None)

    def call(path, auth=""):
        h = handler.__new__(handler)
        h.path, h.headers, h.wfile, sent = path, {"authorization": auth}, io.BytesIO(), []
        h.send_response = sent.append
        h.send_header = lambda *a: None
        h.end_headers = lambda: None
        h.do_GET()
        return sent[0]

    assert call(rt.HEALTH_PATH) == 200
    assert call("/v1/models") == 401
    assert call("/v1/models", "Bearer s3cret") == 200
