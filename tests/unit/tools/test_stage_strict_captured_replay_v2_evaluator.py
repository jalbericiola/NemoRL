# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[3]
STAGER_PATH = ROOT / "examples" / "nemo_gym" / "nemotron-3.5-nano" / "stage_strict_captured_replay_v2_evaluator.py"
EVALUATOR_TEST_PATH = ROOT / "tests/unit/tools/test_evaluate_strict_captured_replay_v2.py"


def _load_stager() -> ModuleType:
    spec = importlib.util.spec_from_file_location("strict_replay_v2_evaluator_stager", STAGER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STAGER = _load_stager()
REAL_AUTHENTICATE_RELEASE = STAGER._authenticate_release


def _load_evaluator_fixtures() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "strict_replay_v2_evaluator_fixture_helpers",
        EVALUATOR_TEST_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EVALUATOR_FIXTURES = _load_evaluator_fixtures()


@pytest.fixture(autouse=True)
def _test_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def authenticate_release(
        *, repository_root: str, expected_release: dict[str, str]
    ) -> tuple[dict[str, str], dict[str, bytes]]:
        release = STAGER._normalize_release(expected_release)
        source: dict[str, bytes] = {}
        for relative in STAGER._release_source_paths():
            path = f"{repository_root}/{relative}"
            parent_fd, leaf = STAGER._parent_and_leaf(
                path,
                name=f"test release source {relative}",
            )
            try:
                descriptor = os.open(leaf, STAGER._READ_FLAGS, dir_fd=parent_fd)
            finally:
                os.close(parent_fd)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise STAGER.EvaluatorStageError(f"test release source {relative} metadata differs")
                source[relative] = os.read(descriptor, metadata.st_size + 1)
            finally:
                os.close(descriptor)
        return release, source

    monkeypatch.setattr(STAGER, "_authenticate_release", authenticate_release)
    yield
    paths = sorted(
        (path for path in tmp_path.rglob("*") if not path.is_symlink()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in paths:
        path.chmod(0o700 if path.is_dir() else 0o600)
    tmp_path.chmod(0o700)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _root_owned_python() -> Path:
    candidates = (
        Path("/Library/Developer/CommandLineTools/Library/Frameworks/" "Python3.framework/Versions/3.9/bin/python3.9"),
        Path("/usr/bin/python3"),
        Path(os.path.realpath(getattr(sys, "_base_executable", sys.executable))),
    )
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            continue
        metadata = candidate.stat()
        if (
            metadata.st_uid == 0
            and metadata.st_gid == 0
            and metadata.st_nlink == 1
            and stat.S_IMODE(metadata.st_mode) == 0o755
        ):
            return candidate
    pytest.skip("test host has no root-owned single-link mode-0755 Python")


def _fixture(tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "repository"
    repository.mkdir(parents=True, mode=0o700)
    stager = repository / STAGER.STAGER_REPOSITORY_PATH
    stager.parent.mkdir(parents=True, mode=0o700)
    stager.write_bytes(STAGER_PATH.read_bytes())
    stager.chmod(0o600)
    evaluator = repository / STAGER.EVALUATOR_REPOSITORY_PATH
    evaluator.write_bytes(b"VALUE = 'evaluator-v2'\n")
    evaluator.chmod(0o600)
    python = _root_owned_python()
    companion_sha256: dict[str, str] = {}
    for name, (_, relative) in STAGER.COMPANION_SOURCES.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(f"VALUE = {name!r}\n".encode("ascii"))
        path.chmod(0o600)
        companion_sha256[name] = _sha256(path)
    return {
        "repository_root": str(repository),
        "release": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "signer": STAGER.RELEASE_SIGNER,
            "key_fingerprint": STAGER.RELEASE_KEY_FINGERPRINT,
            "stager_sha256": _sha256(stager),
        },
        "output_root": str(tmp_path / "staged"),
        "python_path": str(python),
        "python_sha256": _sha256(python),
        "evaluator_path": str(evaluator),
        "evaluator_sha256": _sha256(evaluator),
        "producer_root": str(repository),
        "companion_sha256": companion_sha256,
    }


def _cli_arguments(values: dict[str, object]) -> list[str]:
    release = values["release"]
    assert type(release) is dict
    arguments = [
        "stage",
        "--repository-root",
        str(values["repository_root"]),
        "--expected-release-commit",
        release["commit"],
        "--expected-release-tree",
        release["tree"],
        "--expected-release-signer",
        release["signer"],
        "--expected-release-key-fingerprint",
        release["key_fingerprint"],
        "--expected-stager-sha256",
        release["stager_sha256"],
        "--output-root",
        str(values["output_root"]),
        "--python",
        str(values["python_path"]),
        "--expected-python-sha256",
        str(values["python_sha256"]),
        "--evaluator",
        str(values["evaluator_path"]),
        "--expected-evaluator-sha256",
        str(values["evaluator_sha256"]),
        "--producer-root",
        str(values["producer_root"]),
    ]
    companion_sha256 = values["companion_sha256"]
    assert type(companion_sha256) is dict
    for name in STAGER.COMPANION_SOURCES:
        arguments.extend(
            [
                f"--expected-{name.replace('_', '-')}-sha256",
                companion_sha256[name],
            ]
        )
    return arguments


def _assert_mode(path: Path, expected: int) -> None:
    metadata = path.lstat()
    assert stat.S_IMODE(metadata.st_mode) == expected
    assert metadata.st_nlink == 1
    assert metadata.st_uid == os.geteuid()


def _evaluator_request_and_report(
    program: dict[str, str],
    environment: str = "citation",
) -> tuple[dict[str, object], dict[str, object]]:
    request = EVALUATOR_FIXTURES._request(environment)
    request["evaluator_program"] = dict(program)
    EVALUATOR_FIXTURES._rehash_request(request)
    report = EVALUATOR_FIXTURES.EVALUATOR.evaluate_authenticated_request(
        request,
        evaluator_program=program,
        coordinator_api=EVALUATOR_FIXTURES._api(request),
    )
    return request, report


def test_stage_builds_exact_sealed_source_manifest(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    report = STAGER.stage_evaluator_program(**values)
    assert set(report) == {"schema", "manifest"}
    assert report["schema"] == STAGER.STAGE_REPORT_SCHEMA
    assert set(report["manifest"]) == {"path", "sha256"}

    root = Path(str(values["output_root"]))
    manifest_path = root / STAGER.PROGRAM_MANIFEST_FILENAME
    manifest_raw = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == report["manifest"]["sha256"]
    manifest = json.loads(manifest_raw)
    assert manifest_raw == STAGER._canonical_json(manifest)
    assert set(manifest) == {
        "schema",
        "hash_domain",
        "release",
        "python",
        "evaluator",
        "companions",
    }
    assert manifest["schema"] == STAGER.PROGRAM_MANIFEST_SCHEMA
    assert manifest["hash_domain"] == STAGER.HASH_DOMAIN
    assert manifest["release"] == values["release"]
    assert set(manifest["companions"]) == set(STAGER.COMPANION_SOURCES)
    assert set(manifest["python"]) == {"path", "sha256", "size"}
    assert set(manifest["evaluator"]) == {"path", "sha256", "size"}
    assert manifest["evaluator"]["path"] == str(root / STAGER.STAGED_EVALUATOR_FILENAME)
    for name, (module, relative) in STAGER.COMPANION_SOURCES.items():
        reference = manifest["companions"][name]
        assert set(reference) == {"module", "path", "sha256", "size"}
        assert reference["module"] == module
        assert reference["path"] == str(root / "modules" / relative)
        staged = Path(reference["path"])
        assert reference["sha256"] == _sha256(staged)
        assert reference["size"] == staged.stat().st_size
        _assert_mode(staged, 0o400)

    _assert_mode(root / STAGER.STAGED_EVALUATOR_FILENAME, 0o400)
    _assert_mode(manifest_path, 0o400)
    assert stat.S_IMODE(root.stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "modules").stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "modules" / "nemo_rl").stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "modules" / "nemo_rl" / "utils").stat().st_mode) == 0o555
    assert stat.S_IMODE((root / "modules" / "nemo_rl" / "environments").stat().st_mode) == 0o555
    assert {path.name for path in root.iterdir()} == {
        STAGER.STAGED_EVALUATOR_FILENAME,
        STAGER.PROGRAM_MANIFEST_FILENAME,
        "modules",
    }


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong-type"])
def test_stage_rejects_nonexact_companion_digest_roster(
    tmp_path: Path,
    mutation: str,
) -> None:
    values = _fixture(tmp_path)
    supplied = dict(values["companion_sha256"])
    if mutation == "missing":
        supplied.pop("profiles")
    elif mutation == "extra":
        supplied["ambient"] = "1" * 64
    else:
        supplied = dict(supplied)
        values["companion_sha256"] = tuple(supplied.items())
    if mutation != "wrong-type":
        values["companion_sha256"] = supplied
    with pytest.raises(STAGER.EvaluatorStageError, match="wrong exact key set"):
        STAGER.stage_evaluator_program(**values)
    assert not Path(str(values["output_root"])).exists()


def test_stage_closes_first_sibling_directory_when_second_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture(tmp_path)
    mkdir_at = STAGER._mkdir_at
    utils_descriptors: list[int] = []

    def fail_second_sibling(parent_fd: int, name: str) -> int:
        if name == "environments":
            raise OSError("injected second sibling failure")
        descriptor = mkdir_at(parent_fd, name)
        if name == "utils":
            utils_descriptors.append(descriptor)
        return descriptor

    monkeypatch.setattr(STAGER, "_mkdir_at", fail_second_sibling)
    with pytest.raises(OSError, match="second sibling failure"):
        STAGER.stage_evaluator_program(**values)

    assert len(utils_descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(utils_descriptors[0])


def test_stage_closes_both_sibling_directories_when_sealing_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture(tmp_path)
    mkdir_at = STAGER._mkdir_at
    fsync = STAGER.os.fsync
    sibling_descriptors: dict[str, int] = {}

    def track_siblings(parent_fd: int, name: str) -> int:
        descriptor = mkdir_at(parent_fd, name)
        if name in {"utils", "environments"}:
            sibling_descriptors[name] = descriptor
        return descriptor

    def fail_utils_seal(descriptor: int) -> None:
        if descriptor == sibling_descriptors.get("utils"):
            raise OSError("injected directory seal failure")
        fsync(descriptor)

    monkeypatch.setattr(STAGER, "_mkdir_at", track_siblings)
    monkeypatch.setattr(STAGER.os, "fsync", fail_utils_seal)
    with pytest.raises(OSError, match="directory seal failure"):
        STAGER.stage_evaluator_program(**values)

    assert set(sibling_descriptors) == {"utils", "environments"}
    for descriptor in sibling_descriptors.values():
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_digest_poison_is_rejected_before_output_creation(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    values["evaluator_sha256"] = "1" * 64
    with pytest.raises(STAGER.EvaluatorStageError, match="caller-carried SHA-256"):
        STAGER.stage_evaluator_program(**values)
    assert not Path(str(values["output_root"])).exists()


def test_symlinked_source_component_is_rejected(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    producer = Path(str(values["producer_root"]))
    real_utils = producer / "nemo_rl" / "utils"
    moved_utils = producer / "nemo_rl" / "real-utils"
    real_utils.rename(moved_utils)
    real_utils.symlink_to(moved_utils, target_is_directory=True)
    with pytest.raises(OSError):
        STAGER.stage_evaluator_program(**values)
    assert not Path(str(values["output_root"])).exists()


def test_fifo_source_is_rejected_without_blocking(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    _, relative = STAGER.COMPANION_SOURCES["profiles"]
    profile_source = Path(str(values["producer_root"])) / relative
    profile_source.unlink()
    os.mkfifo(profile_source, 0o600)
    with pytest.raises(STAGER.EvaluatorStageError, match="metadata differs"):
        STAGER.stage_evaluator_program(**values)
    assert not Path(str(values["output_root"])).exists()


def test_existing_output_root_is_never_reused(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    output_root = Path(str(values["output_root"]))
    output_root.mkdir(mode=0o700)
    sentinel = output_root / "sentinel"
    sentinel.write_text("owned\n", encoding="ascii")
    with pytest.raises(STAGER.EvaluatorStageError, match="exclusively create"):
        STAGER.stage_evaluator_program(**values)
    assert sentinel.read_text(encoding="ascii") == "owned\n"


def test_cli_emits_one_canonical_lf_terminated_report(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    values = _fixture(tmp_path)
    assert STAGER.main(_cli_arguments(values)) == 0
    captured = capfd.readouterr()
    assert captured.err == ""
    raw = captured.out.encode("ascii")
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    report = json.loads(raw)
    assert raw[:-1] == STAGER._canonical_json(report)
    assert report["schema"] == STAGER.STAGE_REPORT_SCHEMA


def test_stager_imports_only_the_standard_library() -> None:
    tree = ast.parse(STAGER_PATH.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".")[0])
    assert imports <= {
        "__future__",
        "argparse",
        "collections",
        "hashlib",
        "json",
        "math",
        "os",
        "posixpath",
        "re",
        "selectors",
        "signal",
        "stat",
        "subprocess",
        "sys",
        "tempfile",
        "time",
        "typing",
    }


def test_parent_and_isolated_bootstrap_share_exact_module_order() -> None:
    tree = ast.parse(STAGER._ISOLATED_BOOTSTRAP)
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "LOAD_ORDER" for target in node.targets)
    ]
    assert len(assignments) == 1
    assert ast.literal_eval(assignments[0].value) == STAGER._MODULE_LOAD_ORDER


@pytest.mark.parametrize("environment", ("citation", "freeform"))
def test_parent_mirrors_exact_request_report_and_parity(
    environment: str,
) -> None:
    program = {"path": "/staged/evaluator-source-manifest-v1.json", "sha256": _sha256(STAGER_PATH)}
    request, report = _evaluator_request_and_report(program, environment)
    request_raw = EVALUATOR_FIXTURES._canonical(request)
    report_raw = EVALUATOR_FIXTURES._canonical(report)

    validated_request = STAGER._validate_evaluator_request(request_raw, program=program)
    assert validated_request == request
    assert (
        STAGER._validate_evaluator_report(
            report_raw,
            request=validated_request,
            program=program,
        )
        == report
    )


@pytest.mark.parametrize(
    "mutation",
    (
        "bad-parity",
        "integer-parity",
        "boolean-parity",
        "reused-output-digest",
        "negative-zero",
        "bad-passed",
    ),
)
def test_parent_rejects_forged_child_report(mutation: str) -> None:
    program = {"path": "/staged/evaluator-source-manifest-v1.json", "sha256": _sha256(STAGER_PATH)}
    request, report = _evaluator_request_and_report(program)
    if mutation == "bad-parity":
        report["parity"]["samples_sha256"] = "f" * 64
    elif mutation == "integer-parity":
        report["parity"]["reward_vector"] = [1, 1, 1, 1]
    elif mutation == "boolean-parity":
        report["parity"]["reward_vector"] = [True, True, True, True]
    elif mutation == "reused-output-digest":
        report["attempts"]["replay-2"]["outputs"]["replay_ledger"]["sha256"] = report["attempts"]["replay-1"][
            "outputs"
        ]["replay_ledger"]["sha256"]
    elif mutation == "negative-zero":
        report["attempts"]["replay-1"]["samples"][0]["raw_environment_reward"] = -0.0
    else:
        report["attempts"]["replay-1"]["samples"][0]["match_details"]["passed"] = False

    with pytest.raises(STAGER.EvaluatorStageError):
        STAGER._validate_evaluator_report(
            EVALUATOR_FIXTURES._canonical(report),
            request=request,
            program=program,
        )


@pytest.mark.parametrize(
    ("member", "value"),
    (
        ("commit", "A" * 40),
        ("commit", "0" * 40),
        ("tree", "1" * 39),
        ("signer", "attacker@example.com"),
        ("key_fingerprint", "SHA256:attacker"),
        ("stager_sha256", "0" * 64),
    ),
)
def test_release_object_is_exact_and_pinned(member: str, value: str) -> None:
    release = {
        "commit": "1" * 40,
        "tree": "2" * 40,
        "signer": STAGER.RELEASE_SIGNER,
        "key_fingerprint": STAGER.RELEASE_KEY_FINGERPRINT,
        "stager_sha256": "3" * 64,
    }
    release[member] = value
    with pytest.raises(STAGER.EvaluatorStageError):
        STAGER._normalize_release(release)


@pytest.mark.parametrize(
    "body",
    (
        b"tree 1\n\nsubject\n",
        b"tree 1\n\nSigned-off-by: Attacker <a@example.com>\n",
        b"tree 1\n\nprefix Signed-off-by: Jorge Albericio <jalbericiola@nvidia.com>\n",
        (
            b"tree 1\n\n"
            + STAGER.RELEASE_DCO_LINE.encode("ascii")
            + b"\n"
            + STAGER.RELEASE_DCO_LINE.encode("ascii")
            + b"\n"
        ),
        b"tree 1\r\n\n" + STAGER.RELEASE_DCO_LINE.encode("ascii") + b"\n",
        b"tree 1\n\n" + STAGER.RELEASE_DCO_LINE.encode("ascii") + b"\x00\n",
    ),
)
def test_release_commit_dco_poison_is_rejected(body: bytes) -> None:
    with pytest.raises(STAGER.EvaluatorStageError):
        STAGER._validate_commit_body(body)


def test_git_command_prefix_pins_signature_program_and_repository() -> None:
    prefix = STAGER._git_command_prefix(
        repository_root="/release/repository",
        allowed_signers="/private/allowed-signers",
    )
    assert prefix[0] == STAGER.GIT_EXECUTABLE
    assert "gpg.ssh.program=/usr/bin/ssh-keygen" in prefix
    assert "gpg.ssh.allowedSignersFile=/private/allowed-signers" in prefix
    assert prefix[-2:] == ["-C", "/release/repository"]


@pytest.mark.parametrize("mutation", ("none", "dirty", "signature", "blob"))
def test_release_authentication_checks_git_signature_tree_cleanliness_and_blobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    values = _fixture(tmp_path)
    repository = Path(str(values["repository_root"]))
    release = values["release"]
    assert type(release) is dict
    fake_git = tmp_path / "git"
    fake_git.write_bytes(b"authenticated git\n")
    fake_git.chmod(0o755)
    original_stable_open = STAGER._stable_open

    def stable_open(path: str, **kwargs: object):
        if kwargs.get("name") == "Git executable":
            kwargs["root_owned"] = False
        return original_stable_open(path, **kwargs)

    def run_git_command(
        *,
        repository_root: str,
        allowed_signers: str,
        private_home: str,
        arguments: tuple[str, ...],
        stdout_limit: int,
    ) -> tuple[bytes, bytes]:
        assert repository_root == str(repository)
        allowed = Path(allowed_signers)
        assert allowed.read_bytes() == STAGER.RELEASE_ALLOWED_SIGNER.encode("ascii") + b"\n"
        assert stat.S_IMODE(allowed.stat().st_mode) == 0o400
        assert Path(private_home).stat().st_mode & 0o777 == 0o700
        if arguments == ("rev-parse", "--show-toplevel"):
            return (str(repository).encode("ascii") + b"\n", b"")
        if arguments == ("rev-parse", "--verify", "HEAD^{commit}"):
            return (release["commit"].encode("ascii") + b"\n", b"")
        if arguments == ("rev-parse", "--verify", "HEAD^{tree}"):
            return (release["tree"].encode("ascii") + b"\n", b"")
        if arguments == ("verify-commit", release["commit"]):
            return (b"", b"signature diagnostics\n")
        if arguments == (
            "show",
            "-s",
            "--format=%G?%x00%GS%x00%GF%x00",
            release["commit"],
        ):
            signer = b"attacker@example.com" if mutation == "signature" else STAGER.RELEASE_SIGNER.encode("ascii")
            return (
                b"G\x00" + signer + b"\x00" + STAGER.RELEASE_KEY_FINGERPRINT.encode("ascii") + b"\x00\n",
                b"",
            )
        if arguments == ("cat-file", "commit", release["commit"]):
            return (
                b"tree "
                + release["tree"].encode("ascii")
                + b"\n\nsubject\n\n"
                + STAGER.RELEASE_DCO_LINE.encode("ascii")
                + b"\n",
                b"",
            )
        if arguments[:2] == ("cat-file", "blob"):
            relative = arguments[2].split(":", 1)[1]
            raw = (repository / relative).read_bytes()
            if mutation == "blob" and relative == STAGER.EVALUATOR_REPOSITORY_PATH:
                raw += b"poison"
            return raw, b""
        if arguments == (
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--ignore-submodules=none",
        ):
            return (b"?? poison\n" if mutation == "dirty" else b"", b"")
        raise AssertionError(arguments)

    monkeypatch.setattr(STAGER, "_authenticate_release", REAL_AUTHENTICATE_RELEASE)
    monkeypatch.setattr(STAGER, "GIT_EXECUTABLE", str(fake_git))
    monkeypatch.setattr(STAGER, "GIT_EXECUTABLE_SHA256", _sha256(fake_git))
    monkeypatch.setattr(STAGER, "_stable_open", stable_open)
    monkeypatch.setattr(STAGER, "_validate_root_owned_ancestors", lambda *args, **kwargs: None)
    monkeypatch.setattr(STAGER, "_run_git_command", run_git_command)
    monkeypatch.setattr(
        STAGER,
        "__file__",
        str(repository / STAGER.STAGER_REPOSITORY_PATH),
    )

    if mutation == "none":
        authenticated, sources = REAL_AUTHENTICATE_RELEASE(
            repository_root=str(repository),
            expected_release=release,
        )
        assert authenticated == release
        assert set(sources) == set(STAGER._release_source_paths())
    else:
        with pytest.raises(STAGER.EvaluatorStageError):
            REAL_AUTHENTICATE_RELEASE(
                repository_root=str(repository),
                expected_release=release,
            )


def test_process_group_is_signalled_before_leader_reap_and_cleanup_is_one_shot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[object, ...]] = []

    class FakeProcess:
        pid = 43123
        returncode: int | None = None

        def wait(self, *, timeout: float) -> int:
            events.append(("wait", timeout))
            self.returncode = 0
            return 0

        def kill(self) -> None:
            events.append(("kill",))

    process = FakeProcess()

    def waitid(idtype: int, pid: int, flags: int) -> object:
        events.append(("waitid", idtype, pid, flags))
        return object()

    def killpg(pid: int, sig: int) -> None:
        events.append(("killpg", pid, sig))

    monkeypatch.setattr(STAGER.os, "waitid", waitid)
    monkeypatch.setattr(STAGER.os, "killpg", killpg)
    STAGER._wait_for_process_exit_unreaped(
        process,
        deadline=STAGER.time.monotonic() + 1.0,
    )
    STAGER._terminate_process(process)
    events_after_reap = list(events)
    STAGER._terminate_process(process)

    assert [event[0] for event in events_after_reap] == ["waitid", "killpg", "wait"]
    assert events == events_after_reap


def test_verify_run_uses_exact_isolated_process_and_publishes_authenticated_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _fixture(tmp_path)
    stage_report = STAGER.stage_evaluator_program(**values)
    program = stage_report["manifest"]
    request, report = _evaluator_request_and_report(program)
    request_path = tmp_path / "request.json"
    request_raw = EVALUATOR_FIXTURES._canonical(request)
    request_path.write_bytes(request_raw)
    request_path.chmod(0o400)
    report_path = tmp_path / "accepted-report.json"
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FakeProcess:
        pass

    def popen(arguments: list[str], **kwargs: object) -> FakeProcess:
        calls.append((arguments, kwargs))
        return FakeProcess()

    monkeypatch.setattr(STAGER.subprocess, "Popen", popen)
    monkeypatch.setattr(
        STAGER,
        "_collect_process",
        lambda process: (0, EVALUATOR_FIXTURES._canonical(report), b""),
    )
    monkeypatch.setattr(STAGER, "_terminate_process", lambda process: None)
    execution = STAGER.verify_and_run_evaluator(
        repository_root=str(values["repository_root"]),
        release=values["release"],
        program_manifest_path=program["path"],
        program_manifest_sha256=program["sha256"],
        request_path=str(request_path),
        request_sha256=_sha256(request_path),
        report_path=str(report_path),
    )

    assert execution["schema"] == STAGER.EXECUTION_REPORT_SCHEMA
    assert execution["status"] == "authenticated"
    assert report_path.read_bytes() == EVALUATOR_FIXTURES._canonical(report)
    _assert_mode(report_path, 0o400)
    assert len(calls) == 1
    arguments, kwargs = calls[0]
    assert arguments[:6] == [
        values["python_path"],
        "-I",
        "-S",
        "-B",
        "-c",
        STAGER._ISOLATED_BOOTSTRAP,
    ]
    assert kwargs["cwd"] == "/"
    assert kwargs["env"] == STAGER._EXACT_CHILD_ENVIRONMENT
    assert kwargs["close_fds"] is True
    assert kwargs["start_new_session"] is True
    assert len(kwargs["pass_fds"]) == 12


def test_program_manifest_release_must_equal_oob_authority(tmp_path: Path) -> None:
    values = _fixture(tmp_path)
    stage_report = STAGER.stage_evaluator_program(**values)
    wrong_release = dict(values["release"])
    wrong_release["commit"] = "4" * 40
    with pytest.raises(STAGER.EvaluatorStageError, match="release differs"):
        STAGER._open_authenticated_program(
            program_manifest_path=stage_report["manifest"]["path"],
            program_manifest_sha256=stage_report["manifest"]["sha256"],
            expected_release=wrong_release,
            release_sources={
                relative: (Path(str(values["repository_root"])) / relative).read_bytes()
                for relative in STAGER._release_source_paths()
            },
        )


def test_parser_requires_explicit_stage_or_verify_run_command() -> None:
    with pytest.raises(SystemExit):
        STAGER._parser().parse_args([])
