# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import pickle
import stat
import threading
import zlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import nemo_rl.utils.strict_captured_replay_evidence_v2 as evidence
import nemo_rl.utils.strict_captured_replay_manifest_v2 as manifest_module
import nemo_rl.utils.strict_captured_replay_seal_v2 as seal_module

_VERIFY_SEALED_RESULT_V2 = seal_module.verify_sealed_result_v2
_ENVIRONMENT = "citation"
_PROFILE_ID = "citation-string-match-v1"
_PAIR_ID = "pair-final-v2"
_ATTEMPT_ID = "replay-1"
_JOB_ID = "82001"
_BOOT_ID = "12345678-1234-1234-1234-123456789abc"

_CITATION_APP_B85 = (
    "c-pmBTW{Mo6n^)wI8}f}>J&BY!yY`On41e~tZ<4rZC7N0ph&dCRT4FlN@8^V@B2tyEXj700$Uzhi#*rwT)"
    "9W{_a}cGobi+$i()1D@>XB62R)|tlib*&SBI~Dq1ngj$#g=;=NIqKFD94M^BMg_le4qwyXoZebRN(oP0"
    "8{ol1wpq$C6;QSE+c%W0os+Filvl`GPY!rb!XSx9s3ma5&lnT}_funI}w=zGXE<bK}!TCKVSsfrVT{;M"
    "7-+_bVi<L`hk+qFiVy6@v(_XaVPlJ;bcg#B+*8R-`=2V@CH}-&!@b5M%7;T7<aK5vU{37x-"
    "Fq=_u0H;9ISW@rxJt_xC|$4Gn}`zDO$~_2S+1_;fZuZT<U@r%V}W{w}$Mn>Q<p3fK}yH?S~`?ny|BmXh"
    "I43uEiO<eKM8pOjeW`$#fKNx0II-;}xsRAU+Z?Gga0C?{t!r|I0GH<S5v?nB0>>E+w=50~_5a&a-"
    "4T~1Hubbf)r&Q7K#toS;k$?P-zKAoNT!~hf2u!lk#`(Y$E5Z1wEtY^SNR0gSn#Rzi@zw%`nEg3Dv9g{i"
    "Sp@PYbD-%cs^Abo&dB%05?a78y!3g?<)Ie^ukRl_!DvX%=bds-pI^nVQ!JzQ(Wm%+bRFN(srK5s3v}gF"
    "7yr0&WIm<-2TxG#cq*w^&L@8sYLYum>;c-WVzc(PcXej2E;8^4fzVzE`Bqv|2E)ZS@Gd7M@AxsY2^SME"
    "AD9;~xqtPf%BY=D+WEN@5iu^FZY*p^=G|L-XMFoU?M`+Jvcf*NV$erad6>DI=LVEcI_3I}HuAFu^ad=F"
    "$xrEI4HcS{QF;zdHd;o~M746!9*rGLlD&Ql_l&}RUrpqF{XE#;0xQe9<j}BfpIaY}`e<*S~B2^;oT`w5"
    "&c`cp9jm!;sr6p3UJhr+lN-L|vryG&1#*f9avgDDlj615UlTf4i8)Km<2=QA+u?~SU3M<OBy>^WOLBp"
    "{&zX>JSEk*OaK@g&T8K|1?V9B)Wm_x@W$MHrjhfRP!-`ZvyJ>(TRk}cVTL;sM&^OWZ-ETq6l=#y4%qKP"
    "Ix)hX3FTHza7Ic^UNzt!$-QEVD*z6`CIKd!FJ{R3LVEs7$Ji<!_<ljs>HEK5$M6w+NdADggP)U#4)x?!"
    "|$ecd<Be7|eQzK=ew>1z-3XYyM&Ds`SdTJ+DK+dRlb!qPA)v%>ZK8k3_2nqG+?)&JMJ_UOR;(OZ_HbX5"
    "JdXYYPfW?aM>)3+jNYjRh{w#jO$0`9I6EF!7VTFl@ut>qdqy1H&}%`g+CE=nD8%`!8R8&N@Z-j4-ebjb"
    "sY__HH&GR#ekz%40)_JavuvfhPJYc3GSR&Q5seYYtM`n!B<18Qyq#ug)X8qy+!{ia}f;`XIO+D7+soZv"
    "6PbJw!ff!q<SUe#&4b=vN}YbMy}+2MY*NJXUm!Tt5xGskg`MVV_d?LbvoW>$tMU%?@*v4|K@yA}lN1V6"
    "<tXE}iigkVY0lnE4$Fk7R`?pSW1=DHGW6qKpvFsu=-yr7kKVN=~~W-qa|g0EqEebg9`@jR?gv2LZP<-"
    "_1G=z7yh*h)b^XZx+R>lRw-)%7|`9Zh2(pm*_h=pf-`r~1wQjEy=ml_FVCf4YTd)3z|MHa1@O8@XvRRX"
    "mR9uvcRiYcQH52o!cp81iTCcpKG)__^`{Gc=OunTD-WD<7KG-k9cXrue{spO5I}sDr6e)}a9v0@T5wC1"
    "1DmuRT3ZpWPV`<oN^Y4#eZGq5)lH9Z0Gl>)UU`+H>m!tNm#b$!jOjQ_8V7mHzvxyk+qhEcd7=l#yJSR"
    "|ShPuAoqUF<rjWZ?+%=S=p?>Q^01%I%1Pl3F8!4!BLcwi_(;NCBi~br6A$aL|6yzcw*iwJ8Lq+)3q_Bl"
    "~eKJC5t+7@le6LvQ_3iR;WYAzE`kA+CkCULJpr&`3+4ipi}l3h(HxJo7sY~Gi3vN)12(%xvs_QwS>m+o"
    "wE|W13P<Z-0H&qYZoEy4%K@bLlB~>nY&wVL%KkxPAcMH!Se(YN@55h9qpLfKJ;L-<qMWll!bffHO_af-"
    "Ok#@q?wL=S>Bcgld{~cW8u*e+Ms#=+S4|#?D6Bhrm+xxVTj{KpRxLEkFmc7_l;%{G(Z2(4m84b80JyN!"
    "qE8Ygs>aUFRT}nM;~H=Ec4#`>0ey!@U8"
)
_CITATION_CONFIG_B85 = (
    "c-pN}F;c`Z4D5M@K0tCnmyVJT(0YWUTm+8ov3&>mJ$CG5awbsGYFFCTN|FNxHd+kEO&Gd607u?joFp$ixh"
    "pxJiU64)Vu1&iOH&2_NC!7t!#Z?;);4x7KyM~w-2rPK+(i~4rOYSmbK&HH)ORqT^0ao6myYeC0<VcOgD"
    "hTKM*}(1l3+sj5gkSWm`qP9nYTuJ@^EE@s2YVLE68!Di0VQeyiC@aEA7p}W-hI}36|Xf`!Va@$=le3=g"
    "BG(c#ct&46_Y^mUFqpSTq03uRg`v01##ySNL(N4c9nDys}x)C8^&h?u;+YdZ;yu<(#qgS{TsNx+1`Rp#"
    "~SvUn$(y-o4RE3>L$9uugmMbZruAd;cWrBP^4fW#Tsf%t7Is;67%Y-fXw<W?N*l2U?rNt||FizVg(Znu"
    "i^LE-Ii5>ozwQkG#bgDMsDFvqd=)JT|8)Kl;M-EmeYb^*=K#vy0Pjo%CRH"
)
_FORMAT_REQUIREMENTS_B85 = "c-qrVRme-t&DTw@%#BV-EsIrfP|(xU2O=&2as&r+"


def _digest(value: str | bytes) -> str:
    raw = value.encode("ascii") if isinstance(value, str) else value
    return hashlib.sha256(raw).hexdigest()


def _reference(path: Path, schema: str, raw: bytes | None = None) -> dict[str, str]:
    return {
        "path": str(path),
        "schema": schema,
        "sha256": _digest(raw if raw is not None else str(path)),
    }


def _exact_document(keys: frozenset[str], **values: Any) -> dict[str, Any]:
    document = {name: None for name in keys}
    document.update(values)
    assert set(document) == set(keys)
    return document


def _source_transcript() -> dict[str, Any]:
    return {"schema": evidence.TRANSCRIPT_BUNDLE_SCHEMA, "fixture": "off-source"}


def _authenticated_source() -> manifest_module.AuthenticatedOffSourceCapture:
    transcript = _source_transcript()
    return manifest_module.AuthenticatedOffSourceCapture(
        source_capture={},
        pair_manifest={},
        pair_manifest_sha256=_digest("pair-manifest"),
        pair_submission_receipt={},
        pair_submission_receipt_sha256=_digest("pair-submission"),
        trusted_off_exit_receipt_path="/source/EXIT.json",
        trusted_off_exit_receipt_sha256=_digest("source-exit"),
        pre_receipt={},
        pre_receipt_sha256=_digest("source-pre"),
        exit_receipt={},
        exit_receipt_sha256=_digest("source-exit"),
        main_ledger={},
        transcript_bundle=transcript,
        transport_bundle={},
        transport_manifest={},
        transport_records=(),
    )


def _lifecycle(
    tmp_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, bytes],
    str,
]:
    results_root = tmp_path / "authority" / "results"
    result_root = results_root / "captured-replay-result"
    pair_manifest = results_root / "PAIR_MANIFEST.json"
    manifest_path = results_root / "captured_replay" / "manifests" / _PAIR_ID / f"{_ATTEMPT_ID}.json"
    submission_path = results_root / "captured-replay-submission.json"
    scorer_profile = {
        "environment": _ENVIRONMENT,
        "profile_id": _PROFILE_ID,
    }
    source_transcript_raw = evidence.canonical_ascii_json(_source_transcript())
    manifest = {
        "pair_id": _PAIR_ID,
        "environment": _ENVIRONMENT,
        "attempt_id": _ATTEMPT_ID,
        "scorer_profile": scorer_profile,
        "source_capture": {
            "step1_evidence": {
                "transcript_bundle": _reference(
                    results_root / "strict_pair_step1_evidence" / "transcript-bundle.json",
                    evidence.TRANSCRIPT_BUNDLE_SCHEMA,
                    source_transcript_raw,
                )
            }
        },
        "pair": {
            "manifest": _reference(
                pair_manifest,
                evidence.PAIR_MANIFEST_SCHEMA,
            )
        },
        "scheduler_submission": {
            "receipt": {
                "path": str(submission_path),
                "schema": evidence.REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
            }
        },
        "runtime_tools": {
            "document": {
                "container": {
                    "python": {
                        "path": "/usr/local/bin/python3.13",
                        "sha256": _digest("container-python"),
                    }
                }
            }
        },
        "artifacts": {
            "outputs": {
                "directory": {"path": str(result_root)},
                "evidence_index": {
                    "path": str(result_root / "evidence-index.json"),
                    "schema": evidence.REPLAY_POST_INDEX_V2_SCHEMA,
                },
            }
        },
    }
    manifest_ref = _reference(
        manifest_path,
        evidence.REPLAY_EXECUTION_MANIFEST_V2_SCHEMA,
        evidence.canonical_ascii_json(manifest),
    )
    envelope = {
        "pair_id": _PAIR_ID,
        "environment": _ENVIRONMENT,
        "scorer_profile": scorer_profile,
        "arm": "on",
        "mode": "fresh_verifier_reward_replay",
        "attempt_id": _ATTEMPT_ID,
    }
    submission = _exact_document(
        evidence.REPLAY_SUBMISSION_V2_ROOT_KEYS,
        schema=evidence.REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
        phase="SUBMISSION",
        status="complete",
        replay_execution_manifest=manifest_ref,
        candidate_job_id=_JOB_ID,
        **envelope,
    )
    submission_ref = _reference(
        submission_path,
        evidence.REPLAY_SUBMISSION_RECEIPT_V2_SCHEMA,
        evidence.receipt_bytes(submission),
    )
    receipt_paths = evidence._final_receipt_paths_v2(
        manifest,
        authenticated_job_id=_JOB_ID,
    )
    pre = _exact_document(
        evidence.REPLAY_PRE_V2_ROOT_KEYS,
        schema=evidence.REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
        phase="PRE",
        status="complete",
        replay_execution_manifest=manifest_ref,
        submission_receipt=submission_ref,
        candidate_job_id=_JOB_ID,
        authenticated_job_id=_JOB_ID,
        **envelope,
    )
    pre_ref = _reference(
        Path(receipt_paths["pre"]),
        evidence.REPLAY_JOB_PRE_RECEIPT_V2_SCHEMA,
        evidence.receipt_bytes(pre),
    )
    process = {
        "boot_id_sha256": _digest(f"{_BOOT_ID}\n"),
        "pid": 123,
        "start_time_ticks": 456,
    }
    outputs = {
        name: _reference(
            result_root / f"{name}.json",
            f"schema-{name}",
        )
        for name in sorted(evidence.REPLAY_OUTPUT_V2_KEYS)
    }
    exit_receipt = _exact_document(
        evidence.REPLAY_EXIT_V2_ROOT_KEYS,
        schema=evidence.REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
        phase="EXIT",
        status="complete",
        replay_execution_manifest=manifest_ref,
        submission_receipt=submission_ref,
        candidate_job_id=_JOB_ID,
        authenticated_job_id=_JOB_ID,
        pre_receipt=pre_ref,
        driver_exit_code=0,
        driver_process=process,
        runtime_attestation={"original_process_reaped": True},
        outputs=outputs,
        post_verified=True,
        **envelope,
    )
    exit_ref = _reference(
        Path(receipt_paths["exit"]),
        evidence.REPLAY_JOB_EXIT_RECEIPT_V2_SCHEMA,
        evidence.receipt_bytes(exit_receipt),
    )
    evidence_index = _exact_document(
        evidence.REPLAY_POST_INDEX_V2_ROOT_KEYS,
        schema=evidence.REPLAY_POST_INDEX_V2_SCHEMA,
        hash_domain=evidence.HASH_DOMAIN,
        original_process_reaped=True,
        profile_id=_PROFILE_ID,
        replay_execution_manifest=manifest_ref,
        submission_receipt=submission_ref,
        exit_receipt=exit_ref,
        outputs=outputs,
        identity={
            "candidate_job_id": _JOB_ID,
            "authenticated_job_id": _JOB_ID,
            "driver_process": process,
            "run_id": evidence.replay_run_id(
                environment=_ENVIRONMENT,
                pair_id=_PAIR_ID,
                attempt_id=_ATTEMPT_ID,
            ),
        },
        **envelope,
    )
    index_raw = evidence.canonical_ascii_json(evidence_index)
    inventory_raw = evidence.canonical_ascii_json(
        {
            "schema": seal_module.RESULT_INVENTORY_V2_SCHEMA,
            "root": str(result_root),
            "environment": _ENVIRONMENT,
            "profile_id": _PROFILE_ID,
        }
    )
    inventory_path = result_root / seal_module.RESULT_INVENTORY_V2_FILENAME
    projection = {
        "environment": _ENVIRONMENT,
        "profile_id": _PROFILE_ID,
        "result_root": str(result_root),
        "inventory": {
            "path": str(inventory_path),
            "schema": seal_module.RESULT_INVENTORY_V2_SCHEMA,
            "sha256": _digest(inventory_raw),
            "raw": inventory_raw,
        },
        "members": (),
    }
    scorer_resource_raw = evidence.canonical_ascii_json(
        {
            "schema": "nemo-rl-strict-format-verification-resource-v1",
            "process": {
                "boot_id": _BOOT_ID,
                "hostname": "fixture-host",
                "pid": 321,
                "ppid": 123,
                "proc_exe": "/usr/local/bin/python3.13",
                "start_ticks": 654,
                "sys_base_prefix": "/usr/local",
            },
        }
    )
    return (
        manifest,
        submission,
        pre,
        exit_receipt,
        evidence_index,
        projection,
        {
            "evidence-index.json": index_raw,
            "strict_gym_child_runtime/resource.json": scorer_resource_raw,
        },
        receipt_paths["final"],
    )


def _patch_terminal_validation(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest: dict[str, Any],
    projection: dict[str, Any],
    payloads: dict[str, bytes],
) -> None:
    monkeypatch.setattr(
        evidence,
        "_validated_lifecycle_manifest",
        lambda value, **kwargs: manifest,
    )
    monkeypatch.setattr(
        evidence,
        "_validate_terminal_lifecycle_v2",
        lambda **kwargs: [],
    )

    def validated_lifecycle(**kwargs: Any) -> tuple[Any, ...]:
        exit_receipt = kwargs["exit_receipt"]
        receipt_paths = evidence._final_receipt_paths_v2(
            manifest,
            authenticated_job_id=exit_receipt["authenticated_job_id"],
        )
        return (
            manifest,
            kwargs["submission_receipt"],
            kwargs["pre_receipt"],
            exit_receipt,
            receipt_paths,
        )

    monkeypatch.setattr(
        evidence,
        "_validate_lifecycle_before_result_v2",
        validated_lifecycle,
    )
    monkeypatch.setattr(
        evidence,
        "_validate_terminal_result_v2",
        lambda **kwargs: [],
    )
    monkeypatch.setattr(
        evidence,
        "_sealed_result_payloads_v2",
        lambda *args, **kwargs: (projection, payloads, {}),
    )


def _build_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any], tuple[Any, ...], str]:
    lifecycle = _lifecycle(tmp_path)
    manifest, submission, pre, exit_receipt, index, projection, payloads, final_path = lifecycle
    _patch_terminal_validation(
        monkeypatch,
        manifest=manifest,
        projection=projection,
        payloads=payloads,
    )
    document = evidence.build_captured_replay_result_final_receipt_v2(
        replay_execution_manifest=manifest,
        authenticated_source=object(),
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
        evidence_index=index,
        verified_result=object(),
    )
    return document, manifest, lifecycle, final_path


def _write_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_bytes(raw)
    path.chmod(0o400)


def _replace_final(
    *,
    arguments: dict[str, Any],
    document: dict[str, Any],
    mutate: Any,
) -> dict[str, Any]:
    changed = copy.deepcopy(document)
    mutate(changed)
    raw = evidence.receipt_bytes(changed)
    path = Path(arguments["result_final_receipt_path"])
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o400)
    arguments["result_final_receipt_sha256"] = _digest(raw)
    return changed


def _publish_profile_result(
    *,
    root: Path,
    evidence_index: dict[str, Any],
    generation: int,
) -> str:
    from nemo_rl.utils.strict_captured_replay_profiles import (
        get_strict_captured_replay_profile,
    )

    profile = get_strict_captured_replay_profile(
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    child = root / "strict_gym_child_runtime"
    child.mkdir(mode=0o700, exist_ok=True)
    child.chmod(0o700)
    inventory = root / seal_module.RESULT_INVENTORY_V2_FILENAME
    if inventory.exists():
        inventory.chmod(0o600)
        inventory.unlink()
    anchors: dict[str, str] = {}
    for relative, schema in zip(
        profile.result_files,
        profile.result_file_schemas,
        strict=True,
    ):
        path = root / relative
        if path.exists():
            path.chmod(0o600)
        if relative == "evidence-index.json":
            payload = copy.deepcopy(evidence_index)
            payload["fixture_generation"] = generation
        elif relative == profile.scorer_terminal_index_path:
            payload = {
                "schema": schema,
                "environment": _ENVIRONMENT,
                "profile_id": _PROFILE_ID,
                "fixture_generation": generation,
                "quiescence": {
                    "original_process_reaped": True,
                    "wrapper_returncode": 0,
                },
            }
        else:
            payload = {
                "schema": schema,
                "environment": _ENVIRONMENT,
                "profile_id": _PROFILE_ID,
                "fixture_generation": generation,
            }
        raw = evidence.canonical_ascii_json(payload)
        path.write_bytes(raw)
        path.chmod(0o400)
        if relative in profile.result_anchor_paths:
            anchors[relative] = _digest(raw)
    _, inventory_sha256 = seal_module.publish_sealed_result_v2(
        result_root=str(root),
        anchored_sha256=anchors,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    return inventory_sha256


def _loader_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    tuple[Any, ...],
    list[str],
    dict[str, bool],
]:
    document, manifest, lifecycle, final_path = _build_final(tmp_path, monkeypatch)
    _, submission, pre, exit_receipt, _, projection, _, _ = lifecycle
    documents = (
        (
            Path(document["replay_execution_manifest"]["path"]),
            evidence.canonical_ascii_json(manifest),
        ),
        (
            Path(document["submission_receipt"]["path"]),
            evidence.receipt_bytes(submission),
        ),
        (Path(document["pre_receipt"]["path"]), evidence.receipt_bytes(pre)),
        (Path(document["exit_receipt"]["path"]), evidence.receipt_bytes(exit_receipt)),
        (Path(final_path), evidence.receipt_bytes(document)),
    )
    for path, raw in documents:
        _write_immutable(path, raw)

    source = _authenticated_source()
    monkeypatch.setattr(
        manifest_module,
        "_reload_authenticated_off_source_capture",
        lambda value: value,
    )
    monkeypatch.setattr(
        evidence,
        "_authenticate_program_closure_v2",
        lambda value: None,
    )
    sealed_state = {"verified": False}
    sealed_capability = object()

    def verify_result(**kwargs: Any) -> object:
        assert kwargs == {
            "result_root": document["result"]["root"],
            "expected_inventory_sha256": document["result"]["inventory"]["sha256"],
            "expected_environment": _ENVIRONMENT,
            "expected_profile_id": _PROFILE_ID,
        }
        sealed_state["verified"] = True
        return sealed_capability

    monkeypatch.setattr(seal_module, "verify_sealed_result_v2", verify_result)
    loaded_paths: list[str] = []
    original_loader = evidence._load_evidence_document_owned

    def observed_load(**kwargs: Any) -> tuple[dict[str, Any], str, bytes]:
        path = str(kwargs["path"])
        if sealed_state["verified"] and path.startswith(f'{projection["result_root"]}/'):
            raise AssertionError("sealed result member was reopened by pathname")
        loaded_paths.append(path)
        return original_loader(**kwargs)

    monkeypatch.setattr(evidence, "_load_evidence_document_owned", observed_load)
    arguments = {
        "authenticated_source": source,
        "replay_execution_manifest_path": document["replay_execution_manifest"]["path"],
        "replay_execution_manifest_sha256": document["replay_execution_manifest"]["sha256"],
        "submission_receipt_sha256": document["submission_receipt"]["sha256"],
        "candidate_job_id": _JOB_ID,
        "result_final_receipt_path": final_path,
        "result_final_receipt_sha256": _digest(evidence.receipt_bytes(document)),
        "expected_environment": _ENVIRONMENT,
        "expected_profile_id": _PROFILE_ID,
    }
    return arguments, document, lifecycle, loaded_paths, sealed_state


def test_final_builder_freezes_exact_schema_path_and_reference_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, manifest, lifecycle, final_path = _build_final(tmp_path, monkeypatch)
    _, submission, pre, exit_receipt, index, projection, _, _ = lifecycle

    assert set(document) == {
        "schema",
        "hash_domain",
        "phase",
        "status",
        "pair_id",
        "environment",
        "scorer_profile",
        "arm",
        "mode",
        "attempt_id",
        "candidate_job_id",
        "authenticated_job_id",
        "driver_process",
        "original_process_reaped",
        "replay_execution_manifest",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "evidence_index",
        "result",
    }
    assert set(document) == set(evidence.REPLAY_RESULT_FINAL_V2_ROOT_KEYS)
    assert document["schema"] == evidence.REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA
    assert document["phase"] == "FINAL"
    assert document["status"] == "complete"
    assert document["arm"] == "on"
    assert document["mode"] == "fresh_verifier_reward_replay"
    assert document["candidate_job_id"] == document["authenticated_job_id"] == _JOB_ID
    assert document["original_process_reaped"] is True
    assert document["submission_receipt"]["sha256"] == _digest(evidence.receipt_bytes(submission))
    assert document["pre_receipt"]["sha256"] == _digest(evidence.receipt_bytes(pre))
    assert document["exit_receipt"]["sha256"] == _digest(evidence.receipt_bytes(exit_receipt))
    assert document["evidence_index"]["sha256"] == _digest(evidence.canonical_ascii_json(index))
    assert document["result"] == {
        "root": projection["result_root"],
        "inventory": {name: projection["inventory"][name] for name in ("path", "schema", "sha256")},
    }
    assert (
        final_path
        == evidence._final_receipt_paths_v2(
            manifest,
            authenticated_job_id=_JOB_ID,
        )["final"]
    )
    expected_suffix = f"/captured_replay/replay_job_state/{_PAIR_ID}/{_ATTEMPT_ID}/" f"{_JOB_ID}-0/receipts/FINAL.json"
    assert final_path.endswith(expected_suffix)


def test_final_publisher_writes_exactly_one_lf_mode_0400_and_one_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, manifest, lifecycle, final_path = _build_final(tmp_path, monkeypatch)
    _, submission, pre, exit_receipt, index, projection, payloads, _ = lifecycle
    Path(final_path).parent.mkdir(mode=0o700, parents=True)
    _patch_terminal_validation(
        monkeypatch,
        manifest=manifest,
        projection=projection,
        payloads=payloads,
    )

    published, digest = evidence.publish_captured_replay_result_final_receipt_v2(
        output=final_path,
        document=document,
        replay_execution_manifest=manifest,
        authenticated_source=object(),
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
        evidence_index=index,
        verified_result=object(),
    )

    raw = published.read_bytes()
    metadata = os.lstat(published)
    assert published == Path(final_path)
    assert raw == evidence.canonical_ascii_json(document) + b"\n"
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert digest == _digest(raw)
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o400
    assert metadata.st_uid == os.geteuid()
    assert metadata.st_nlink == 1


def test_final_publisher_rejects_alternate_path_before_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, manifest, lifecycle, _ = _build_final(tmp_path, monkeypatch)
    _, submission, pre, exit_receipt, index, projection, payloads, _ = lifecycle
    alternate = tmp_path / "alternate" / "FINAL.json"
    alternate.parent.mkdir(mode=0o700)
    _patch_terminal_validation(
        monkeypatch,
        manifest=manifest,
        projection=projection,
        payloads=payloads,
    )

    with pytest.raises(ValueError, match="authenticated job receipt root"):
        evidence.publish_captured_replay_result_final_receipt_v2(
            output=alternate,
            document=document,
            replay_execution_manifest=manifest,
            authenticated_source=object(),
            expected_environment=_ENVIRONMENT,
            expected_profile_id=_PROFILE_ID,
            submission_receipt=submission,
            pre_receipt=pre,
            exit_receipt=exit_receipt,
            evidence_index=index,
            verified_result=object(),
        )
    assert not alternate.exists()


def test_final_publisher_rejects_schema_shape_or_relabelled_job(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document, manifest, lifecycle, final_path = _build_final(tmp_path, monkeypatch)
    _, submission, pre, exit_receipt, index, projection, payloads, _ = lifecycle
    Path(final_path).parent.mkdir(mode=0o700, parents=True)
    _patch_terminal_validation(
        monkeypatch,
        manifest=manifest,
        projection=projection,
        payloads=payloads,
    )

    for poison in (
        {**copy.deepcopy(document), "extra": True},
        {name: value for name, value in document.items() if name != "result"},
        {**copy.deepcopy(document), "candidate_job_id": "82002"},
    ):
        with pytest.raises(ValueError):
            evidence.publish_captured_replay_result_final_receipt_v2(
                output=final_path,
                document=poison,
                replay_execution_manifest=manifest,
                authenticated_source=object(),
                expected_environment=_ENVIRONMENT,
                expected_profile_id=_PROFILE_ID,
                submission_receipt=submission,
                pre_receipt=pre,
                exit_receipt=exit_receipt,
                evidence_index=index,
                verified_result=object(),
            )
    assert not Path(final_path).exists()


def test_authenticated_result_capability_is_privately_minted_immutable_and_unpickleable() -> None:
    arguments = {
        "authenticated_source": object(),
        "candidate_job_id": _JOB_ID,
        "expected_environment": _ENVIRONMENT,
        "expected_profile_id": _PROFILE_ID,
        "final_path": "/receipt/FINAL.json",
        "final_sha256": _digest("final"),
        "manifest_raw": b"{}",
        "submission_raw": b"{}\n",
        "pre_raw": b"{}\n",
        "exit_raw": b"{}\n",
        "final_raw": b"{}\n",
        "source_transcript_raw": b"{}",
        "result_capability": object(),
    }
    with pytest.raises(ValueError, match="public loader"):
        evidence.AuthenticatedCapturedReplayResultV2(
            _mint_token=object(),
            **arguments,
        )

    capability = evidence.AuthenticatedCapturedReplayResultV2(
        _mint_token=evidence._AUTHENTICATED_REPLAY_RESULT_V2_MINT_TOKEN,
        **arguments,
    )
    with pytest.raises(AttributeError, match="immutable"):
        capability.candidate_job_id = "82002"  # type: ignore[attr-defined]
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(capability)
    with pytest.raises(TypeError, match="cannot be pickled"):
        copy.copy(capability)


def test_loader_mints_only_after_oob_authority_and_never_reopens_a_result_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, _, loaded_paths, sealed_state = _loader_fixture(
        tmp_path,
        monkeypatch,
    )
    _write_immutable(
        Path(document["evidence_index"]["path"]),
        evidence.canonical_ascii_json({"schema": evidence.REPLAY_POST_INDEX_V2_SCHEMA, "poison": True}),
    )

    capability = evidence.load_authenticated_captured_replay_result_v2(**arguments)

    assert type(capability) is evidence.AuthenticatedCapturedReplayResultV2
    assert sealed_state == {"verified": True}
    assert loaded_paths == [
        document["replay_execution_manifest"]["path"],
        arguments["result_final_receipt_path"],
        document["submission_receipt"]["path"],
        document["pre_receipt"]["path"],
        document["exit_receipt"]["path"],
    ]
    assert all(not path.startswith(f'{document["result"]["root"]}/') for path in loaded_paths)


@pytest.mark.parametrize(
    ("argument", "poison"),
    (
        ("replay_execution_manifest_sha256", _digest("wrong-manifest")),
        ("submission_receipt_sha256", _digest("wrong-submission")),
        ("candidate_job_id", "82002"),
        ("result_final_receipt_sha256", _digest("wrong-final")),
    ),
)
def test_loader_rejects_oob_sha_candidate_and_s5_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
    poison: str,
) -> None:
    arguments, _, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)
    arguments[argument] = poison

    with pytest.raises(ValueError):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


def test_loader_rejects_identical_final_bytes_at_an_alternate_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)
    alternate = tmp_path / "attacker-copy" / "FINAL.json"
    _write_immutable(alternate, evidence.receipt_bytes(document))
    arguments["result_final_receipt_path"] = str(alternate)

    with pytest.raises(ValueError, match="authenticated job receipt root"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


def test_loader_rejects_coherently_rehashed_final_and_s5_against_saved_s5_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, lifecycle, _, sealed_state = _loader_fixture(
        tmp_path,
        monkeypatch,
    )
    submission = copy.deepcopy(lifecycle[1])
    submission["candidate_job_id"] = "82002"
    submission_raw = evidence.receipt_bytes(submission)
    submission_path = Path(document["submission_receipt"]["path"])
    submission_path.chmod(0o600)
    submission_path.write_bytes(submission_raw)
    submission_path.chmod(0o400)

    rehashed_final = copy.deepcopy(document)
    rehashed_final["submission_receipt"]["sha256"] = _digest(submission_raw)
    final_raw = evidence.receipt_bytes(rehashed_final)
    final_path = Path(arguments["result_final_receipt_path"])
    final_path.chmod(0o600)
    final_path.write_bytes(final_raw)
    final_path.chmod(0o400)
    arguments["result_final_receipt_sha256"] = _digest(final_raw)

    with pytest.raises(ValueError, match="OOB S5 authority"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


def test_loader_rejects_replaced_final_bytes_against_unchanged_oob_final_sha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)
    original_final_sha256 = arguments["result_final_receipt_sha256"]
    _replace_final(
        arguments=arguments,
        document=document,
        mutate=lambda value: value.__setitem__("status", "attacker-replaced"),
    )
    arguments["result_final_receipt_sha256"] = original_final_sha256

    with pytest.raises(ValueError, match="expected SHA-256"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


def test_loader_rejects_cross_attempt_final_before_result_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)
    _replace_final(
        arguments=arguments,
        document=document,
        mutate=lambda value: value.__setitem__("attempt_id", "replay-2"),
    )

    with pytest.raises(ValueError, match="FINAL envelope"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


def test_loader_rejects_cross_profile_before_result_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)
    arguments["expected_environment"] = "freeform"
    arguments["expected_profile_id"] = "freeform-regex-v1"

    with pytest.raises(ValueError):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


def test_loader_rejects_relabelled_reasoning_gym_with_wrong_outer_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, lifecycle, _, sealed_state = _loader_fixture(
        tmp_path,
        monkeypatch,
    )
    profile_id = "reasoning-gym-exact-match-v2"
    manifest = copy.deepcopy(lifecycle[0])
    manifest["environment"] = "reasoning_gym"
    manifest["scorer_profile"] = {
        "environment": "reasoning_gym",
        "profile_id": profile_id,
    }
    manifest_raw = evidence.canonical_ascii_json(manifest)
    manifest_path = Path(arguments["replay_execution_manifest_path"])
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(manifest_raw)
    manifest_path.chmod(0o400)
    arguments["replay_execution_manifest_sha256"] = _digest(manifest_raw)
    arguments["expected_environment"] = "reasoning_gym"
    arguments["expected_profile_id"] = profile_id

    def relabel_final(value: dict[str, Any]) -> None:
        value["environment"] = "reasoning_gym"
        value["scorer_profile"] = copy.deepcopy(manifest["scorer_profile"])
        value["replay_execution_manifest"]["sha256"] = _digest(manifest_raw)

    _replace_final(
        arguments=arguments,
        document=document,
        mutate=relabel_final,
    )

    registry_called = False

    def forbidden_registry_gate(
        value: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        nonlocal registry_called
        del value, kwargs
        registry_called = True
        raise AssertionError("wrong outer profile passed the public profile gate")

    monkeypatch.setattr(
        evidence,
        "_validated_lifecycle_manifest",
        forbidden_registry_gate,
    )
    with pytest.raises(ValueError, match="unsupported"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert registry_called is False
    assert sealed_state == {"verified": False}


def test_loader_rejects_an_alternate_copied_result_root_before_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)
    alternate_root = tmp_path / "attacker-result-copy"

    def relabel_root(value: dict[str, Any]) -> None:
        value["result"]["root"] = str(alternate_root)
        value["result"]["inventory"]["path"] = str(alternate_root / seal_module.RESULT_INVENTORY_V2_FILENAME)

    _replace_final(
        arguments=arguments,
        document=document,
        mutate=relabel_root,
    )
    with pytest.raises(ValueError, match="M4 output authority"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


def test_loader_rejects_driver_process_disagreement_across_final_and_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)

    def change_process(value: dict[str, Any]) -> None:
        value["driver_process"]["pid"] += 1

    _replace_final(
        arguments=arguments,
        document=document,
        mutate=change_process,
    )
    with pytest.raises(ValueError, match="authenticated EXIT lifecycle"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


def test_loader_rejects_false_reaping_claim_before_result_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)
    _replace_final(
        arguments=arguments,
        document=document,
        mutate=lambda value: value.__setitem__("original_process_reaped", False),
    )

    with pytest.raises(ValueError, match="FINAL envelope"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


@pytest.mark.parametrize(
    ("lifecycle_index", "final_reference", "message"),
    (
        (2, "pre_receipt", "loaded PRE3"),
        (3, "exit_receipt", "loaded EXIT6"),
    ),
)
def test_loader_rejects_poisoned_pre_or_exit_before_result_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle_index: int,
    final_reference: str,
    message: str,
) -> None:
    arguments, document, lifecycle, _, sealed_state = _loader_fixture(
        tmp_path,
        monkeypatch,
    )
    poisoned = copy.deepcopy(lifecycle[lifecycle_index])
    poisoned["schema"] = "attacker-schema"
    poisoned_raw = evidence.receipt_bytes(poisoned)
    poisoned_path = Path(document[final_reference]["path"])
    poisoned_path.chmod(0o600)
    poisoned_path.write_bytes(poisoned_raw)
    poisoned_path.chmod(0o400)

    def bind_poison(value: dict[str, Any]) -> None:
        value[final_reference]["sha256"] = _digest(poisoned_raw)

    _replace_final(
        arguments=arguments,
        document=document,
        mutate=bind_poison,
    )
    with pytest.raises(ValueError, match=message):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


@pytest.mark.parametrize("arm", ("off", "on"))
def test_replay_candidate_must_differ_from_both_authenticated_pair_jobs(
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
) -> None:
    pair_jobs = {"off": "82001", "on": "82002"}
    monkeypatch.setattr(
        evidence,
        "_load_lifecycle_pair_submission_receipt",
        lambda value: {"authenticated_jobs": {name: [{"job_id": job_id}] for name, job_id in pair_jobs.items()}},
    )
    with pytest.raises(ValueError, match="reuses an authenticated Pair"):
        evidence._require_replay_job_disjoint_from_pair(
            {},
            pair_jobs[arm],
            name="candidate_job_id",
        )


def test_loader_rejects_coherently_replaced_sealed_tree_with_stale_oob_final(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, lifecycle, _, sealed_state = _loader_fixture(
        tmp_path,
        monkeypatch,
    )
    index = lifecycle[4]
    result_root = Path(document["result"]["root"])
    monkeypatch.setattr(
        seal_module,
        "verify_sealed_result_v2",
        _VERIFY_SEALED_RESULT_V2,
    )
    first_inventory_sha256 = _publish_profile_result(
        root=result_root,
        evidence_index=index,
        generation=1,
    )

    def bind_first_inventory(value: dict[str, Any]) -> None:
        value["result"]["inventory"]["sha256"] = first_inventory_sha256

    final_with_first_inventory = _replace_final(
        arguments=arguments,
        document=document,
        mutate=bind_first_inventory,
    )
    second_inventory_sha256 = _publish_profile_result(
        root=result_root,
        evidence_index=index,
        generation=2,
    )
    assert second_inventory_sha256 != first_inventory_sha256

    with pytest.raises(Exception, match="inventory"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert final_with_first_inventory["result"]["inventory"]["sha256"] == (first_inventory_sha256)
    assert sealed_state == {"verified": False}


def test_stable_loader_rejects_a_canonical_parent_swap_with_identical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = {"schema": "fixture-parent-swap-v1"}
    raw = evidence.receipt_bytes(document)
    original_parent = tmp_path / "original"
    alternate_parent = tmp_path / "alternate"
    original_path = original_parent / "receipt.json"
    alternate_path = alternate_parent / "receipt.json"
    _write_immutable(original_path, raw)
    _write_immutable(alternate_path, raw)
    original_open = evidence._open_absolute_directory_without_symlinks
    calls = 0

    def swapped_parent(path: Path) -> int:
        nonlocal calls
        calls += 1
        return original_open(alternate_parent if calls == 2 else path)

    monkeypatch.setattr(
        evidence,
        "_open_absolute_directory_without_symlinks",
        swapped_parent,
    )
    with pytest.raises(RuntimeError, match="parent changed"):
        evidence._load_evidence_document_owned(
            path=original_path,
            expected_sha256=_digest(raw),
            trailing_lf=True,
        )
    assert calls == 2


def test_publisher_rejects_a_canonical_parent_swap_with_identical_final_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = {"schema": "fixture-parent-swap-v1"}
    raw = evidence.receipt_bytes(document)
    original_parent = tmp_path / "original"
    alternate_parent = tmp_path / "alternate"
    original_parent.mkdir(mode=0o700)
    original_parent.chmod(0o700)
    alternate_path = alternate_parent / "receipt.json"
    _write_immutable(alternate_path, raw)
    original_open = evidence._open_absolute_directory_without_symlinks
    calls = 0

    def swapped_parent(path: Path) -> int:
        nonlocal calls
        calls += 1
        return original_open(alternate_parent if calls == 2 else path)

    monkeypatch.setattr(
        evidence,
        "_open_absolute_directory_without_symlinks",
        swapped_parent,
    )
    original_path = original_parent / "receipt.json"
    with pytest.raises(RuntimeError, match="exact verification"):
        evidence.publish_evidence_document(
            output=original_path,
            document=document,
            trailing_lf=True,
        )
    assert calls == 2
    assert original_path.read_bytes() == raw


def test_loader_rejects_hardlinked_or_symlinked_final_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, _, _, _, sealed_state = _loader_fixture(tmp_path, monkeypatch)
    final_path = Path(arguments["result_final_receipt_path"])
    hardlink = tmp_path / "hardlink-final.json"
    os.link(final_path, hardlink)
    with pytest.raises(RuntimeError, match="single-link"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}

    hardlink.unlink()
    symlink = tmp_path / "symlink-final.json"
    symlink.symlink_to(final_path)
    arguments["result_final_receipt_path"] = str(symlink)
    with pytest.raises(RuntimeError, match="regular file"):
        evidence.load_authenticated_captured_replay_result_v2(**arguments)
    assert sealed_state == {"verified": False}


@pytest.mark.parametrize(
    "raw",
    (
        b'{"schema":"x","schema":"x"}\n',
        b'{"schema":"x","value":NaN}\n',
        b'{"schema":"x","value":-0.0}\n',
        b'{ "schema": "x" }\n',
        b'{"schema":"x"}',
        b'{"schema":"x"}\n\n',
    ),
)
def test_final_byte_decoder_rejects_noncanonical_or_ambiguous_framing(
    raw: bytes,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        evidence.decode_evidence_document_bytes(
            raw=raw,
            expected_sha256=_digest(raw),
            trailing_lf=True,
        )


def test_loader_signature_requires_saved_s5_candidate_and_final_authority() -> None:
    import inspect

    signature = inspect.signature(evidence.load_authenticated_captured_replay_result_v2)
    assert tuple(signature.parameters) == (
        "authenticated_source",
        "replay_execution_manifest_path",
        "replay_execution_manifest_sha256",
        "submission_receipt_sha256",
        "candidate_job_id",
        "result_final_receipt_path",
        "result_final_receipt_sha256",
        "expected_environment",
        "expected_profile_id",
    )
    with pytest.raises(TypeError):
        evidence.load_authenticated_captured_replay_result_v2(  # type: ignore[call-arg]
            authenticated_source=object(),
            replay_execution_manifest_path="/manifest.json",
            replay_execution_manifest_sha256=_digest("manifest"),
            result_final_receipt_path="/FINAL.json",
            result_final_receipt_sha256=_digest("final"),
            expected_environment=_ENVIRONMENT,
            expected_profile_id=_PROFILE_ID,
        )


def test_authenticated_snapshot_is_exact_fresh_detached_and_filesystem_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, document, lifecycle, _, _ = _loader_fixture(tmp_path, monkeypatch)
    capability = evidence.load_authenticated_captured_replay_result_v2(**arguments)
    _, _, _, _, _, _, _, _ = lifecycle
    samples = [
        {
            "sample_index": index,
            "fixture_row_index": 0,
            "rollout_index": index,
            "generation_seed": 100 + index,
            "model_transport_entry_sha256": _digest(f"entry-{index}"),
            "model_transport_request_body_sha256": _digest(f"request-{index}"),
            "model_transport_response_body_sha256": _digest(f"response-{index}"),
            "model_response_sha256": _digest(f"model-response-{index}"),
            "match_details": {
                "expected": ["fixture-citation"],
                "missing": [] if index % 2 == 0 else ["fixture-citation"],
                "spurious": [],
                "passed": index % 2 == 0,
            },
            "raw_environment_reward": float(index % 2),
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        evidence,
        "_validate_sealed_result_outputs_v2",
        lambda **kwargs: (
            {"scorer_call_index": {"quiescence": {"original_process_reaped": True}}},
            copy.deepcopy(samples),
        ),
    )

    def forbidden_filesystem_load(**kwargs: Any) -> Any:
        del kwargs
        raise AssertionError("snapshot must consume only capability-owned bytes")

    monkeypatch.setattr(
        evidence,
        "_load_evidence_document_owned",
        forbidden_filesystem_load,
    )

    first = evidence.snapshot_authenticated_captured_replay_result_v2(capability)
    first["outputs"]["scorer_call_index"]["path"] = "/mutated"
    first["samples"][0]["match_details"]["mutated"] = True
    second = evidence.snapshot_authenticated_captured_replay_result_v2(capability)

    assert first is not second
    assert set(second) == {
        "schema",
        "pair_id",
        "environment",
        "profile_id",
        "attempt_id",
        "candidate_job_id",
        "authenticated_job_id",
        "run_id",
        "driver_process",
        "scorer_process_identity",
        "manifest",
        "submission_receipt",
        "pre_receipt",
        "exit_receipt",
        "result_final_receipt",
        "result_root",
        "result_inventory",
        "evidence_index",
        "outputs",
        "samples",
    }
    assert set(second) == set(evidence._AUTHENTICATED_RESULT_SNAPSHOT_V2_KEYS)
    assert second["schema"] == evidence.AUTHENTICATED_REPLAY_RESULT_SNAPSHOT_V2_SCHEMA
    assert second["candidate_job_id"] == second["authenticated_job_id"] == _JOB_ID
    assert second["result_final_receipt"] == {
        "path": arguments["result_final_receipt_path"],
        "schema": evidence.REPLAY_RESULT_FINAL_RECEIPT_V2_SCHEMA,
        "sha256": arguments["result_final_receipt_sha256"],
    }
    assert second["result_root"] == document["result"]["root"]
    assert second["outputs"]["scorer_call_index"]["path"] != "/mutated"
    assert "mutated" not in second["samples"][0]["match_details"]
    assert len(second["samples"]) == 4
    assert all(
        set(sample)
        == {
            "sample_index",
            "fixture_row_index",
            "rollout_index",
            "generation_seed",
            "model_transport_entry_sha256",
            "model_transport_request_body_sha256",
            "model_transport_response_body_sha256",
            "model_response_sha256",
            "match_details",
            "raw_environment_reward",
        }
        for sample in second["samples"]
    )
    with pytest.raises(TypeError, match="exact V2 capability"):
        evidence.snapshot_authenticated_captured_replay_result_v2(object())


def test_driver_and_scorer_processes_require_one_boot_and_distinct_identities() -> None:
    driver = {
        "boot_id_sha256": _digest(f"{_BOOT_ID}\n"),
        "pid": 123,
        "start_time_ticks": 456,
    }
    scorer = {
        "boot_id": _BOOT_ID,
        "hostname": "fixture-host",
        "pid": 321,
        "start_ticks": 654,
    }
    admitted_driver, admitted_scorer = evidence._validate_driver_scorer_process_join_v2(
        driver_process=driver,
        scorer_process=scorer,
    )
    assert admitted_driver == driver
    assert admitted_scorer == scorer

    wrong_boot = copy.deepcopy(scorer)
    wrong_boot["boot_id"] = "87654321-4321-4321-4321-cba987654321"
    with pytest.raises(ValueError, match="boot identities differ"):
        evidence._validate_driver_scorer_process_join_v2(
            driver_process=driver,
            scorer_process=wrong_boot,
        )

    same_process = copy.deepcopy(scorer)
    same_process["pid"] = driver["pid"]
    assert same_process["start_ticks"] != driver["start_time_ticks"]
    with pytest.raises(ValueError, match="distinct processes"):
        evidence._validate_driver_scorer_process_join_v2(
            driver_process=driver,
            scorer_process=same_process,
        )


def _run_bounded_fixture_load(
    path: Path,
    *,
    expected_sha256: str,
) -> BaseException:
    outcome: list[BaseException] = []

    def load() -> None:
        try:
            evidence.load_strict_fixture_row0(
                path=path,
                expected_sha256=expected_sha256,
            )
        except BaseException as error:
            outcome.append(error)

    worker = threading.Thread(target=load, daemon=True)
    worker.start()
    worker.join(timeout=1.0)
    assert not worker.is_alive(), "stable fixture reader blocked on a FIFO"
    assert len(outcome) == 1
    return outcome[0]


def test_stable_prompt_fixture_loader_rejects_fifo_without_blocking(
    tmp_path: Path,
) -> None:
    prompt = tmp_path / "prompt.jsonl"
    os.mkfifo(prompt, mode=0o400)
    error = _run_bounded_fixture_load(prompt, expected_sha256=_digest("prompt"))
    assert isinstance(error, RuntimeError)
    assert "bounded regular file" in str(error)


def test_stable_prompt_fixture_loader_rejects_regular_to_fifo_race_without_blocking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = tmp_path / "prompt.jsonl"
    raw = b'{"prompt":"captured"}\n' * 5
    prompt.write_bytes(raw)
    prompt.chmod(0o400)
    real_stat = os.stat
    swapped = False

    def swap_after_named_stat(
        path: str | bytes | int,
        *args: Any,
        **kwargs: Any,
    ) -> os.stat_result:
        nonlocal swapped
        result = real_stat(path, *args, **kwargs)
        if (
            not swapped
            and path == prompt.name
            and kwargs.get("dir_fd") is not None
            and kwargs.get("follow_symlinks") is False
        ):
            swapped = True
            prompt.unlink()
            os.mkfifo(prompt, mode=0o400)
        return result

    monkeypatch.setattr(evidence.os, "stat", swap_after_named_stat)
    error = _run_bounded_fixture_load(prompt, expected_sha256=_digest(raw))
    assert swapped is True
    assert isinstance(error, RuntimeError)
    assert "changed before stable read" in str(error)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("proc_exe", "/opt/alternate/bin/python3.13", "base interpreter differs"),
        ("sys_base_prefix", "/opt/alternate", "base interpreter differs"),
        ("ppid", 124, "parent process differs"),
    ),
)
def test_scorer_resource_process_requires_authenticated_python_and_driver_parent(
    field: str,
    value: str | int,
    message: str,
) -> None:
    manifest = {
        "runtime_tools": {
            "document": {
                "container": {
                    "python": {
                        "path": "/usr/local/bin/python3.13",
                        "sha256": _digest("container-python"),
                    }
                }
            }
        }
    }
    driver = {
        "boot_id_sha256": _digest(f"{_BOOT_ID}\n"),
        "pid": 123,
        "start_time_ticks": 456,
    }
    resource = {
        "boot_id": _BOOT_ID,
        "hostname": "fixture-host",
        "pid": 321,
        "ppid": driver["pid"],
        "proc_exe": "/usr/local/bin/python3.13",
        "start_ticks": 654,
        "sys_base_prefix": "/usr/local",
    }
    admitted_driver, admitted_scorer = evidence._validate_scorer_resource_process_v2(
        replay_execution_manifest=manifest,
        driver_process=driver,
        resource_process=resource,
    )
    assert admitted_driver == driver
    assert admitted_scorer == {name: resource[name] for name in ("boot_id", "hostname", "pid", "start_ticks")}

    poisoned = copy.deepcopy(resource)
    poisoned[field] = value
    with pytest.raises(ValueError, match=message):
        evidence._validate_scorer_resource_process_v2(
            replay_execution_manifest=manifest,
            driver_process=driver,
            resource_process=poisoned,
        )


def _seal_e2e_bytes(path: Path, raw: bytes, *, mode: int = 0o400) -> str:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(mode)
    return hashlib.sha256(raw).hexdigest()


def _e2e_scheduler_raw(record: dict[str, Any]) -> bytes:
    document = {
        "errors": [],
        "jobs": [
            {
                "comment": record["comment"],
                "current_working_directory": record["work_dir"],
                "hold": record["held"],
                "job_id": int(record["job_id"]),
                "job_state": [record["job_state"]],
                "name": record["job_name"],
                "restart_cnt": record["restart_count"],
                "state_reason": record["reason"],
                "user_id": int(record["user_id"]),
            }
        ],
        "last_backfill": {},
        "last_update": {},
        "meta": {},
        "warnings": [],
    }
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )


def _e2e_scheduler_record(
    manifest: dict[str, Any],
    *,
    comment: str,
    phase: str,
) -> dict[str, Any]:
    held = phase == "PRE_RELEASE"
    return {
        "job_id": _JOB_ID,
        "job_name": manifest["scheduler_submission"]["identity"]["job_name"],
        "comment": comment,
        "user_id": str(os.geteuid()),
        "work_dir": manifest["replay_contract"]["source_snapshot"]["ref"]["path"],
        "job_state": "PENDING" if held else "RUNNING",
        "reason": "JobHeldUser" if held else "None",
        "held": held,
        "restart_count": 0,
    }


def _publish_e2e_scheduler_query(
    *,
    manifest: dict[str, Any],
    source: manifest_module.AuthenticatedOffSourceCapture,
    raw_path: Path,
    phase: str,
    comment: str,
) -> dict[str, str]:
    raw_sha256 = _seal_e2e_bytes(
        raw_path,
        _e2e_scheduler_raw(_e2e_scheduler_record(manifest, comment=comment, phase=phase)),
    )
    document = evidence.build_captured_replay_scheduler_query_v2(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        phase=phase,
        raw_output_path=str(raw_path),
        raw_output_sha256=raw_sha256,
        record=_e2e_scheduler_record(manifest, comment=comment, phase=phase),
    )
    path = raw_path.with_name(raw_path.name.removesuffix(".scontrol.raw") + ".scontrol-query.json")
    published, digest = evidence.publish_captured_replay_scheduler_query_v2(
        output=path,
        document=document,
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    return {
        "path": str(published),
        "schema": evidence.REPLAY_SCHEDULER_QUERY_SCHEMA,
        "sha256": digest,
    }


def _e2e_sbatch_argv(
    manifest: dict[str, Any],
    *,
    authenticated_pair: dict[str, Any],
    manifest_path: Path,
    manifest_sha256: str,
    comment: str,
) -> list[str]:
    slurm = authenticated_pair["campaign"]["slurm"]
    snapshot_root = manifest["replay_contract"]["source_snapshot"]["ref"]["path"]
    slurm_root = manifest["execution_environment"]["attempt"]["operational"]["slurm"]
    source_exit = manifest["source_capture"]["job_receipts"]["exit"]
    return [
        "--parsable",
        "--hold",
        f"--chdir={snapshot_root}",
        f"--nodes={authenticated_pair['campaign']['nodes']}",
        f"--account={slurm['account']}",
        f"--job-name={manifest['scheduler_submission']['identity']['job_name']}",
        f"--partition={slurm['partition']}",
        "--time=04:00:00",
        "--gres=gpu:4",
        "--exclusive",
        "--mem=0",
        "--dependency=singleton",
        "--segment=1",
        f"--output={slurm_root}/slurm-%j.out",
        f"--error={slurm_root}/slurm-%j.err",
        f"--qos={slurm['qos']}",
        f"--comment={comment}",
        "--export-file=71",
        "/proc/self/fd/72",
        "--pair-manifest",
        manifest["pair"]["manifest"]["path"],
        "--pair-manifest-sha256",
        manifest["pair"]["manifest"]["sha256"],
        "--pair-submission-receipt",
        manifest["pair"]["submission_receipt"]["path"],
        "--pair-submission-receipt-sha256",
        manifest["pair"]["submission_receipt"]["sha256"],
        "--off-exit-receipt",
        source_exit["path"],
        "--off-exit-receipt-sha256",
        source_exit["sha256"],
        "--replay-manifest",
        str(manifest_path),
        "--replay-manifest-sha256",
        manifest_sha256,
        "--environment",
        _ENVIRONMENT,
        "--profile-id",
        _PROFILE_ID,
    ]


def _embedded_citation_resources() -> dict[str, bytes]:
    compressed = {
        "app": _CITATION_APP_B85,
        "config": _CITATION_CONFIG_B85,
        "requirements": _FORMAT_REQUIREMENTS_B85,
    }
    return {name: zlib.decompress(base64.b85decode(value)) for name, value in compressed.items()}


def _install_e2e_profiled_snapshot(
    root: Path,
    pair: dict[str, Any],
    *,
    profile: Any,
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    snapshot = root / "snapshots" / "on-pair-abc"
    snapshot.mkdir(mode=0o700, parents=True)
    resources = _embedded_citation_resources()
    assert _digest(resources["app"]) == profile.resource_app_sha256
    assert _digest(resources["config"]) == profile.resource_config_sha256
    assert _digest(resources["requirements"]) == profile.requirements_sha256

    selected_config = b"config"
    entrypoint = b"main-entrypoint"
    pair["selection"]["config"]["sha256"] = _digest(selected_config)
    pair["source"]["config_sha256"] = _digest(selected_config)
    pair["source"]["entrypoint_sha256"] = _digest(entrypoint)
    pair["runtime_tools"]["document"]["container"]["python"] = {
        "path": "/usr/local/bin/python3.13",
        "sha256": _digest("container-python-3.13"),
    }
    pair["slurm_export_boundary"]["schema"] = manifest_module.PAIR_SLURM_EXPORT_SCHEMA
    pair["slurm_export_boundary"]["allowed_names"] = list(manifest_module.SLURM_EXPORT_ALLOWED_NAMES)
    pair["wandb"]["group"]["value"] = "citation-pair-abc"
    for arm in ("off", "on"):
        pair["wandb"]["arms"][arm]["name"] = f"{arm}-citation-pair-abc"

    payloads = {
        relative: (repo_root / relative).read_bytes() for relative in manifest_module.REPLAY_PROGRAM_V2_PATHS.values()
    }
    payloads.update(
        {
            pair["selection"]["config"]["path"]: selected_config,
            "examples/run_grpo_single_controller.py": entrypoint,
            profile.fixture_path: (repo_root / profile.fixture_path).read_bytes(),
            ("3rdparty/Gym-workspace/Gym/" + profile.resource_app_path): resources["app"],
            ("3rdparty/Gym-workspace/Gym/" + profile.resource_config_path): resources["config"],
            ("3rdparty/Gym-workspace/Gym/" + profile.requirements_path): resources["requirements"],
        }
    )
    assert _digest(payloads[profile.fixture_path]) == profile.fixture_sha256
    executable = {
        manifest_module.REPLAY_PROGRAM_V2_PATHS["job_wrapper"],
        manifest_module.REPLAY_PROGRAM_V2_PATHS["submission_launcher"],
    }
    for relative, raw in payloads.items():
        path = snapshot / relative
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_bytes(raw)

    symlink_name = "strict-pair-snapshot-symlinks.json"
    mode_name = "strict-pair-snapshot-modes.json"
    symlink_raw = (
        evidence.canonical_ascii_json(
            {
                "schema": "nemo-rl-strict-snapshot-symlinks-v1",
                "symlinks": {},
            }
        )
        + b"\n"
    )
    (snapshot / symlink_name).write_bytes(symlink_raw)
    regular_paths = [*payloads, symlink_name, mode_name]
    mode_raw = (
        evidence.canonical_ascii_json(
            {
                "schema": "nemo-rl-strict-snapshot-modes-v1",
                "regular_file_executable": {relative: relative in executable for relative in regular_paths},
            }
        )
        + b"\n"
    )
    (snapshot / mode_name).write_bytes(mode_raw)
    manifest_raw = b"".join(
        _digest((snapshot / relative).read_bytes()).encode("ascii") + b"  " + relative.encode("ascii") + b"\n"
        for relative in sorted(regular_paths)
    )
    manifest_path = snapshot / "strict-pair-snapshot-manifest.sha256"
    manifest_path.write_bytes(manifest_raw)
    for relative in regular_paths:
        (snapshot / relative).chmod(0o500 if relative in executable else 0o400)
    manifest_path.chmod(0o400)
    for directory in sorted(
        (path for path in snapshot.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    snapshot.chmod(0o500)
    pair["source"]["snapshots"]["on"] = {
        "config_sha256": _digest(selected_config),
        "entrypoint_sha256": _digest(entrypoint),
        "manifest_sha256": _digest(manifest_raw),
        "path": str(snapshot),
    }


def _rewrite_e2e_document(path: Path, document: dict[str, Any], *, lf: bool) -> str:
    raw = evidence.canonical_ascii_json(document) + (b"\n" if lf else b"")
    path.chmod(0o600)
    path.write_bytes(raw)
    path.chmod(0o400)
    return _digest(raw)


def _e2e_real_citation_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    manifest_module.AuthenticatedOffSourceCapture,
    manifest_module.AuthenticatedReplayStaticInputs,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    import tests.unit.utils.test_strict_captured_replay_manifest as source_fixture
    import tests.unit.utils.test_strict_model_transport as transport_fixture
    from nemo_rl.utils.strict_captured_replay_profiles import (
        get_strict_captured_replay_profile,
    )
    from tests.unit.utils.test_strict_captured_replay_evidence import (
        _derivation,
        _format_entry,
        _format_fixture_row,
        _generation,
        _ledger_rows,
    )
    from nemo_rl.utils.strict_main_step_ledger import build_main_step1_ledger

    profile = get_strict_captured_replay_profile(
        expected_environment="citation",
        expected_profile_id="citation-string-match-v1",
    )
    original_pair = source_fixture._pair
    original_call = source_fixture.build_model_transport_call
    original_bundle = source_fixture.build_model_transport_bundle
    original_transport_manifest = source_fixture.build_model_transport_manifest
    original_transcript = source_fixture.build_transcript_bundle

    def citation_source_fixture_row() -> dict[str, Any]:
        fixture = copy.deepcopy(_format_fixture_row("citation"))
        fixture["verifier"] = {
            "type": "string_match",
            "patterns": ["Zoey"],
            "expected_markers": ["Zoey"],
        }
        return fixture

    def profiled_pair(root: str) -> dict[str, Any]:
        pair = original_pair(root)
        pair["selection"]["environment"] = "citation"
        fixture_path = f"{pair['source']['root']}/{profile.fixture_path}"
        pair["artifacts"]["fixture"] = {
            "path": fixture_path,
            "rows": profile.fixture_rows,
            "sha256": profile.fixture_sha256,
        }
        pair["execution_environment"]["fixed"]["train_path"] = fixture_path
        pair["execution_environment"]["fixed"]["val_path"] = fixture_path
        pair["selection"]["gym_resources"] = {
            "config": {
                "path": profile.resource_config_path,
                "sha256": profile.resource_config_sha256,
            },
            "requirements": {
                "path": profile.requirements_path,
                "sha256": profile.requirements_sha256,
            },
            "verifier_source": {
                "path": profile.resource_app_path,
                "sha256": profile.resource_app_sha256,
            },
        }
        return pair

    def profiled_snapshot(
        root: Path,
        pair: dict[str, Any],
        *,
        snapshot_forgery: str | None = None,
    ) -> None:
        assert snapshot_forgery is None
        _install_e2e_profiled_snapshot(root, pair, profile=profile)

    def profiled_call(**kwargs: Any) -> dict[str, Any]:
        kwargs["environment"] = "citation"
        return original_call(**kwargs)

    def profiled_bundle(**kwargs: Any) -> dict[str, Any]:
        kwargs["environment"] = "citation"
        return original_bundle(**kwargs)

    def profiled_transport_manifest(**kwargs: Any) -> dict[str, Any]:
        kwargs["environment"] = "citation"
        return original_transport_manifest(**kwargs)

    def profiled_entry(index: int, transport: dict[str, Any]) -> dict[str, Any]:
        entry = copy.deepcopy(_format_entry(index, "citation"))
        verifier = copy.deepcopy(citation_source_fixture_row()["verifier"])
        entry["agent_run_request"]["verifier"] = copy.deepcopy(verifier)
        entry["derived_verifier_request"]["verifier"] = copy.deepcopy(verifier)
        entry["verifier_response"]["verifier"] = copy.deepcopy(verifier)
        passed = bool(index % 2)
        entry["verifier_response"]["reward"] = float(passed)
        entry["verifier_response"]["match_details"] = {
            "expected": ["Zoey"],
            "missing": [] if passed else ["Zoey"],
            "spurious": [],
            "passed": passed,
        }
        entry["raw_environment_reward"] = float(passed)
        entry["model_transport_entry_sha256"] = transport["entry_sha256"]
        entry["model_transport_request_body_sha256"] = transport["request_body_sha256"]
        entry["model_transport_response_body_sha256"] = transport["response_body_sha256"]
        return entry

    def profiled_transcript(**kwargs: Any) -> dict[str, Any]:
        kwargs["environment"] = "citation"
        kwargs["fixture_row"] = citation_source_fixture_row()
        entries = copy.deepcopy(kwargs["entry_inputs"])
        for entry in entries:
            entry["verifier_response"].pop("extracted_answer", None)
        kwargs["entry_inputs"] = entries
        return original_transcript(**kwargs)

    monkeypatch.setattr(source_fixture, "_pair", profiled_pair)
    monkeypatch.setattr(source_fixture, "_seal_pair_on_snapshot", profiled_snapshot)
    monkeypatch.setattr(
        source_fixture,
        "SLURM_EXPORT_ALLOWED_NAMES",
        manifest_module.SLURM_EXPORT_ALLOWED_NAMES,
    )
    monkeypatch.setattr(source_fixture, "build_model_transport_call", profiled_call)
    monkeypatch.setattr(
        source_fixture,
        "build_model_transport_bundle",
        profiled_bundle,
    )
    monkeypatch.setattr(
        source_fixture,
        "build_model_transport_manifest",
        profiled_transport_manifest,
    )
    monkeypatch.setattr(
        source_fixture,
        "build_transcript_bundle",
        profiled_transcript,
    )
    monkeypatch.setattr(
        transport_fixture,
        "_fixture_row",
        lambda: copy.deepcopy(citation_source_fixture_row()),
    )
    monkeypatch.setattr(transport_fixture, "_transcript_entry", profiled_entry)

    source_root = tmp_path / "real-source"
    source_root.mkdir(mode=0o700)
    source_kwargs = source_fixture._authenticated_source_fixture(source_root)
    pair = source_kwargs["pair_manifest"]
    pair_path = Path(source_kwargs["pair_manifest_path"])
    pair_sha256 = source_kwargs["pair_manifest_sha256"]
    pair_receipt_path = Path(source_kwargs["pair_submission_receipt_path"])
    pair_receipt = json.loads(pair_receipt_path.read_text(encoding="ascii"))
    for arm in ("off", "on"):
        expected_name = f"{arm}-citation-{pair['pair_id']}"
        pair_receipt["authenticated_jobs"][arm][0]["job_name"] = expected_name
        for phase in ("pre_release_query", "post_release_query"):
            pair_receipt[phase]["records"][arm][0]["job_name"] = expected_name
    pair_receipt_sha256 = _rewrite_e2e_document(
        pair_receipt_path,
        pair_receipt,
        lf=True,
    )

    results_root = Path(pair["paths"]["results_root"])
    step1_root = results_root / "off" / "strict_pair_step1_evidence"
    transport_root = results_root / "off" / "strict_model_transport"
    transport_bundle_path = transport_root / "model-transport-bundle.json"
    transport_bundle = json.loads(transport_bundle_path.read_text(encoding="ascii"))
    transport_bundle_ref = {
        "path": str(transport_bundle_path),
        "schema": manifest_module.TRANSPORT_BUNDLE_SCHEMA,
        "sha256": _digest(transport_bundle_path.read_bytes()),
    }
    source_entries = [copy.deepcopy(_format_entry(index, "citation")) for index in range(4)]
    for entry, transport in zip(
        source_entries,
        transport_bundle["entries"],
        strict=True,
    ):
        entry["model_transport_entry_sha256"] = transport["entry_sha256"]
        entry["model_transport_request_body_sha256"] = transport["request_body_sha256"]
        entry["model_transport_response_body_sha256"] = transport["response_body_sha256"]
    transcript_path = step1_root / "transcript-bundle.json"
    transcript = evidence.build_transcript_bundle(
        pair_id=pair["pair_id"],
        environment="citation",
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings={
            "pair_manifest_sha256": pair_sha256,
            "submission_receipt_sha256": pair_receipt_sha256,
            "job_id": "6787903",
            "run_id": pair["wandb"]["arms"]["off"]["run_id"],
            "fixture_sha256": pair["artifacts"]["fixture"]["sha256"],
            "verifier_source_sha256": pair["selection"]["gym_resources"]["verifier_source"]["sha256"],
            "config_sha256": pair["selection"]["config"]["sha256"],
            "snapshot_manifest_sha256": pair["source"]["snapshots"]["off"]["manifest_sha256"],
        },
        fixture_row=_format_fixture_row("citation"),
        model_transport_bundle=transport_bundle_ref,
        verifier_request_derivation=_derivation(),
        entry_inputs=source_entries,
    )
    transcript_sha256 = _rewrite_e2e_document(
        transcript_path,
        transcript,
        lf=False,
    )
    transcript_ref = {
        "path": str(transcript_path),
        "schema": manifest_module.TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": transcript_sha256,
    }

    ledger_path = step1_root / "main-ledger.json"
    ledger_rows = _ledger_rows(transcript)
    for row in ledger_rows:
        row["advantages"] = [0.0] * len(row["token_ids"])
    ledger = build_main_step1_ledger(
        pair_id=pair["pair_id"],
        environment="citation",
        arm="off",
        mode="observe",
        generation=transcript["generation"],
        bindings={
            **transcript["bindings"],
            "restart_count": 0,
            "pair_campaign_sha256": pair["pair_campaign_sha256"],
            "pair_campaign_reward_and_advantage_sha256": pair["pair_campaign_reward_and_advantage_sha256"],
        },
        transcript_bundle=transcript_ref,
        row_inputs=ledger_rows,
        update_successful=True,
    )
    ledger_sha256 = _rewrite_e2e_document(ledger_path, ledger, lf=False)
    ledger_ref = {
        "path": str(ledger_path),
        "schema": manifest_module.MAIN_LEDGER_SCHEMA,
        "sha256": ledger_sha256,
    }

    transport_manifest_path = transport_root / "model-transport-manifest.json"
    transport_manifest = json.loads(transport_manifest_path.read_text(encoding="ascii"))
    transport_manifest["submission_receipt_sha256"] = pair_receipt_sha256
    transport_manifest["main_transcript_bundle"] = copy.deepcopy(transcript_ref)
    transport_manifest["main_ledger"] = copy.deepcopy(ledger_ref)
    transport_manifest_sha256 = _rewrite_e2e_document(
        transport_manifest_path,
        transport_manifest,
        lf=False,
    )

    receipt_root = results_root / "off" / "strict_pair_job_state" / "6787903-0" / "receipts"
    pre_path = receipt_root / "PRE.json"
    pre = json.loads(pre_path.read_text(encoding="ascii"))
    pre["environment"] = "citation"
    pre["job_name"] = f"off-citation-{pair['pair_id']}"
    pre["submission_receipt_sha256"] = pair_receipt_sha256
    pre_sha256 = _rewrite_e2e_document(pre_path, pre, lf=True)
    exit_path = receipt_root / "EXIT.json"
    exit_receipt = json.loads(exit_path.read_text(encoding="ascii"))
    for key in manifest_module.PAIR_PRE_RECEIPT_KEYS - {"phase", "post_verified"}:
        exit_receipt[key] = copy.deepcopy(pre[key])
    exit_receipt["pre_receipt_sha256"] = pre_sha256
    exit_receipt["step1_evidence"]["transcript_bundle"] = transcript_ref
    exit_receipt["step1_evidence"]["main_ledger"] = ledger_ref
    exit_receipt["step1_evidence"]["model_transport"]["manifest"]["sha256"] = transport_manifest_sha256
    exit_sha256 = _rewrite_e2e_document(exit_path, exit_receipt, lf=True)
    source_kwargs["pair_submission_receipt_sha256"] = pair_receipt_sha256
    source_kwargs["trusted_off_exit_receipt_sha256"] = exit_sha256

    source = manifest_module.load_authenticated_off_source_capture(**source_kwargs)
    source_fixture._seal_replay_export(
        source_root,
        pair=source.pair_manifest,
        attempt_id=_ATTEMPT_ID,
    )
    contract = manifest_module.build_replay_submission_contract(
        authenticated_source=source,
        attempt_id=_ATTEMPT_ID,
        submission_nonce=_digest("real-citation-replay-contract"),
    )
    contract_path = Path(
        manifest_module._submission_contract_path(
            source.pair_manifest,
            attempt_id=_ATTEMPT_ID,
        )
    )
    contract_path.parent.mkdir(mode=0o700, parents=True)
    manifest_module.publish_replay_submission_contract(
        authenticated_source=source,
        attempt_id=_ATTEMPT_ID,
        document=contract,
    )

    def local_container_identity(pair_document: dict[str, Any]) -> dict[str, Any]:
        return {
            **copy.deepcopy(pair_document["artifacts"]["container"]),
            "owner_uid": manifest_module.REPLAY_CONTAINER_OWNER_UID,
            "owner_gid": manifest_module.REPLAY_CONTAINER_OWNER_GID,
        }

    monkeypatch.setattr(
        manifest_module,
        "_stable_container_asset_identity",
        local_container_identity,
    )
    manifest_module.load_authenticated_replay_static_inputs(
        authenticated_source=source,
        attempt_id=_ATTEMPT_ID,
    )
    static_inputs = manifest_module._load_authenticated_replay_static_inputs_v2(
        authenticated_source=source,
        attempt_id=_ATTEMPT_ID,
        profile=profile,
    )
    source_entries = [
        {key: copy.deepcopy(entry[key]) for key in evidence.TRANSCRIPT_ENTRY_INPUT_KEYS}
        for entry in source.transcript_bundle["entries"]
    ]
    assert pair_path.read_bytes() == evidence.canonical_ascii_json(pair) + b"\n"
    assert _digest(pair_path.read_bytes()) == pair_sha256
    return source, static_inputs, pair, pair_receipt, source_entries


def _e2e_profiled_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    manifest_module.AuthenticatedOffSourceCapture,
    manifest_module.AuthenticatedReplayStaticInputs,
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
]:
    from tests.unit.utils.test_strict_captured_replay_evidence import (
        _derivation,
        _format_entry,
        _format_fixture_row,
        _generation,
    )
    from tests.unit.utils.test_strict_captured_replay_manifest_v2 import (
        _profiled_authority,
    )

    if _ENVIRONMENT == "citation":
        return _e2e_real_citation_authority(tmp_path, monkeypatch)

    _, base_source, static_inputs = _profiled_authority(
        tmp_path,
        environment=_ENVIRONMENT,
        profile_id=_PROFILE_ID,
    )
    pair = copy.deepcopy(base_source.pair_manifest)
    pair["campaign"].update(
        {
            "nodes": 1,
            "slurm": {
                "account": "nemotron_sw_post",
                "partition": "batch",
                "qos": "normal",
            },
            "reward_and_advantage": {"strict": True},
        }
    )
    tool_root = tmp_path / "trusted-tools"
    tool_root.mkdir(mode=0o700)
    host_tools: dict[str, dict[str, str]] = {}
    for filename in ("env", "python", "sbatch", "scancel", "scontrol", "nvidia-smi"):
        path = tool_root / filename
        raw = f"tool-{filename}".encode("ascii")
        _seal_e2e_bytes(path, raw, mode=0o500)
        name = "nvidia_smi" if filename == "nvidia-smi" else filename
        host_tools[name] = {"path": str(path), "sha256": _digest(raw)}
    tool_root.chmod(0o500)
    pair["runtime_tools"]["document"] = {
        "schema": "nemo-rl-strict-runtime-tools-v2",
        "host": host_tools,
        "container": {
            "python": {
                "path": "/usr/local/bin/python3.13",
                "sha256": _digest("container-python"),
            }
        },
    }
    pair["scheduler_submission"]["identity"]["submitter_euid"] = os.geteuid()
    pair_raw = evidence.canonical_ascii_json(pair) + b"\n"
    pair_sha256 = _digest(pair_raw)

    slurm_conf = tmp_path / "slurm.conf"
    slurm_conf_sha256 = _seal_e2e_bytes(slurm_conf, b"ClusterName=strict\n")
    client_environment = {
        "ambient_merge": False,
        "env": copy.deepcopy(host_tools["env"]),
        "variables": {
            "LC_ALL": "C",
            "SLURM_CONF": {
                "path": str(slurm_conf),
                "sha256": slurm_conf_sha256,
            },
        },
    }
    scheduler_tools = {
        "client_environment": client_environment,
        "sbatch": copy.deepcopy(host_tools["sbatch"]),
        "scancel": copy.deepcopy(host_tools["scancel"]),
        "scontrol": copy.deepcopy(host_tools["scontrol"]),
    }
    pair_receipt = {
        "schema": manifest_module.PAIR_SUBMISSION_RECEIPT_SCHEMA,
        "authenticated_jobs": {
            "off": [{"job_id": "6787903"}],
            "on": [{"job_id": "6787904"}],
        },
        "scheduler_tools": scheduler_tools,
    }
    pair_receipt_raw = evidence.receipt_bytes(pair_receipt)
    pair_receipt_sha256 = _digest(pair_receipt_raw)

    pair_manifest_path = Path(pair["paths"]["results_root"]) / "PAIR_MANIFEST.json"
    pair_receipt_path = Path(pair["paths"]["results_root"]) / "PAIR_SUBMISSION_RECEIPT.json"
    _seal_e2e_bytes(pair_manifest_path, pair_raw)
    _seal_e2e_bytes(pair_receipt_path, pair_receipt_raw)

    source_capture = copy.deepcopy(base_source.source_capture)
    source_capture["authenticated_job"]["comment"] = (
        "nemo-rl-strict-pair-v1:off:" f"{pair['scheduler_submission']['nonce']}:{pair_sha256}"
    )
    source_entries = [copy.deepcopy(_format_entry(index, _ENVIRONMENT)) for index in range(4)]
    source_transcript = evidence.build_transcript_bundle(
        pair_id=pair["pair_id"],
        environment=_ENVIRONMENT,
        arm="off",
        mode="observe",
        attempt_id=None,
        generation=_generation(),
        bindings={
            "pair_manifest_sha256": pair_sha256,
            "submission_receipt_sha256": pair_receipt_sha256,
            "job_id": source_capture["authenticated_job"]["job_id"],
            "run_id": _digest("source-run"),
            "fixture_sha256": pair["artifacts"]["fixture"]["sha256"],
            "verifier_source_sha256": pair["selection"]["gym_resources"]["verifier_source"]["sha256"],
            "config_sha256": pair["selection"]["config"]["sha256"],
            "snapshot_manifest_sha256": pair["source"]["snapshots"]["off"]["manifest_sha256"],
        },
        fixture_row=_format_fixture_row(_ENVIRONMENT),
        model_transport_bundle=source_capture["step1_evidence"]["model_transport"]["bundle"],
        verifier_request_derivation=_derivation(),
        entry_inputs=source_entries,
    )
    source_transcript_ref = source_capture["step1_evidence"]["transcript_bundle"]
    source_transcript_raw = evidence.canonical_ascii_json(source_transcript)
    source_transcript_ref["sha256"] = _digest(source_transcript_raw)
    _seal_e2e_bytes(Path(source_transcript_ref["path"]), source_transcript_raw)
    source = replace(
        base_source,
        source_capture=source_capture,
        pair_manifest=pair,
        pair_manifest_sha256=pair_sha256,
        pair_submission_receipt=pair_receipt,
        pair_submission_receipt_sha256=pair_receipt_sha256,
        transcript_bundle=source_transcript,
    )

    snapshot_root = Path(static_inputs.source_snapshot["path"])
    for name, reference in static_inputs.replay_program.items():
        raw = name.encode("ascii")
        assert _digest(raw) == reference["sha256"]
        _seal_e2e_bytes(snapshot_root / reference["path"], raw)

    monkeypatch.setattr(
        manifest_module,
        "_reload_authenticated_off_source_capture",
        lambda value: value,
    )
    monkeypatch.setattr(
        manifest_module,
        "_load_authenticated_replay_static_inputs_v2",
        lambda **kwargs: static_inputs,
    )
    return source, static_inputs, pair, pair_receipt, source_entries


@pytest.mark.parametrize(
    ("environment", "profile_id"),
    (
        ("citation", "citation-string-match-v1"),
        ("freeform", "freeform-regex-v1"),
    ),
)
def test_public_final_lifecycle_profiled_round_trip_with_real_sealer_and_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment: str,
    profile_id: str,
) -> None:
    monkeypatch.setitem(globals(), "_ENVIRONMENT", environment)
    monkeypatch.setitem(globals(), "_PROFILE_ID", profile_id)
    source, static_inputs, pair, pair_receipt, source_entries = _e2e_profiled_authority(tmp_path, monkeypatch)
    manifest = manifest_module.build_replay_execution_manifest_v2(
        authenticated_source=source,
        attempt_id=_ATTEMPT_ID,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    manifest_path = Path(evidence._canonical_replay_manifest_path_v2(manifest))
    manifest_path.parent.mkdir(mode=0o700, parents=True)
    manifest_path, manifest_sha256 = manifest_module.publish_replay_execution_manifest_v2(
        output=manifest_path,
        document=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    assert static_inputs.attempt_id == manifest["attempt_id"]
    assert manifest_sha256 == _digest(manifest_path.read_bytes())

    accepted_path = Path(manifest["scheduler_submission"]["accepted_id_record"]["path"])
    accepted_sha256 = _seal_e2e_bytes(
        accepted_path,
        f"{_JOB_ID}\n".encode("ascii"),
    )
    slurm_root = Path(manifest["execution_environment"]["attempt"]["operational"]["slurm"])
    slurm_root.mkdir(mode=0o700, parents=True)
    comment = (
        f"nemo-rl-strict-captured-replay-v2:{manifest['attempt_id']}:"
        f"{manifest['scheduler_submission']['nonce']}:{manifest_sha256}"
    )
    pre_release_query = _publish_e2e_scheduler_query(
        manifest=manifest,
        source=source,
        raw_path=accepted_path.parent / "PRE_RELEASE.scontrol.raw",
        phase="PRE_RELEASE",
        comment=comment,
    )
    submission = evidence.build_captured_replay_submission_receipt_v2(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        replay_execution_manifest_path=str(manifest_path),
        replay_execution_manifest_sha256=manifest_sha256,
        scheduler_client_environment=pair_receipt["scheduler_tools"]["client_environment"],
        scheduler_tools={name: pair_receipt["scheduler_tools"][name] for name in ("sbatch", "scancel", "scontrol")},
        sbatch_argv=_e2e_sbatch_argv(
            manifest,
            authenticated_pair=pair,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            comment=comment,
        ),
        parsable_stdout=f"{_JOB_ID}\n",
        accepted_id_record={
            "path": str(accepted_path),
            "sha256": accepted_sha256,
            "parsed_candidate_job_id": _JOB_ID,
            "format": "ascii-positive-decimal-lf",
            "mode": "0400",
        },
        pre_release_scheduler_query=pre_release_query,
        submitted_at_unix_ns=1_788_350_000_000_000_000,
    )
    submission_path, submission_sha256 = evidence.publish_captured_replay_submission_receipt_v2(
        output=manifest["scheduler_submission"]["receipt"]["path"],
        document=submission,
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )

    receipt_paths = evidence._final_receipt_paths_v2(
        manifest,
        authenticated_job_id=_JOB_ID,
    )
    job_root = Path(receipt_paths["pre"]).parents[1]
    pre_query = _publish_e2e_scheduler_query(
        manifest=manifest,
        source=source,
        raw_path=job_root / "queries" / "PRE.scontrol.raw",
        phase="PRE",
        comment=comment,
    )
    (job_root / "receipts").mkdir(mode=0o700)
    job = {
        "account": pair["campaign"]["slurm"]["account"],
        "name": manifest["scheduler_submission"]["identity"]["job_name"],
        "num_nodes": pair["campaign"]["nodes"],
        "partition": pair["campaign"]["slurm"]["partition"],
        "qos": pair["campaign"]["slurm"]["qos"],
        "gpus_per_node": 4,
        "restart_count": 0,
    }
    pre = evidence.build_captured_replay_pre_receipt_v2(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        submission_receipt=submission,
        authenticated_job_id=_JOB_ID,
        job=job,
        pre_scheduler_query=pre_query,
    )
    pre_path, pre_sha256 = evidence.publish_captured_replay_pre_receipt_v2(
        output=receipt_paths["pre"],
        document=pre,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    assert submission_path == Path(manifest["scheduler_submission"]["receipt"]["path"])
    assert pre_path == Path(receipt_paths["pre"])
    assert pre_sha256 == _digest(pre_path.read_bytes())

    from nemo_rl.environments import strict_gym_child_runtime_v2 as child_runtime
    from nemo_rl.utils.strict_model_transport_replay_v3 import (
        ReplayVerifierMaterial,
        StrictModelTransportReplaySourceV3,
        publish_strict_model_transport_replay_consumption_v3,
    )
    from tests.unit.environments.test_strict_gym_child_runtime_v2 import (
        _format_finalizer_fixture,
    )
    from tests.unit.utils.test_strict_captured_replay_evidence import (
        _derivation,
        _format_fixture_row,
        _generation,
        _ledger_rows,
    )

    result_root = Path(manifest["artifacts"]["outputs"]["directory"]["path"])
    result_root.mkdir(mode=0o700, parents=True)
    replay_bindings = {
        "pair_manifest_sha256": manifest["pair"]["manifest"]["sha256"],
        "submission_receipt_sha256": submission_sha256,
        "job_id": _JOB_ID,
        "run_id": evidence.replay_run_id(
            environment=_ENVIRONMENT,
            pair_id=manifest["pair_id"],
            attempt_id=manifest["attempt_id"],
        ),
        "fixture_sha256": manifest["artifacts"]["fixture"]["sha256"],
        "verifier_source_sha256": manifest["replay_contract"]["gym_scorer"]["resources"]["verifier_source"]["sha256"],
        "config_sha256": manifest["replay_contract"]["selected_config"]["sha256"],
        "snapshot_manifest_sha256": manifest["replay_contract"]["source_snapshot"]["ref"]["manifest_sha256"],
    }
    replay_transcript = evidence.build_transcript_bundle(
        pair_id=manifest["pair_id"],
        environment=_ENVIRONMENT,
        arm="on",
        mode="captured_replay",
        attempt_id=manifest["attempt_id"],
        generation=_generation(),
        bindings=replay_bindings,
        fixture_row=_format_fixture_row(_ENVIRONMENT),
        model_transport_bundle=manifest["source_capture"]["step1_evidence"]["model_transport"]["bundle"],
        verifier_request_derivation=_derivation(),
        entry_inputs=source_entries,
    )
    transcript_path, transcript_sha256 = evidence.publish_evidence_document(
        output=manifest["artifacts"]["outputs"]["transcript_bundle"]["path"],
        document=replay_transcript,
        trailing_lf=False,
    )
    transcript_ref = {
        "path": str(transcript_path),
        "schema": evidence.TRANSCRIPT_BUNDLE_SCHEMA,
        "sha256": transcript_sha256,
    }

    child_root = result_root / "strict_gym_child_runtime"
    child_root.mkdir(mode=0o700)
    bootstrap_program = manifest["replay_contract"]["program"]["gym_child_bootstrap"]
    bootstrap_path = Path(manifest["replay_contract"]["source_snapshot"]["ref"]["path"]) / bootstrap_program["path"]
    real_build_spec = child_runtime._build_spec
    real_target_matrix = child_runtime._target_matrix

    def selected_target_matrix(
        ignored_environment: str,
        gym_root: Path,
        *,
        scope: str,
    ) -> list[dict[str, Any]]:
        del ignored_environment
        return real_target_matrix(_ENVIRONMENT, gym_root, scope=scope)

    monkeypatch.setattr(child_runtime, "_target_matrix", selected_target_matrix)

    def build_bound_spec(**kwargs: Any) -> dict[str, Any]:
        kwargs.update(
            environment=_ENVIRONMENT,
            pair_id=manifest["pair_id"],
            job_id=_JOB_ID,
            bootstrap_root=bootstrap_path.parent,
            bootstrap_sha256=bootstrap_program["sha256"],
        )
        return real_build_spec(**kwargs)

    monkeypatch.setattr(child_runtime, "_build_spec", build_bound_spec)
    child_session, _, child_documents, child_run_helper = _format_finalizer_fixture(monkeypatch, child_root)
    object.__setattr__(child_session, "environment", _ENVIRONMENT)
    child_index_path = child_root / "index.json"
    child_index = json.loads(child_index_path.read_bytes())
    child_index["environment"] = _ENVIRONMENT
    child_index_path.chmod(0o600)
    child_index_raw = child_runtime.canonical_ascii_json(child_index)
    child_index_path.write_bytes(child_index_raw)
    child_index_path.chmod(0o400)
    object.__setattr__(child_session, "_started_index", child_index)
    object.__setattr__(
        child_session,
        "_started_index_sha256",
        _digest(child_index_raw),
    )
    expected_calls: list[dict[str, Any]] = []
    call_refs: list[dict[str, Any]] = []
    for sequence, (entry, call_document) in enumerate(
        zip(source_entries, child_documents, strict=True),
        start=1,
    ):
        format_request = {
            name: copy.deepcopy(entry["derived_verifier_request"][name])
            for name in ("responses_create_params", "response", "verifier")
        }
        expected = child_runtime.format_verification_call_expectation(
            environment=_ENVIRONMENT,
            derived_verifier_request=format_request,
            verifier_response=entry["verifier_response"],
        )
        expected_calls.append(expected)
        call_document.update(
            {
                "environment": _ENVIRONMENT,
                "profile_id": _PROFILE_ID,
                "method": expected["method"],
                "input": {
                    name: expected[name]
                    for name in (
                        "request_sha256",
                        "verifier_sha256",
                        "response_text_sha256",
                    )
                },
                "outcome": {
                    "kind": "returned",
                    "response_sha256": expected["response_sha256"],
                    "match_details_sha256": expected["match_details_sha256"],
                    "float_result": expected["float_result"],
                },
            }
        )
        call_path = child_root / f"format-verification-call-{sequence:08d}.json"
        call_path.chmod(0o600)
        call_raw = child_runtime.canonical_ascii_json(call_document)
        call_path.write_bytes(call_raw)
        call_path.chmod(0o400)
        call_refs.append(
            {
                "sequence": sequence,
                "path": str(call_path),
                "sha256": _digest(call_raw),
                "schema": child_runtime.STRICT_GYM_FORMAT_CALL_SCHEMA,
            }
        )
    closed_path = child_root / "format-verification-closed.json"
    closed_document = json.loads(closed_path.read_bytes())
    closed_document["environment"] = _ENVIRONMENT
    closed_document["profile_id"] = _PROFILE_ID
    closed_document["calls"] = call_refs
    closed_path.chmod(0o600)
    closed_path.write_bytes(child_runtime.canonical_ascii_json(closed_document))
    closed_path.chmod(0o400)
    scorer_index, scorer_index_sha256 = child_session.finalize_format_verification_calls(
        expected_calls,
        run_helper=child_run_helper,
    )
    scorer_index_path = child_root / "format-verification-call-index.json"
    scorer_index_ref = {
        "path": str(scorer_index_path),
        "schema": child_runtime.STRICT_GYM_FORMAT_CALL_INDEX_SCHEMA,
        "sha256": scorer_index_sha256,
    }

    driver_process = {
        "boot_id_sha256": _digest(f"{_BOOT_ID}\n"),
        "pid": 50,
        "start_time_ticks": 900,
    }
    device_environment = {
        "schema": evidence.SCHEDULER_DEVICE_ENVIRONMENT_SCHEMA,
        "cuda_visible_devices": "0,1,2,3",
        "gpu_device_ordinal": "0,1,2,3",
        "nvidia_visible_devices": "all",
        "rocr_visible_devices": None,
        "ze_affinity_mask": None,
    }
    materials = tuple(
        ReplayVerifierMaterial(
            rollout_index=index,
            generation_seed=entry["generation_seed"],
            model_response=copy.deepcopy(entry["model_response"]),
            agent_run_request=copy.deepcopy(entry["agent_run_request"]),
            derived_verifier_request=copy.deepcopy(entry["derived_verifier_request"]),
            source_entry_sha256=entry["model_transport_entry_sha256"],
            request_body_sha256=entry["model_transport_request_body_sha256"],
            response_body_sha256=entry["model_transport_response_body_sha256"],
        )
        for index, entry in enumerate(source_entries)
    )
    source_step1 = manifest["source_capture"]["step1_evidence"]
    source_transport = source_step1["model_transport"]
    transport_source = StrictModelTransportReplaySourceV3(
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        pair_id=manifest["pair_id"],
        attempt_id=manifest["attempt_id"],
        replay_attempt_root=str(result_root),
        source_job_id=manifest["source_capture"]["authenticated_job"]["job_id"],
        pair_manifest_sha256=manifest["pair"]["manifest"]["sha256"],
        source_submission_receipt_sha256=manifest["pair"]["submission_receipt"]["sha256"],
        source_refs={
            "main_ledger": source_step1["main_ledger"],
            "transcript_bundle": source_step1["transcript_bundle"],
            "transport_bundle": source_transport["bundle"],
            "transport_manifest": source_transport["manifest"],
            "raw_log": source_transport["raw_log"],
            "ordered_entries_sha256": source_transport["ordered_entries_sha256"],
        },
        materials=materials,
        transcript_document=source.transcript_bundle,
        main_ledger_document=source.main_ledger,
    )
    for index, entry in enumerate(source_entries):
        transport_source.consume(
            rollout_index=index,
            generation_seed=entry["generation_seed"],
        )
        transport_source.record_fresh_verifier_result(
            rollout_index=index,
            verifier_response=entry["verifier_response"],
        )
    consumption = transport_source.finalize(
        replay_execution_manifest_sha256=manifest_sha256,
        authenticated_job_id=_JOB_ID,
        process=driver_process,
        scheduler_device_environment=device_environment,
        scorer_call_index_ref=scorer_index_ref,
    )
    consumption_path, consumption_sha256 = publish_strict_model_transport_replay_consumption_v3(
        output=manifest["artifacts"]["outputs"]["transport_consumption"]["path"],
        document=consumption,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )

    ledger_rows = _ledger_rows(replay_transcript)
    for row in ledger_rows:
        row["advantages"] = [0.0] * row["input_length"]
    replay_ledger = evidence.build_captured_replay_step1_ledger(
        pair_id=manifest["pair_id"],
        environment=_ENVIRONMENT,
        attempt_id=manifest["attempt_id"],
        source_main_ledger_sha256=source_step1["main_ledger"]["sha256"],
        source_transcript_bundle=source_step1["transcript_bundle"],
        source_transcript_document=source.transcript_bundle,
        generation=_generation(),
        bindings={
            **replay_bindings,
            "restart_count": 0,
            "pair_campaign_sha256": manifest["pair"]["pair_campaign_sha256"],
            "pair_campaign_reward_and_advantage_sha256": manifest["pair"]["pair_campaign_reward_and_advantage_sha256"],
            "process": driver_process,
        },
        transcript_bundle=transcript_ref,
        transcript_document=replay_transcript,
        row_inputs=ledger_rows,
    )
    ledger_path, ledger_sha256 = evidence.publish_evidence_document(
        output=manifest["artifacts"]["outputs"]["replay_ledger"]["path"],
        document=replay_ledger,
        trailing_lf=False,
    )
    output_refs = {
        "scorer_call_index": scorer_index_ref,
        "transport_consumption": {
            "path": str(consumption_path),
            "schema": evidence.REPLAY_TRANSPORT_CONSUMPTION_V3_SCHEMA,
            "sha256": consumption_sha256,
        },
        "transcript_bundle": transcript_ref,
        "replay_ledger": {
            "path": str(ledger_path),
            "schema": evidence.CAPTURED_REPLAY_STEP1_LEDGER_SCHEMA,
            "sha256": ledger_sha256,
        },
    }
    assert scorer_index["quiescence"]["original_process_reaped"] is True

    post_query = _publish_e2e_scheduler_query(
        manifest=manifest,
        source=source,
        raw_path=job_root / "queries" / "POST.scontrol.raw",
        phase="POST",
        comment=comment,
    )
    gpu_raw = "NVIDIA GB200, 580.126.20"
    gpu_rows = [
        {
            "index": index,
            "raw": gpu_raw,
            "gpu_model": "NVIDIA GB200",
            "driver_version": "580.126.20",
        }
        for index in range(4)
    ]
    hardware = {
        "schema": evidence.HARDWARE_OBSERVATION_SCHEMA,
        "gpu_model": "NVIDIA GB200",
        "driver_version": "580.126.20",
        "gpu_row_count": 4,
        "ordered_rows": gpu_rows,
        "raw_output_sha256": _digest(((gpu_raw + "\n") * 4).encode("ascii")),
        "ordered_rows_sha256": evidence.domain_sha256(
            evidence.HARDWARE_ORDERED_ROWS_HASH_LABEL,
            gpu_rows,
        ),
        "nvidia_smi": copy.deepcopy(manifest["runtime_tools"]["document"]["host"]["nvidia_smi"]),
    }
    exit_receipt = evidence.build_captured_replay_exit_receipt_v2(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        submission_receipt=submission,
        pre_receipt=pre,
        post_scheduler_query=post_query,
        driver_exit_code=0,
        hardware=hardware,
        scheduler_device_environment=device_environment,
        driver_scheduler_device_environment=device_environment,
        driver_process=driver_process,
        outputs=output_refs,
    )
    exit_path, exit_sha256 = evidence.publish_captured_replay_exit_receipt_v2(
        output=receipt_paths["exit"],
        document=exit_receipt,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    evidence_index = evidence.build_captured_replay_evidence_index_v2(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
    )
    index_path, index_sha256 = evidence.publish_captured_replay_evidence_index_v2(
        output=manifest["artifacts"]["outputs"]["evidence_index"]["path"],
        document=evidence_index,
        replay_execution_manifest=manifest,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )

    generated_gym_root = result_root / "strict_gym_child_runtime-format-gym"
    (generated_gym_root / "resources_servers" / "format_verification").rmdir()
    (generated_gym_root / "resources_servers").rmdir()
    generated_gym_root.rmdir()
    anchors = {
        "evidence-index.json": index_sha256,
        "model-transport-replay-consumption.json": consumption_sha256,
        "replay-ledger.json": ledger_sha256,
        "strict_gym_child_runtime/format-verification-call-index.json": (scorer_index_sha256),
        "transcript-bundle.json": transcript_sha256,
    }
    inventory_path, inventory_sha256, verified_result = seal_module.publish_sealed_result_v2_with_authority(
        result_root=str(result_root),
        anchored_sha256=anchors,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    final_document = evidence.build_captured_replay_result_final_receipt_v2(
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
        evidence_index=evidence_index,
        verified_result=verified_result,
    )
    final_path, final_sha256 = evidence.publish_captured_replay_result_final_receipt_v2(
        output=receipt_paths["final"],
        document=final_document,
        replay_execution_manifest=manifest,
        authenticated_source=source,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
        submission_receipt=submission,
        pre_receipt=pre,
        exit_receipt=exit_receipt,
        evidence_index=evidence_index,
        verified_result=verified_result,
    )
    capability = evidence.load_authenticated_captured_replay_result_v2(
        authenticated_source=source,
        replay_execution_manifest_path=str(manifest_path),
        replay_execution_manifest_sha256=manifest_sha256,
        submission_receipt_sha256=submission_sha256,
        candidate_job_id=_JOB_ID,
        result_final_receipt_path=str(final_path),
        result_final_receipt_sha256=final_sha256,
        expected_environment=_ENVIRONMENT,
        expected_profile_id=_PROFILE_ID,
    )
    snapshot = evidence.snapshot_authenticated_captured_replay_result_v2(capability)
    assert Path(inventory_path) == (result_root / seal_module.RESULT_INVENTORY_V2_FILENAME)
    assert exit_path == Path(receipt_paths["exit"])
    assert index_path == result_root / "evidence-index.json"
    assert snapshot["result_final_receipt"]["sha256"] == final_sha256
    assert snapshot["result_inventory"]["sha256"] == inventory_sha256
    assert snapshot["candidate_job_id"] == snapshot["authenticated_job_id"]
    assert len(snapshot["samples"]) == 4
    assert [sample["raw_environment_reward"] for sample in snapshot["samples"]] == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
