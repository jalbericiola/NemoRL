# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


LEGACY_V1_PROGRAM_SHA256 = {
    "examples/nemo_gym/run_strict_captured_replay.py": (
        "02f57db8a2915a4ef2cd9748c8bfdda5734b270035fa0cc8aca74eed6d660dea"
    ),
    "nemo_rl/algorithms/strict_captured_replay_runtime.py": (
        "77ec90f44538ebf26ea3742af72d4e3369eaa90a66dd16ad12346fa49ce818ea"
    ),
    "nemo_rl/environments/_strict_gym_child_bootstrap/sitecustomize.py": (
        "2b45b2c45684b6ab77a2c3e004f8a780b15dcd0d1899f2b2a4ee41dda086025e"
    ),
    "nemo_rl/environments/strict_gym_child_runtime.py": (
        "f2e32f4d42482c388750924b01728b6cd7ced15a8dec03ef09b05d82a099e8b3"
    ),
    "nemo_rl/utils/strict_captured_replay_evidence.py": (
        "73ace52f36504ebd8b46c63cc501836235e3df34e2a5ff46699ad5a383aa80c6"
    ),
    "nemo_rl/utils/strict_captured_replay_manifest.py": (
        "4253f6db05a3b2a77e1980fd9e06aebfe4e1a0f8fe6980d468d9bab03fb7d441"
    ),
    "nemo_rl/utils/strict_captured_replay_seal.py": (
        "f2b323568c73af721a5c49fbf582d6e687ff08c49bd18d6669e356b6d049a60c"
    ),
    "nemo_rl/utils/strict_model_transport_replay.py": (
        "fa0433232e8630f8c288dc317efdb081365fc4890ab8c4c81cab8e43322e6495"
    ),
    "strict_pair_replay_job_wrapper.sh": (
        "ec682724562d60ced2153ca1ba712e00f8492f8a4ec2d3af48f930a7ff34ad11"
    ),
    "strict_pair_replay_launch.sh": (
        "e36554c3f1d269d647f444b79176a7f21812f33e8e5164486ce375f28c0ff0ad"
    ),
}


def test_legacy_v1_authenticated_program_bytes_are_exactly_preserved() -> None:
    repository = Path(__file__).parents[3]
    assert len(LEGACY_V1_PROGRAM_SHA256) == 10
    actual = {
        relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        for relative in LEGACY_V1_PROGRAM_SHA256
    }
    assert actual == LEGACY_V1_PROGRAM_SHA256


def test_legacy_manifest_and_sealer_import_never_execute_profile_registry() -> None:
    repository = Path(__file__).parents[3]
    script = r"""
import builtins
import sys

repository = sys.argv[1]
sys.path.insert(0, repository)
profile_module = "nemo_rl.utils.strict_captured_replay_profiles"
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == profile_module:
        raise RuntimeError("legacy import executed unauthenticated profile registry")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import nemo_rl.utils.strict_captured_replay_manifest
import nemo_rl.utils.strict_captured_replay_seal
assert profile_module not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(repository)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_v2_module_import_defers_profile_registry_until_authenticated_dispatch() -> (
    None
):
    repository = Path(__file__).parents[3]
    script = r"""
import builtins
import sys

repository = sys.argv[1]
sys.path.insert(0, repository)
profile_module = "nemo_rl.utils.strict_captured_replay_profiles"
original_import = builtins.__import__

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == profile_module:
        raise RuntimeError("V2 module import executed profile registry before authentication")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = guarded_import
import nemo_rl.utils.strict_captured_replay_manifest_v2
import nemo_rl.utils.strict_captured_replay_seal_v2
assert profile_module not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(repository)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
