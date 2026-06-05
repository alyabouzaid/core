# Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
# Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""Tests for the MQT MLIR Python bindings."""

from __future__ import annotations

import pytest

from mqt.core.mlir import compile_program, compile_qc_to_qco, make_context, qasm_to_qco

BELL_QASM = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
"""

SINGLE_QUBIT_QASM = """\
OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
t q[0];
h q[0];
"""


# ---------------------------------------------------------------------------
# make_context / register_dialects
# ---------------------------------------------------------------------------


def test_make_context_returns_context() -> None:
    from mlir.ir import Context

    ctx = make_context()
    assert isinstance(ctx, Context)


def test_make_context_can_be_used_as_context_manager() -> None:
    with make_context():
        pass


# ---------------------------------------------------------------------------
# qasm_to_qco
# ---------------------------------------------------------------------------


def test_qasm_to_qco_returns_string() -> None:
    result = qasm_to_qco(BELL_QASM)
    assert isinstance(result, str)


def test_qasm_to_qco_output_is_qco_dialect() -> None:
    result = qasm_to_qco(BELL_QASM)
    assert "module" in result
    assert "qco." in result or "func.func" in result


def test_qasm_to_qco_bell_circuit() -> None:
    result = qasm_to_qco(BELL_QASM)
    assert "OPENQASM" not in result
    assert "module" in result


def test_qasm_to_qco_single_qubit() -> None:
    result = qasm_to_qco(SINGLE_QUBIT_QASM)
    assert isinstance(result, str)
    assert "module" in result


def test_qasm_to_qco_invalid_input_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017
        qasm_to_qco("this is not valid QASM")


# ---------------------------------------------------------------------------
# compile_qc_to_qco
# ---------------------------------------------------------------------------


def test_compile_qc_to_qco_accepts_qco_mlir() -> None:
    qco_ir = qasm_to_qco(BELL_QASM)
    result = compile_qc_to_qco(qco_ir)
    assert isinstance(result, str)
    assert "module" in result


# ---------------------------------------------------------------------------
# compile_program
# ---------------------------------------------------------------------------


def test_compile_program_returns_string_by_default() -> None:
    result = compile_program(BELL_QASM)
    assert isinstance(result, str)
    assert "module" in result


def test_compile_program_single_qubit() -> None:
    result = compile_program(SINGLE_QUBIT_QASM)
    assert isinstance(result, str)
    assert "module" in result


def test_compile_program_record_intermediates_returns_dict() -> None:
    result = compile_program(BELL_QASM, record_intermediates=True)
    assert isinstance(result, dict)


def test_compile_program_record_has_all_stages() -> None:
    result = compile_program(BELL_QASM, record_intermediates=True)
    expected_keys = {
        "result",
        "after_qc_import",
        "after_initial_canon",
        "after_qco_conversion",
        "after_qco_canon",
        "after_optimization",
        "after_optimization_canon",
        "after_qc_conversion",
        "after_qc_canon",
        "after_qir_conversion",
        "after_qir_canon",
    }
    assert expected_keys == set(result.keys())


def test_compile_program_record_result_is_mlir() -> None:
    result = compile_program(BELL_QASM, record_intermediates=True)
    assert "module" in result["result"]


def test_compile_program_record_qc_import_non_empty() -> None:
    result = compile_program(BELL_QASM, record_intermediates=True)
    assert result["after_qc_import"].strip() != ""


def test_compile_program_record_stages_are_strings() -> None:
    result = compile_program(BELL_QASM, record_intermediates=True)
    for key, value in result.items():
        assert isinstance(value, str), f"stage '{key}' is not a string"


def test_compile_program_hadamard_lifting() -> None:
    result = compile_program(BELL_QASM, enable_hadamard_lifting=True)
    assert isinstance(result, str)
    assert "module" in result


def test_compile_program_disable_rotation_merging() -> None:
    result = compile_program(
        SINGLE_QUBIT_QASM, disable_merge_single_qubit_rotation_gates=True
    )
    assert isinstance(result, str)
    assert "module" in result


def test_compile_program_invalid_input_raises() -> None:
    with pytest.raises(Exception):  # noqa: B017
        compile_program("not valid qasm at all")
