/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h" // NOLINT(misc-include-cleaner)
#include "mlir/CAPI/Dialects.h"
#include "mlir/Compiler/CompilerPipeline.h"
#include "mlir/Conversion/QCToQCO/QCToQCO.h" // NOLINT(misc-include-cleaner)
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h" // NOLINT(misc-include-cleaner)
#include "mlir/Support/Passes.h" // NOLINT(misc-include-cleaner)
#include "qasm3/Importer.hpp"

#include <llvm/Support/raw_ostream.h>   // NOLINT(misc-include-cleaner)
#include <mlir/IR/BuiltinOps.h>         // NOLINT(misc-include-cleaner)
#include <mlir/IR/MLIRContext.h>        // NOLINT(misc-include-cleaner)
#include <mlir/Pass/PassManager.h>      // NOLINT(misc-include-cleaner)
#include <mlir/Support/LogicalResult.h> // NOLINT(misc-include-cleaner)

#include <stdexcept> // NOLINT(misc-include-cleaner)
#include <string>    // NOLINT(misc-include-cleaner)

namespace nb = nanobind; // NOLINT(misc-unused-alias-decls)

// NOLINTNEXTLINE(misc-use-internal-linkage,readability-identifier-naming,readability-named-parameter)
NB_MODULE(_mqtCoreMlir, m) {
  mqtMlirRegisterAllPasses();

  m.doc() = "MQT Core MLIR Python bindings";

  m.def("register_dialects",
        [](MlirContext context) { mqtMlirRegisterAllDialects(context); },
        nb::arg("context"),
        "Register and load QC, QCO, QTensor, and dependent MLIR dialects.");

  m.def(
      "qasm_to_qco",
      [](const std::string& qasm) -> std::string {
        auto qc = qasm3::Importer::imports(qasm);

        mlir::MLIRContext ctx;
        MlirContext cCtx{&ctx};
        mqtMlirRegisterAllDialects(cCtx);

        auto module = mlir::translateQuantumComputationToQC(&ctx, qc);
        if (!module) {
          throw std::runtime_error("failed to translate circuit to QC MLIR");
        }

        mlir::PassManager pm(&ctx);
        populateQCCleanupPipeline(pm);
        pm.addPass(mlir::createQCToQCO());
        if (mlir::failed(pm.run(*module))) {
          throw std::runtime_error("qc-to-qco conversion failed");
        }

        std::string out;
        llvm::raw_string_ostream os(out);
        module->print(os);
        return out;
      },
      nb::arg("qasm"),
      "Run the (QASM) -> (QC dialect) -> (QCO dialect) pipeline.");

  m.def(
      "compile_program",
      [](const std::string& qasm, bool convertToQIR, bool recordIntermediates,
         bool disableMergeSingleQubitRotationGates,
         bool enableHadamardLifting) -> nb::object {
        auto qc = qasm3::Importer::imports(qasm);

        mlir::MLIRContext ctx;
        MlirContext cCtx{&ctx};
        mqtMlirRegisterAllDialects(cCtx);

        auto module = mlir::translateQuantumComputationToQC(&ctx, qc);
        if (!module) {
          throw std::runtime_error("failed to translate circuit to QC MLIR");
        }

        mlir::QuantumCompilerConfig config;
        config.convertToQIR = convertToQIR;
        config.recordIntermediates = recordIntermediates;
        config.disableMergeSingleQubitRotationGates =
            disableMergeSingleQubitRotationGates;
        config.enableHadamardLifting = enableHadamardLifting;

        mlir::CompilationRecord record;
        const mlir::QuantumCompilerPipeline pipeline(config);
        if (mlir::failed(pipeline.runPipeline(
                module.get(), recordIntermediates ? &record : nullptr))) {
          throw std::runtime_error("compilation pipeline failed");
        }

        const std::string finalIR = mlir::captureIR(module.get());

        if (!recordIntermediates) {
          return nb::str(finalIR.c_str());
        }

        nb::dict result;
        result["result"] = nb::str(finalIR.c_str());
        result["after_qc_import"] = nb::str(record.afterQCImport.c_str());
        result["after_initial_canon"] =
            nb::str(record.afterInitialCanon.c_str());
        result["after_qco_conversion"] =
            nb::str(record.afterQCOConversion.c_str());
        result["after_qco_canon"] = nb::str(record.afterQCOCanon.c_str());
        result["after_optimization"] =
            nb::str(record.afterOptimization.c_str());
        result["after_optimization_canon"] =
            nb::str(record.afterOptimizationCanon.c_str());
        result["after_qc_conversion"] =
            nb::str(record.afterQCConversion.c_str());
        result["after_qc_canon"] = nb::str(record.afterQCCanon.c_str());
        result["after_qir_conversion"] =
            nb::str(record.afterQIRConversion.c_str());
        result["after_qir_canon"] = nb::str(record.afterQIRCanon.c_str());
        return result;
      },
      nb::arg("qasm"), nb::arg("convert_to_qir") = false,
      nb::arg("record_intermediates") = false,
      nb::arg("disable_merge_single_qubit_rotation_gates") = false,
      nb::arg("enable_hadamard_lifting") = false,
      "Run the full MQT compiler pipeline on a QASM string.\n\n"
      "Returns the final IR as a string. When record_intermediates=True,\n"
      "returns a dict mapping stage names to IR snapshots plus 'result'.");
}
