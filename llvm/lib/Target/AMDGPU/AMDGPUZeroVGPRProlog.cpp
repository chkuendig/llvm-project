//===-- AMDGPUZeroVGPRProlog.cpp - Zero VGPRs at kernel entry -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ZLUDA addition (2026-05-11): zero all kernel-allocated VGPRs (except
// ABI-passed live-ins) at the start of the kernel entry block to eliminate
// stale high-32-bit residue from prior kernels on the same wave.
//
// Targets the inter-kernel VGPR-residue hypothesis for IOFX_actorRangeKernel
// and related PhysX kernels' wild-pointer faults. The captured fault VAs
// (per project_phase3e_iofx_actorrange_dominant) consistently show a
// page-aligned address ~10 GB from any live allocation, with the high-32
// bits clustered around HSA-mapped values — the canonical signature of a
// pointer reconstructed from one VGPR holding stale-high-32 and another
// holding a freshly-computed low-32 offset.
//
// This pass attacks that source generically: every VGPR the kernel will
// use is zero-initialized at entry, so any read-before-write of a 32-bit
// half during pointer construction sees 0, producing either a low-VA fault
// (catchable) or a valid heap address (no fault) instead of the
// high-7f-stale pattern.
//
// Per project rule, fix code must NOT branch on kernel name. This pass
// applies generically to every GPU entry function via calling-convention
// check (`isEntryFunctionCC`).
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "SIInstrInfo.h"
#include "SIRegisterInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "amdgpu-zero-vgpr-prolog"

static cl::opt<bool> EnableVGPRZeroProlog(
    "amdgpu-zero-vgpr-prolog",
    cl::desc("ZLUDA: zero all kernel-allocated VGPRs (except ABI live-ins) "
             "at entry to eliminate cross-kernel VGPR residue"),
    cl::init(true), cl::Hidden);

namespace {

class AMDGPUZeroVGPRProlog : public MachineFunctionPass {
public:
  static char ID;

  AMDGPUZeroVGPRProlog() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override {
    return "AMDGPU Zero VGPR Prolog";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

} // namespace

// opt registers legacy pass names as command-line choices. Keep this name
// distinct from the boolean option above so target initialization cannot
// register "amdgpu-zero-vgpr-prolog" twice.
INITIALIZE_PASS(AMDGPUZeroVGPRProlog, "amdgpu-insert-zero-vgpr-prolog",
                "AMDGPU Zero VGPR Prolog", false, false)

char AMDGPUZeroVGPRProlog::ID = 0;

FunctionPass *llvm::createAMDGPUZeroVGPRPrologPass() {
  return new AMDGPUZeroVGPRProlog();
}

bool AMDGPUZeroVGPRProlog::runOnMachineFunction(MachineFunction &MF) {
  if (!EnableVGPRZeroProlog)
    return false;

  // Only run on actual GPU entry functions. Generic CC check, no kernel
  // name branching.
  if (!AMDGPU::isEntryFunctionCC(MF.getFunction().getCallingConv()))
    return false;

  const GCNSubtarget &ST = MF.getSubtarget<GCNSubtarget>();
  const SIInstrInfo *TII = ST.getInstrInfo();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  // Collect ABI live-ins (workitem.id.x/y/z, dispatch ptr, etc.) — these
  // contain caller-provided values and must NOT be overwritten.
  DenseSet<Register> LiveIns;
  for (const auto &LI : MRI.liveins())
    LiveIns.insert(LI.first);

  // Collect physical VGPRs that the kernel actually uses anywhere in its
  // body (operand of any instruction in any basic block).
  DenseSet<Register> UsedVGPRs;
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        Register R = MO.getReg();
        if (!R.isPhysical())
          continue;
        if (!AMDGPU::VGPR_32RegClass.contains(R))
          continue;
        UsedVGPRs.insert(R);
      }
    }
  }

  if (UsedVGPRs.empty())
    return false;

  MachineBasicBlock &Entry = MF.front();
  MachineBasicBlock::iterator InsertPt = Entry.begin();
  DebugLoc DL;
  unsigned Zeroed = 0;

  for (Register R : UsedVGPRs) {
    if (LiveIns.contains(R))
      continue;
    // V_MOV_B32_e32 vN, 0 — per-lane VGPR zero. Each lane writes 0 to
    // its own copy of vN, eliminating any stale residue from a prior
    // kernel's wave that ran on the same SIMD.
    BuildMI(Entry, InsertPt, DL, TII->get(AMDGPU::V_MOV_B32_e32), R)
        .addImm(0);
    ++Zeroed;
  }

  LLVM_DEBUG(dbgs() << "AMDGPUZeroVGPRProlog: zeroed " << Zeroed
                    << " VGPRs in " << MF.getName() << "\n");
  return Zeroed > 0;
}
