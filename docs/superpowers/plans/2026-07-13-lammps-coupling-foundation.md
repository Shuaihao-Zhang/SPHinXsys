# Portable LAMMPS Coupling Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide a cross-platform optional LAMMPS module, integer DEM-aligned SPH coupling steps with clock verification, rotational proxy kinematics, torque diagnostics, and maintainable example integration.

**Architecture:** CMake discovers an upstream LAMMPS package first and falls back to an explicitly configured imported target. A header-based optional module owns LAMMPS lifecycle, external-force buffering, step planning, and clock verification; examples retain geometry and fluid physics while consuming the shared runtime.

**Tech Stack:** CMake 3.16+, C++17, SPHinXsys, LAMMPS C library API, Visual Studio 2022 x64, CTest.

## Global Constraints

- Do not modify `G:/SPH_DEM/lammps_source`.
- Keep `SPHINXSYS_WITH_LAMMPS` optional and `OFF` by default.
- Preserve the current Visual Studio 2022 x64 Release build and existing `sph_build` settings.
- Support package-based Linux/HPC discovery without hard-coded platform suffixes.
- Keep SPHinXsys as the coupling driver and fluid-physics owner.
- Keep force filtering case-local; retain the 2D underwater cap as a documented diagnostic safeguard.
- Compute and output torque, but do not feed torque back to LAMMPS in this milestone.
- Keep MPI coupling out of scope.

---

### Task 1: Portable Dependency Target And Optional Module

**Files:**
- Create: `cmake/SPHinXsysLAMMPS.cmake`
- Create: `modules/lammps/CMakeLists.txt`
- Modify: `CMakeLists.txt`
- Modify: `tests/lammps_examples/CMakeLists.txt`
- Modify: `AGENTS.md`

**Interfaces:**
- Produces: `LAMMPS::lammps` from package discovery or fallback discovery.
- Produces: `SPHinXsys::sphinxsys_lammps` available whenever `SPHINXSYS_WITH_LAMMPS=ON`.
- Consumes: `SPHINXSYS_LAMMPS_ROOT`, `SPHINXSYS_LAMMPS_INCLUDE_ROOT`, `SPHINXSYS_LAMMPS_LIBRARY`, `SPHINXSYS_LAMMPS_RUNTIME`, and `SPHINXSYS_LAMMPS_SIZES` cache variables.

- [ ] Add a configure probe that fails against the current CMake because the optional module target is absent outside `tests`.
- [ ] Run the probe and verify the expected missing-target failure.
- [ ] Implement package-first discovery and a platform-neutral fallback imported target with explicit integer-size ABI definition.
- [ ] Add the optional module target and include it independently of `SPHINXSYS_BUILD_TESTS`.
- [ ] Gate 2D and 3D example directories by their corresponding build options.
- [ ] Add Windows/Linux/HPC portability requirements to `G:/SPH_DEM/AGENTS.md`.
- [ ] Reconfigure the existing build and verify current fallback discovery.

### Task 2: Common Runtime And Test-First Time Planning

**Files:**
- Create: `modules/lammps/lammps_instance.h`
- Create: `modules/lammps/lammps_coupling_clock.h`
- Create: `modules/lammps/lammps_external_force.h`
- Create: `tests/lammps_examples/common/test_lammps_common.cpp`
- Modify: `tests/lammps_examples/common/CMakeLists.txt`
- Modify: `tests/lammps_examples/common/lammps_instance.h`
- Modify: `tests/lammps_examples/common/lammps_dem_adapter_common.h`

**Interfaces:**
- Produces: `CouplingStepPlan makeCouplingStepPlan(double acoustic_limit, double nominal_dem_dt)`.
- Produces: `LammpsTimeIntegrator::advance(const CouplingStepPlan&)` and clock diagnostics.
- Produces: `ExternalForceBuffer` keyed by actual LAMMPS atom ID.
- Produces: `LammpsInstance::atomCount()` and `LammpsInstance::accumulatedTime()`.

- [ ] Write failing unit tests for `5.3e-5 -> 5 x 1e-5`, exact multiples, a sub-DEM acoustic limit, invalid inputs, and ID-keyed force lookup.
- [ ] Build and run the common test to verify failure because the APIs are absent.
- [ ] Implement the minimum planner and force buffer required by the tests.
- [ ] Add LAMMPS accumulated-time access and an integrator that checks `atime` after each run.
- [ ] Add a runtime test that opens LAMMPS, advances aligned steps, and verifies the clock.
- [ ] Run the common tests and verify they pass.

### Task 3: Centralized Example CMake And Runtime Staging

**Files:**
- Modify: `tests/lammps_examples/common/CMakeLists.txt`
- Modify: all LAMMPS-linked example `CMakeLists.txt` files under `tests/lammps_examples`
- Modify: LAMMPS-linked example source files that check `liblammps.dll`.

**Interfaces:**
- Produces: `sphinxsys_configure_lammps_example(target)` for linking, runtime staging, and runtime filename definition.
- Consumes: `$<TARGET_FILE:LAMMPS::lammps>` and `$<TARGET_FILE_NAME:LAMMPS::lammps>`.

- [ ] Add a source scan test that detects hard-coded `.dll`, `.lib`, and `Release` paths in example CMake/source files.
- [ ] Run the scan and verify it finds the existing literals.
- [ ] Implement the shared CMake helper and migrate all seven linked examples.
- [ ] Replace runtime DLL literals with the configured runtime filename.
- [ ] Re-run the scan and verify it is clean.

### Task 4: Migrate Adapters To Aligned Clocking And Explicit Identity

**Files:**
- Modify: linked example headers and source files under `tests/lammps_examples`

**Interfaces:**
- Consumes: `makeCouplingStepPlan`, `LammpsTimeIntegrator`, `LammpsInstance::atomCount`, and explicit atom-ID validation.
- Produces: requested acoustic limit, coupled acoustic step, LAMMPS accumulated time, and synchronization error diagnostics.

- [ ] Add clock assertions to the link-only test and verify the current adapter lacks the required diagnostics.
- [ ] Replace per-case full-step/remainder logic with the common plan and integrator.
- [ ] Make SPH pressure, density, and physical-time increments use `plan.coupled_dt`.
- [ ] Query atom count before gathers and validate the fixed ID contract explicitly.
- [ ] Update CSV and console diagnostics to report coupled time and synchronization error.
- [ ] Run link-only, driven-sphere, one-way, and two-way clock tests.

### Task 5: Rotational Proxy Velocity And Torque Diagnostics

**Files:**
- Modify: 2D/3D moving-boundary example headers.
- Modify: one-way, two-way, and underwater force CSV/VTP writers.
- Modify: `tests/lammps_examples/common/lammps_io.h` only if a generic torque field is needed.

**Interfaces:**
- Produces: `v_proxy = Vc + omega cross r` for circular and spherical proxies.
- Produces: per-particle hydrodynamic torque reductions and diagnostic fields.
- Does not produce: LAMMPS torque feedback.

- [ ] Add focused kinematics tests with nonzero angular velocity and known relative positions.
- [ ] Verify the tests fail because proxy velocities currently omit rotation.
- [ ] Implement 2D and 3D rotational surface velocity.
- [ ] Add pressure-plus-viscous torque reductions around each DEM center.
- [ ] Add raw torque columns and VTP fields with correct 2D/3D units.
- [ ] Run the kinematics and output tests.

### Task 6: Underwater Viscous Update And Scalable Diagnostics

**Files:**
- Modify: `tests/lammps_examples/test_2d_lammps_underwater_granular_collapse/test_2d_lammps_underwater_granular_collapse.cpp`
- Modify: `tests/lammps_examples/test_2d_lammps_underwater_granular_collapse/test_2d_lammps_underwater_granular_collapse.h`
- Modify: its `CMakeLists.txt`

**Interfaces:**
- Restores: evolving `ViscousForceWithWall` and `ViscousForceFromFluid` calls.
- Produces: configurable CSV sampling interval and cap fraction summary.

- [ ] Add a regression scan/test that fails while viscous updates are commented out.
- [ ] Restore both viscous update calls in the advection loop.
- [ ] Sample per-particle CSV on a configurable iteration stride while retaining in-memory maxima each step.
- [ ] Report cap fraction and label the case as capped/non-conservative when cap activation is nonzero.
- [ ] Mark the full underwater and relaxation CTests with the `long` label.
- [ ] Build the case and run its short/common checks without requiring the full one-second simulation.

### Task 7: Final Verification

**Files:**
- Verify all modified files; do not modify `lammps_source`.

**Interfaces:**
- Verifies the complete feature set and reports environmental limitations.

- [ ] Run `git diff --check` and source portability scans.
- [ ] Reconfigure `G:/SPH_DEM/sph_build` using its existing generator and toolchain.
- [ ] Build all LAMMPS-linked targets in Release.
- [ ] Run fast CTests labelled `lammps` while excluding `long`.
- [ ] Run selected physical integration tests within a bounded timeout.
- [ ] Compare `lammps_source` against the release ZIP or verify it remains byte-identical by the existing audit method.
- [ ] Report changed files, commands, test results, clock error, torque-output status, and Linux/HPC verification still pending.
