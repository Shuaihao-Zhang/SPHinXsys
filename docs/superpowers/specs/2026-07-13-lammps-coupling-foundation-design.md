# Portable LAMMPS Coupling Foundation Design

## Goal

Turn the current SPHinXsys-LAMMPS milestone examples into a portable and maintainable coupling foundation without moving fluid physics into LAMMPS, modifying `lammps_source`, or implementing strong coupling.

The result must continue to run in the current Visual Studio 2022 x64 Release build and provide a supported CMake path for Linux and future HPC builds.

## Scope

This work includes:

- portable LAMMPS discovery and imported-target handling;
- a reusable optional SPHinXsys LAMMPS module outside the test hierarchy;
- integer DEM substep alignment with LAMMPS `atime` verification;
- dynamic atom-count and explicit atom-ID validation;
- rotational velocity on SPHinXsys DEM proxy particles;
- hydrodynamic torque accumulation and diagnostic output;
- restoration of evolving viscous-force updates in the underwater granular case;
- reduced diagnostic-output volume for multi-particle cases;
- focused unit and integration tests;
- Windows, Linux, and HPC portability rules in the project agent guide.

This work does not include:

- hydrodynamic torque feedback to LAMMPS;
- MPI coupling;
- strong or iterative coupling;
- removal of the experimental two-dimensional granular force cap;
- changes to `lammps_source`;
- sediment scour physics or empirical drag laws.

## Ownership And Dependency Boundary

SPHinXsys remains the top-level driver. It owns fluid state, body relations, hydrodynamic force and torque reduction, coupling scheduling, and output. LAMMPS owns DEM particle state, gravity, contact, friction, rolling behavior, and DEM integration.

The dependency remains optional through `SPHINXSYS_WITH_LAMMPS`, defaulting to `OFF`. Enabling the option creates an in-tree target named `SPHinXsys::sphinxsys_lammps` that is available even when examples and tests are disabled.

Low-level lifecycle and time-integration code belongs in `modules/lammps`. Case geometry, SPH force dynamics, force filtering, validation criteria, and visualization remain in `tests/lammps_examples`.

## Portable CMake Discovery

LAMMPS discovery uses two ordered paths.

1. Prefer `find_package(LAMMPS CONFIG QUIET)` and consume the upstream `LAMMPS::lammps` target. This is the supported path for installed Linux and HPC builds because it preserves LAMMPS compile definitions and transitive dependencies.
2. If no package target is available, construct a fallback imported target from cache variables and platform-independent `find_path`, `find_library`, and `find_file` searches. The fallback exposes an include root containing `lammps/library.h` and explicitly propagates `LAMMPS_SMALLBIG` or `LAMMPS_BIGBIG` according to `SPHINXSYS_LAMMPS_SIZES`.

The fallback supports the current sibling `lammps_build` directory but does not hard-code a drive letter. Runtime staging uses `$<TARGET_FILE:LAMMPS::lammps>` instead of `.dll`, `.lib`, `Release`, or Linux-specific suffixes. Example source code receives the runtime filename from CMake and does not contain `liblammps.dll` literals.

LAMMPS example registration respects `SPHINXSYS_2D` and `SPHINXSYS_3D`. The SPH-only fixed-sphere hydro-force baseline remains buildable without LAMMPS discovery.

## Common Runtime Module

`modules/lammps` provides a small header-based target with these responsibilities:

- `LammpsInstance`: RAII ownership, command execution, error translation, version, atom count, and accumulated simulation time access;
- `CouplingStepPlan`: a pure value describing coupled duration, DEM timestep, number of DEM steps, and whether a shortened safety step is used;
- `makeCouplingStepPlan`: converts an SPH acoustic-step upper bound into a DEM-aligned step;
- `LammpsTimeIntegrator`: executes a plan, tracks expected LAMMPS time, and checks `atime` after every advance;
- `ExternalForceBuffer`: stores forces by actual LAMMPS atom ID and supplies the `fix external` callback without assuming local atom ordering.

The common module does not create particles, prescribe contact models, know about water, filter hydrodynamic forces, or write case-specific output.

## Time Synchronization

For nominal DEM timestep `dt_dem` and SPH acoustic upper bound `dt_candidate`:

```text
if dt_candidate >= dt_dem:
    n_dem = floor((dt_candidate + tolerance) / dt_dem)
    dt_coupled = n_dem * dt_dem
    run n_dem steps of dt_dem
else:
    n_dem = 1
    dt_coupled = dt_candidate
    run one shortened DEM step of dt_candidate
```

The normal path never uses a remainder step. For example, `5.3e-5 s` becomes five `1.0e-5 s` DEM steps and a `5.0e-5 s` SPH acoustic step. Rounding down preserves the SPH stability limit.

The shortened-step branch is only permitted when the SPH stability limit or a final phase boundary is below the nominal DEM timestep. The SPH timestep is never enlarged to force alignment.

SPHinXsys advances pressure, density, and `physical_time` by `dt_coupled`. LAMMPS advances by the same plan. After every LAMMPS run, the integrator reads `lammps_extract_global(handle, "atime")`. The caller supplies the SPHinXsys physical time at the start of the step. The integrator establishes a constant driver-to-LAMMPS offset, which permits an intentional fluid-only relaxation interval, and rejects any later discontinuity in the SPHinXsys driver clock. It requires:

```text
abs((lammps_atime + driver_offset) - sph_physical_time) <=
    absolute_tolerance + relative_tolerance * max(1, sph_physical_time)
```

Diagnostics report requested acoustic limit, coupled acoustic step, DEM timestep, DEM steps, shortened-step usage, LAMMPS accumulated time, and synchronization error. Physical time, rather than raw DEM step count, is the authoritative clock.

## State, Identity, And Dimensional Conventions

Adapters query `lammps_get_natoms` and the serial `nlocal` value before reading state. They use `lammps_extract_atom` with the ABI-correct `tagint` type and sort local property views into ID order. This avoids the `lammps_gather_atoms` restriction under `LAMMPS_BIGBIG`. Each case verifies the expected particle count. Atom IDs are represented explicitly and validated rather than synthesized silently from array positions.

The first implementation remains serial and requires consecutive IDs plus locally owned atom views, but it fails with a clear message when that contract is violated. MPI-scale state exchange is a later milestone.

Two-dimensional SPH quantities are interpreted per unit out-of-plane depth. A named `out_of_plane_depth = 1 m` conversion is used when mapping 2D mass, force, and torque to LAMMPS's 3D numerical units. CSV and console labels consistently distinguish `N/m` and `N m/m` from 3D `N` and `N m`.

## Rotation And Hydrodynamic Torque

For each proxy particle with relative position `r`:

```text
x_proxy = Xc + r
v_proxy = Vc + omega cross r
```

For spherical and circular particles, rotating the point cloud is not required to preserve geometry, but the tangential surface velocity is required for viscous interaction.

Hydrodynamic torque is reduced on the SPHinXsys side:

```text
T_hydro = sum((x_i - Xc) cross (F_pressure_i + F_viscous_i))
```

The torque is written to CSV and VTP diagnostics. It is not sent to LAMMPS in this change. A later isolated validation will implement and test a per-atom torque channel, likely using the GRANULAR `fix addtorque/atom` capability.

## Force Filtering And Two-Dimensional Cap

Force relaxation and the safety cap remain case-level policies. They are not part of the common adapter.

The two-dimensional underwater granular collapse keeps its cap because closed or nearly closed 2D pores can produce large pressure transients that do not represent connected three-dimensional pore flow. The case reports cap count, cap fraction, maximum raw force, and maximum applied force. Documentation states that a capped run is numerically bounded but not a momentum-conservation validation.

Three-dimensional and single-particle validation cases continue to expect no cap activation. Exact action-reaction conservation with filtered interface forces remains a separate coupling-validation milestone.

## Output And Testing

Multi-particle CSV output is sampled at a configurable interval instead of every acoustic step. Summary statistics remain available at every step in memory. VTP/PVD output continues on physical-time intervals.

Tests are divided into:

- fast unit tests for aligned-step planning, short-step behavior, clock tolerance, force-ID lookup, and dimensional conversion;
- fast library tests for LAMMPS open/close, callback registration, atom-count validation, and `atime` synchronization;
- existing physical integration tests, labelled separately from fast tests;
- long underwater granular and relaxation cases, labelled `long` so they are not required for every edit-build cycle.

The current Windows build must pass. Linux portability is verified through platform-neutral CMake logic and documented configure commands; an actual Linux/HPC run is reported as pending until that environment is available.

## Success Criteria

- `lammps_source` remains unchanged.
- The current Visual Studio 2022 x64 Release configuration still builds all LAMMPS examples.
- No LAMMPS example CMake file contains a hard-coded DLL, import-library suffix, or Release directory.
- An installed LAMMPS package can provide the upstream imported target without reconstructing its ABI properties.
- Normal acoustic coupling steps contain only integer nominal DEM steps.
- LAMMPS `atime` and SPHinXsys coupling time remain within the declared tolerance.
- Proxy velocities include rotational surface velocity.
- Hydrodynamic torque is produced in diagnostics but is not fed back to LAMMPS.
- The underwater case updates both pressure and viscous fluid-solid forces during dynamics.
- Source-tree tests remain optional, while the low-level coupling target is available when tests are disabled.
- The project guide contains explicit Windows, Linux, and HPC portability requirements.
