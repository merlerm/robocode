# Scene observations

Use `obstruction2d_scene` or `dynobstruction2d_scene` for experiments where
policies observe the whole scene. These configurations enable
`include_static_objects` on `VariableObjectCountEnv`.

Reset, step and get_state return the existing `ObjectCentricState`, including
boundary walls and supporting surfaces. Geometry uses the same typed features
as the rest of that domain. Read each object's pose, dimensions and `static`
flag; enumerate objects instead of assuming a fixed set of wall names. The
robot and goal retain their existing types and action semantics. Kinematic
and dynamic Obstruction2D have separate schemas.

An environment revision may change object names, geometry and the number of
scene objects. Candidate validation still requires the same action space,
object types/features and configured count schedule. Scene validation requires
`robot`, `target_block` and `target_surface` with unchanged types/features, and
checks the actual count-defining object count. Other scene object IDs are not
part of that schema contract. Non-scene configurations retain their previous
object-identity checks. Count-defining objects retain their prefix convention;
do not use that prefix for extra scene geometry.

The current dynamic simulator identifies walls by names containing `wall`.
New fixed walls must follow that convention; the kinematic domain has no such
wall-name requirement. Observation compatibility does not remove the simulator's
construction rules.

State restoration is supported within the same environment revision. It checks
revision-owned constant geometry, removes it before restoring the simulator's
variable state, then reconstructs the full observation. Changed or missing
constant geometry raises an error. Do not restore states across revisions.
Rendering and source-access clients use the same configured environment.
Source code is not part of an observation, so strict blackbox policies see
geometry without simulator implementation access.

Geometry sampled anew on reset belongs in the simulator's current state,
with its `static` flag set for immovable objects. Reserve initial constant
state for geometry that stays fixed within that environment revision.

This describes the public interface only. Independent evaluation layouts,
generators and scores must remain outside both development agents' mounts.
