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
object types/features and configured count schedule. Count-defining objects
retain their existing prefix convention. Observation values and object IDs
are not part of the schema contract.

State restoration removes revision-owned constant objects before restoring
the simulator's variable state, then reconstructs the full observation.
Rendering and source-access clients use the same configured environment.
Source code is not part of an observation, so strict blackbox policies see
geometry without simulator implementation access.

Geometry sampled anew on reset belongs in the simulator's current state,
with its `static` flag set for immovable objects. Reserve initial constant
state for geometry that stays fixed within that environment revision.

This describes the public interface only. Independent evaluation layouts,
generators and scores must remain outside both development agents' mounts.
