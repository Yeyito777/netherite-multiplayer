# Boxing action/checkpoint contracts

Checkpoint consumers must call `checkpoint_action_schema(config)`. A checkpoint
without `config.action_schema` is frozen as `legacy_5hz_v1`; unknown schemas are
rejected rather than guessed.

## `legacy_5hz_v1`

- `repeat=4`, nominal control rate 5 Hz;
- head 2: yaw `{-15,0,+15}` degrees;
- head 3: pitch `{-10,0,+10}` degrees;
- actor shape: `(3,3,3,3,2,2,2)`.

## `fine_yaw_20hz_v2`

- `repeat=1`, nominal control rate 20 Hz;
- the old coarse yaw head remains `{-15,0,+15}`;
- the old pitch head is redefined as a fine yaw residual `{-5,0,+5}`;
- their Cartesian sum is the unique grid
  `{-20,-15,-10,-5,0,+5,+10,+15,+20}` degrees;
- pitch is fixed at zero because the MVP arena and target eye heights are flat;
- actor shape remains `(3,3,3,3,2,2,2)`, but semantics are not interchangeable.

The unchanged tensor shape is deliberate: it isolates the control-frequency and
yaw-resolution experiment from a network-capacity change. It does **not** permit
silent checkpoint reuse; the schema field is authoritative.

New checkpoints also record:

- `checkpoint_contract=netherite_pvp_actor_critic_v2`;
- `observation_schema=egocentric_state_24_v2`;
- `obs_basis=movement_v2`;
- `control_hz=20`;
- `repeat=1`.

## `continuous_yaw_20hz_v3`

- `repeat=1`, nominal control rate 20 Hz;
- forward, strafe, jump, sprint, and attack remain categorical with actor shape
  `(3,3,2,2,2)`;
- yaw is a tanh-squashed Gaussian bounded to `(-20,+20)` degrees per tick;
- PPO uses the transformed distribution log probability, including its scale and
  tanh-Jacobian correction;
- training samples yaw for exploration, while real deployment uses the yaw mean
  by default and may independently sample the tactical categorical heads;
- pitch remains fixed at zero for the flat boxing MVP;
- checkpoints record
  `checkpoint_contract=netherite_pvp_mixed_actor_critic_v3` and are not shape
  compatible with v1/v2 actor heads.
