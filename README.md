# Stacking Three Cubes

Task: `Stack all three cubes.` One SO-101 arm, three identical cubes, one Pi-local agent, medium reasoning.

- Astra: `gpt-6-astra`, runs 09–18, Codex harness.
- Fable: `claude-fable-5-2`, runs 19–27 and 30, Claude Code 2.1.327.
- The tenth displayed Fable trial is run 30. Run 28 was replaced because its wrist camera disconnected; run 29 was operator-interrupted.

Open `index.html`. Text is editable in the browser; use **Save HTML copy** to retain edits. Keep the HTML beside `assets`.

`assets` contains the overhead recordings, accelerated from the 2 Hz source to under two minutes each. `data/runs.json` contains timing metrics. Visual timings are conservative confirmed-by times; grasp timing is the tool-batch timestamp.

`baseline` contains the clean robot-interface code, FK/IK helpers, camera intrinsics/distortion, motor calibration, and home pose. These are inspection snapshots; execution requires the ArmFarm Pi services and Python dependencies. No credentials or agent session logs are included.

The comparison changes both model and harness. Cube positions were approximate. Completion requires three cubes freestanding, with the gripper clear, stable for five seconds. Run 30 had 3,981 fresh samples per camera, no stale samples over two seconds, and no motor faults.
