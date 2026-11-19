# Stacking Three Cubes

Task: `Stack all three cubes.` One SO-101 arm, three identical cubes, one Pi-local agent, medium reasoning.

- Astra: `gpt-6-astra`, runs 09–18, Codex harness.
- Fable: `claude-fable-5-2`, runs 19–27 and 30, Claude Code 2.1.327.
- The tenth displayed Fable trial is run 30. Run 28 was replaced because its wrist camera disconnected; run 29 was operator-interrupted.

Open `index.html`. Text is editable in the browser; use **Save HTML copy** to retain edits. Keep the HTML beside `assets`.

Each model has ten collapsed trial rows. Expand a row to reveal its video, timestamps and notes; collapsing pauses playback.

Choose a section or trial in **Add to**, then **Add text** or **Add image**. Move or remove added blocks with their controls; **Undo remove** restores the last removal. Images can also be pasted or dropped into a section. Added images are resized and embedded in the exported HTML. If browser storage is full, save an HTML copy before closing.

`assets` contains overhead recordings at 20× (2 Hz source frames played at 40 fps). `data/runs.json` contains timing metrics and recorded-frame seek targets. Click a real elapsed timestamp to play from two video seconds before that milestone. Every failed run includes the recorded reason and exact final agent message. Visual timings are conservative confirmed-by times; grasp timing is the tool-batch timestamp. Placement-event notes distinguish failed releases from disruptions of an established stack.

Results contains five charts with SVG/PNG exports in `assets/charts` and inputs in `data/charts.json`. Use **Hide chart** and **Restore hidden charts** to edit the selection; saved HTML retains that choice. Charts use trial numbers and the page’s teal/brown model colours. Final positions estimate the gripper pose at final release in robot-base coordinates (X forward, Y left, centimetres); they are not calibrated measurements of the lowest cube centre. `data/placements.json` records the method, joint angles, release timestamps and source hashes. Fable trial 10 has all seven grasp calls reviewed: four retained lifts toward stacking, one miss, and two deliberate repositioning calls, giving 1.25 attempts per eligible pickup. `data/grasp-review-trial10.json` and the review contact sheets document that audit. Completion, stage, and position charts include successful trials only.

`baseline` contains the clean robot-interface code, FK/IK helpers, camera intrinsics/distortion, motor calibration, and home pose. These are inspection snapshots; execution requires the ArmFarm Pi services and Python dependencies. No credentials or agent session logs are included.

The comparison changes both model and harness. Cube positions were approximate. Completion requires three cubes freestanding, with the gripper clear, stable for five seconds. Run 30 had 3,981 fresh samples per camera, no stale samples over two seconds, and no motor faults.
