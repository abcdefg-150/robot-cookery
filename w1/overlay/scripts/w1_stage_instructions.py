"""Derive frame-aligned English instructions from W1 expert state transitions."""


OBJECT_NAMES = {
    "SMALL_PEPPER": "the small green pepper",
    "GREEN_PEPPER": "the large green pepper",
    "POTATO": "the potato",
    "EGGPLANT": "the eggplant",
    "SPATULA": "the spatula",
}

OBJECT_ACTIONS = {
    "APPROACH": "Move the gripper above {object}.",
    "DESCEND": "Lower the gripper to {object}.",
    "GRASP": "Close the gripper around {object}.",
    "LIFT": "Lift {object} off the table.",
    "TRANSFER": "Carry {object} to the pan.",
    "RELEASE": "Place {object} in the pan and release it.",
    "RETREAT": "Move the empty gripper up and away from the pan.",
    "PARK": "Return the arm to its home position.",
    "POLICY": "Pick up and place {object} in the pan.",
}

FIXED = {
    "SETUP": "Hold both arms at the starting position.",
    "IDLE": "Hold the arm still.",
    "LEFT_ARM_PARK": "Return the left arm to its home position.",
    "RIGHT_ARM_PARK": "Return the right arm to its home position.",
    "TOOL_CLEARANCE_PARK": "Move the left arm away from the spatula workspace.",
    "FINAL_PARK": "Return the tool arm to its home position.",
    "FINAL_LEFT_PARK": "Return the left arm to its home position.",
    "FINAL_SETTLE": "Keep both arms still while the food settles in the pan.",
    "DONE": "Keep both arms at rest after completing the task.",
    "STIR_SEARCH": "Hold the spatula while preparing to stir.",
    "STIR_APPROACH": "Bring the spatula above the vegetables in the pan.",
    "STIR_LOWER": "Lower the spatula into the pan near the vegetables.",
    "STIR_PUSH": "Push the vegetables gently with the spatula.",
    "STIR_REVERSE": "Draw the spatula back for the next stirring stroke.",
    "STIR_WITHDRAW": "Lift the spatula clear of the pan.",
    "STIR_VERIFIED": "Hold the spatula steady above the pan after stirring.",
    "STIR_POLICY": "Stir the vegetables with short back-and-forth strokes.",
    "SPATULA_RETURN_CLEAR": "Lift the spatula clear of the pan before returning it.",
    "SPATULA_RETURN_APPROACH": "Carry the spatula to its resting spot on the table.",
    "SPATULA_RETURN_LOWER": "Lower the spatula onto its resting spot.",
    "SPATULA_RETURN_RELEASE": "Open the gripper to release the spatula.",
    "SPATULA_RETURN_RETREAT": "Move the empty gripper clear of the spatula.",
    "PAIR_FINAL_PARK": "Return both arms to their home positions.",
    "PAIR_FINAL_CLOSE": "Close both empty grippers at the home positions.",
}

TURN_ACTIONS = {
    "APPROACH": "Bring the spatula above the food to prepare a turn.",
    "INSERT": "Lower the spatula beside the food.",
    "SCOOP": "Scoop and turn the food with the spatula.",
    "WITHDRAW": "Lift the spatula clear of the pan.",
    "VERIFIED": "Hold the spatula steady above the pan.",
    "POLICY": "Turn the food with the spatula.",
}

PAIRS = {
    "1": ("the large green pepper", "the small green pepper"),
    "2": ("the eggplant", "the potato"),
}

PAIRED_ACTIONS = {
    "APPROACH": "Move the left gripper above {left} and the right gripper above {right}.",
    "OPEN": "Open both grippers above the ingredients.",
    "DESCEND": "Lower the left gripper to {left} and the right gripper to {right}.",
    "GRASP": "Grasp {left} with the left gripper and {right} with the right gripper.",
    "LIFT": "Lift {left} and {right} off the table together.",
    "TRANSFER": "Carry {right} over the pan and bring {left} to the left side of the pan.",
    "RIGHT_LOWER": "Lower {right} into the pan while holding {left} to the left of the pan.",
    "RIGHT_RELEASE": "Release {right} into the pan while holding {left} to the left of the pan.",
    "LEFT_ENTER_RIGHT_RETREAT": "Bring {left} over the pan as the right gripper moves away.",
    "LEFT_LOWER": "Lower {left} into the pan.",
    "LEFT_RELEASE": "Release {left} into the pan.",
    "LEFT_RETREAT": "Move the empty left gripper up and away from the pan.",
}


def stage_instruction(phase):
    if phase in FIXED:
        return FIXED[phase]
    if phase.startswith("PAIR_"):
        _, index, action = phase.split("_", 2)
        left, right = PAIRS[index]
        return PAIRED_ACTIONS[action].format(left=left, right=right)
    if phase.startswith("TURN_"):
        return TURN_ACTIONS[phase.rsplit("_", 1)[1]]
    for key, name in OBJECT_NAMES.items():
        prefix = key + "_"
        if phase.startswith(prefix):
            return OBJECT_ACTIONS[phase[len(prefix):]].format(object=name)
    raise ValueError(f"Unmapped W1 expert phase: {phase}")


def arm_instruction(global_phase, arm_phase, side):
    hold = f"Hold the {side} arm still."
    if global_phase == "PAIR_FINAL_CLOSE":
        return f"Close the empty {side} gripper at its home position."
    if global_phase == "PAIR_FINAL_PARK":
        return f"Return the {side} arm to its home position."
    if not global_phase.startswith("PAIR_"):
        return stage_instruction(arm_phase) if arm_phase == global_phase else hold

    _, index, action = global_phase.split("_", 2)
    target = PAIRS[index][0 if side == "left" else 1]
    gripper = f"the {side} gripper"
    if action == "OPEN":
        return f"Open {gripper} above {target}."
    if action in {"RIGHT_LOWER", "RIGHT_RELEASE"} and side == "left":
        return f"Hold {target} just outside the pan with {gripper}."
    if action in {"LEFT_LOWER", "LEFT_RELEASE", "LEFT_RETREAT"} and side == "right":
        return hold
    if action.endswith("_LOWER"):
        return f"Lower {target} into the pan with {gripper}."
    if action.endswith("_RELEASE"):
        return f"Open {gripper} to release {target} into the pan."
    if action == "TRANSFER" and side == "left":
        return f"Carry {target} to the left side of the pan."
    return stage_instruction(arm_phase)


def write_stage_instructions(h5_file):
    """Write labels aligned to the existing simulated FSM samples."""
    import h5py

    group = h5_file["additional_info"]
    phases = group["fsm_state"].asstr()[:]
    values = {"fsm_instruction": [stage_instruction(phase) for phase in phases]}
    if any(phase.startswith("PAIR_") for phase in phases):
        for side in ("left", "right"):
            arm_phases = group[f"{side}_fsm_state"].asstr()[:]
            values[f"{side}_fsm_instruction"] = [
                arm_instruction(phase, arm, side)
                for phase, arm in zip(phases, arm_phases, strict=True)
            ]
    for name, instructions in values.items():
        if name in group:
            if group[name].asstr()[:].tolist() != instructions:
                raise ValueError(f"Existing W1 instructions differ: {name}")
            continue
        group.create_dataset(name, data=instructions, dtype=h5py.string_dtype("utf-8"))
    group.attrs["stage_instruction_source"] = "expert_fsm_templates_v1"
