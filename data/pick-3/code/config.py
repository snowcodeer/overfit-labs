# config.py

MILESTONES = [
    {
        "label": "approach",
        "frame": 0,
        "description": "The hand begins moving toward the cuboid puzzle."
    },
    {
        "label": "grasp",
        "frame": 92,
        "description": "The hand makes contact and grips the cuboid puzzle."
    },
    {
        "label": "lift",
        "frame": 138,
        "description": "The hand slides the puzzle laterally across the wooden table."
    },
    {
        "label": "release",
        "frame": 210,
        "description": "The hand releases the puzzle at the new position."
    },
    {
        "label": "retract",
        "frame": 240,
        "description": "The hand moves away from the puzzle."
    }
]

REWARD_CONFIG = {
    "w_palm_obj": 2.0,       # Encourage hand to reach object
    "w_obj_target": 10.0,    # Main task: Move object to target
    "w_contact": 1.5,        # Bonus for maintaining contact
    "w_action": 0.01,        # Penalty for jerky movements
    "w_success": 50.0,       # Large bonus for reaching target
    "w_lift_penalty": 5.0,   # Penalty if the object is lifted off table (for slide task)
    "success_threshold": 0.05, # Distance in meters to consider 'success'
    "table_height": 0.02,    # Approx height of object on table
}