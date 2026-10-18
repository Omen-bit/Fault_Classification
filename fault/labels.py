"""Reference tables from the brief (Appendix A and B). No logic."""

# Appendix B order. Rule C5 picks the earlier of two equally supported categories.
LABELS = ["no_power", "not_heating", "not_cooling", "water_leak", "drainage_or_blockage",
          "noise_or_vibration", "door_seal_or_latch", "control_panel_or_display",
          "electrical_or_control_board", "motor_or_drive", "installation_or_setup", "other"]
ORDER = {label: i for i, label in enumerate(LABELS)}

PRODUCTS = ["washing_machine", "refrigerator", "dishwasher", "oven_or_cooktop",
            "air_conditioner", "water_heater", "microwave"]
GAS_PRODUCTS = {"oven_or_cooktop", "water_heater"}        # rule C2 applies to these only

NONE = "none"                                             # a sentence that carries no symptom
