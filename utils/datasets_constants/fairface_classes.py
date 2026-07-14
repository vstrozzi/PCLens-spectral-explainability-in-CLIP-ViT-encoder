# FairFace attribute label vocabularies (CSV string order -> index).
# Class-name lists (for zero-shot prompts) mirror the CSV categories.

fairface_gender_classes = ["male", "female"]

fairface_race_classes = [
    "white", "black", "latino hispanic", "east asian",
    "southeast asian", "indian", "middle eastern",
]

fairface_age_classes = [
    "0-2", "3-9", "10-19", "20-29", "30-39",
    "40-49", "50-59", "60-69", "more than 70",
]

# raw CSV string -> index, per attribute
GENDER_TO_IDX = {"Male": 0, "Female": 1}
RACE_TO_IDX = {
    "White": 0, "Black": 1, "Latino_Hispanic": 2, "East Asian": 3,
    "Southeast Asian": 4, "Indian": 5, "Middle Eastern": 6,
}
AGE_TO_IDX = {
    "0-2": 0, "3-9": 1, "10-19": 2, "20-29": 3, "30-39": 4,
    "40-49": 5, "50-59": 6, "60-69": 7, "more than 70": 8,
}

FAIRFACE_CLASSES = {
    "gender": fairface_gender_classes,
    "race": fairface_race_classes,
    "age": fairface_age_classes,
}
FAIRFACE_LABEL_MAP = {"gender": GENDER_TO_IDX, "race": RACE_TO_IDX, "age": AGE_TO_IDX}

# nr_of_classes / elements_per_class use race (the attribute in experiments_info.json,
# num_classes=7); gender has 2 classes and age has 9.
nr_of_classes = len(fairface_race_classes)

# FairFace val split is not class-balanced, so this is the minimum images per
# race. min 1209 (Middle Eastern), avg 1565, max 2085 (White); 10954 total.
fairface_elements_per_class = 1209
