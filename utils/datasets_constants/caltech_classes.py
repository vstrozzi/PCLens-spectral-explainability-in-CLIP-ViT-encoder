# Caltech-101 class names in torchvision.datasets.Caltech101 label order.
# torchvision builds labels as sorted(os.listdir("101_ObjectCategories")) with
# "BACKGROUND_Google" removed, so the four capitalised folders (Faces,
# Faces_easy, Leopards, Motorbikes) sort ahead of the lowercase ones. Names are
# spaced (underscores -> spaces) for CLIP prompts; index == label.
caltech_101_classes = [
    "faces", "faces easy", "leopards", "motorbikes", "accordion", "airplanes",
    "anchor", "ant", "barrel", "bass", "beaver", "binocular", "bonsai", "brain",
    "brontosaurus", "buddha", "butterfly", "camera", "cannon", "car side",
    "ceiling fan", "cellphone", "chair", "chandelier", "cougar body",
    "cougar face", "crab", "crayfish", "crocodile", "crocodile head", "cup",
    "dalmatian", "dollar bill", "dolphin", "dragonfly", "electric guitar",
    "elephant", "emu", "euphonium", "ewer", "ferry", "flamingo", "flamingo head",
    "garfield", "gerenuk", "gramophone", "grand piano", "hawksbill", "headphone",
    "hedgehog", "helicopter", "ibis", "inline skate", "joshua tree", "kangaroo",
    "ketch", "lamp", "laptop", "llama", "lobster", "lotus", "mandolin", "mayfly",
    "menorah", "metronome", "minaret", "nautilus", "octopus", "okapi", "pagoda",
    "panda", "pigeon", "pizza", "platypus", "pyramid", "revolver", "rhino",
    "rooster", "saxophone", "schooner", "scissors", "scorpion", "sea horse",
    "snoopy", "soccer ball", "stapler", "starfish", "stegosaurus", "stop sign",
    "strawberry", "sunflower", "tick", "trilobite", "umbrella", "watch",
    "water lilly", "wheelchair", "wild cat", "windsor chair", "wrench",
    "yin yang",
]

nr_of_classes = len(caltech_101_classes)

# Caltech-101 is not class-balanced, so this is the minimum images per class.
# min 31 (inline_skate), avg 86, max 800 (airplanes); 8677 images total.
caltech_101_elements_per_class = 31
