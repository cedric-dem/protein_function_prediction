lowest_value = 0.05

NN_MODEL_NAME = "model_v7.keras"
XGB_MODEL_NAME = "model_v7_xgb.pkl"
KNN_MODEL_NAME = "model_v7_knn.pkl"

SUBMISSION_NAME = "model_v7.tsv"

STRATEGY = "NN"
HIDDEN_LAYER = 2000
EXPONENT_OCCURENCES =  3

positions = ['A', "B", 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', "U", 'V', 'W', "X", 'Y', "Z"]
# positions = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# Dont know why but some have b, u,x

BATCH_SIZE_TRAIN = 32
N_EPOCHS = 10

BATCH_SIZE_TEST = 256