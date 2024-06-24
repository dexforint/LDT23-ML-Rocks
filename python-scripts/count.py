from glob import glob

print(len(glob("datasets/*/transformed_train/images/*")))
