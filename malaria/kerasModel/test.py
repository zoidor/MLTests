from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

modelPath = "../../kerasModel/malaria_model.h5"
images_path = "../../TF/application/data/images/"

net_model = load_model(modelPath)

valAug = ImageDataGenerator(rescale = 1.0 / 255.0)
testGen = valAug.flow_from_directory(
	images_path,
	class_mode="categorical",
	target_size=(64, 64),
	color_mode="rgb",
	shuffle=False,
	batch_size=1)

testGen.reset()
predIdxs = net_model.predict_generator(testGen, steps = len(testGen))

print(predIdxs)

