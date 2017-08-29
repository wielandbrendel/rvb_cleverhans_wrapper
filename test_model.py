import sys
sys.path.insert(0, '/gpfs01/bethge/home/wbrendel/github/foolbox')

import foolbox
print(foolbox.__path__)
import keras
import numpy
from keras.applications.resnet50 import ResNet50

from model import RobustVisionModel

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
preprocessing = (numpy.array([104, 116, 123]), 1)
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255), preprocessing=preprocessing)

# get source image and label
image, label = foolbox.utils.imagenet_example()

# instantiate adversarial object
criterion = foolbox.criteria.Misclassification()
adversarial = foolbox.Adversarial(fmodel, criterion, image, label)

# turn adversarial object into CleverHans model
cmodel = RobustVisionModel(adversarial)

# apply CleverHans attack on image
from cleverhans.attacks import FastGradientMethod
fgsm = FastGradientMethod(cmodel)
x_adv = fgsm.generate(image, eps=16, clip_min=0, clip_max=255)

print(x_adv.shape)
