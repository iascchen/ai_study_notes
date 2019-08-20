import hashlib
import os

import tensorflow_hub as hub

models_path = "../../models"

os.environ['TFHUB_CACHE_DIR'] = models_path
print("TFHUB_CACHE_DIR", os.environ.get('TFHUB_CACHE_DIR'))

module_handle = "https://tfhub.dev/google/openimages_v4/ssd/mobilenet_v2/1"

handle_hash = hashlib.sha1(module_handle.encode("utf8")).hexdigest()  # f34b2684786cf6de38511148638abf91283beb1f
# module_handle = "%s/%s" % (models_path, handle_hash)

obj_detector_1 = hub.Module(module_handle)
print(obj_detector_1.get_input_info_dict())
print(obj_detector_1.get_output_info_dict())
print("obj_detector done")

#####################
# Download some modules
#####################

classifier_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/3"
classifier_hash = hashlib.sha1(classifier_url.encode("utf8")).hexdigest()
print("classifier_hash", classifier_hash)
# classifier_hash 8ba51acecbfe5ceeaf1c04d6ee05b1703dd63bf0
classifier = hub.Module(classifier_url)
print("classifier done")

feature_extractor_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/3"
feature_extractor_hash = hashlib.sha1(feature_extractor_url.encode("utf8")).hexdigest()
print("feature_extractor_hash", feature_extractor_hash)
# feature_extractor_hash 9a40df43ae974de74f59ca892971f265fec3d319
feature_extractor = hub.Module(feature_extractor_url)
print("feature_extractor done")

obj_detector2_url = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"
obj_detector2_hash = hashlib.sha1(obj_detector2_url.encode("utf8")).hexdigest()
print("obj_detector2_hash", obj_detector2_hash)
# obj_detector2_hash 6e850c920451d5243d1fb87a3242c087535b9183
obj_detector2 = hub.Module(obj_detector2_url)
print("obj_detector2 done")
