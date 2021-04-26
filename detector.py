from imageai.Detection import ObjectDetection
detector=ObjectDetection()
models_path='/home/kushagra/python/object detection/models/yolo-tiny.h5'
input_path='/home/kushagra/python/object detection/input/download.jpeg'
output_path='/home/kushagra/python/object detection/output/new-image2.jpg'
detector.setModelTypeAsTinyYOLOv3()
detector.setModelPath(models_path)
detector.loadModel()
detection = detector.detectObjectsFromImage(input_image=input_path,output_image_path=output_path)
for eachItem in detection:
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])