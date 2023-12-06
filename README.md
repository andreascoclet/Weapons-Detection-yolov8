# Weapons-Detection-yolov8

### Solution Description
A web service with the ability to upload videos/photos for detecting the presence of weapons and armed individuals, as well as counting the confidence level in the presence of such elements in the images.

To prepare the service, we:

- Gathered data from open sources.
- Annotated images, highlighted bounding boxes for weapons and people with weapons.
- Implemented algorithms for geometric analysis and corrected distortions in the images.
- Performed augmentation (artificially increased the number of images).
- Trained the YoloV8 med model.
- Developed a React frontend.
- Integrated FastApi.
- Created instructions for connecting external cameras to our service.

The web service includes machine corrections for various camera distortions and the ability to provide feedback on the model's performance.

### Solution Structure
- The file `solution.py` is the main file, where models are initialized, inference is performed, and predictions are formatted.
- The file `model.py` contains the code responsible for model inference.

### Building the Solution 🐳
To begin, you need to install Docker. You can find detailed instructions on how to do this at the following link: [Docker Installation](https://docs.docker.com/get-docker/).

After you have installed Docker and ensured its functionality, you can build the current baseline. To do this, run the following command from the project's root directory:
```bash
docker build -t urbancode-baseline .
