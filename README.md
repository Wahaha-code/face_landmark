# Face Detection Project :blush:

This project is a real-time,high accuracy face detection. The main script of the project is `main.py`.

## Requirements

- Python 3.8 or later
- PyTorch 2.1.0 or later
- torchvision 0.16.0 or later
- OpenCV 4.5.2 or later

## Getting Started

First, clone the repository to your local machine:

```bash
git clone https://github.com/Wahaha-code/face_landmark.git
cd main.py
```

## Usage

You can run the script `main.py` with several options:

- `--weights`: Path(s) to the model weights. Default is 'checkpoints\detface.pt'.
- `--source`: The source of the images to process. This can be a file, a folder, or '0' for webcam. Default is '0'.
- `--img-size`: The size of the images for inference, in pixels. Default is 640.
- `--project`: The directory to save the results to. Default is 'runs/detect'.
- `--name`: The name of the experiment. Results will be saved to 'project/name'. Default is 'exp'.
- `--exist-ok`: If set, existing project/name is okay, do not increment. Default is True.
- `--save-img`: If set, save the results as images. Default is True.
- `--view-img`: If set, show the results. Default is True.

Here is an example of how to run the script:
```bash
python main.py --weights checkpoints\detface.pt --source 0 --img-size 640 --project runs/detect --name exp --exist-ok --save-img --view-img
```

This will run the script with the default options, using the webcam as the source and saving the results to 'runs/detect/exp'.

## License

This project is licensed under the terms of the CAIR license.
