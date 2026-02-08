# Driver_drowsiness
Overview

This project is an AI-based real-time driver monitoring system designed to improve road safety by detecting driver drowsiness and distraction. The system uses computer vision techniques and deep learning models to analyze facial landmarks and driver behavior through a webcam. When signs of fatigue or distraction are detected, the system immediately provides audio and visual alerts to warn the driver.

Features

Real-time driver monitoring using webcam

Drowsiness detection using Eye Aspect Ratio (EAR)

Yawning detection using Mouth Aspect Ratio (MAR)

Distraction detection such as mobile phone usage using deep learning

Instant audio and visual alert system

Works on standard hardware without wearable sensors

Lightweight and efficient implementation

How It Works

The system captures live video from the webcam and processes each frame using computer vision techniques. Facial landmarks are detected to monitor eye and mouth movements. EAR is used to detect prolonged eye closure, and MAR is used to detect yawning. Additionally, a deep learning model detects distraction such as mobile phone usage. If drowsiness or distraction is detected, the system triggers an alert to notify the driver.

Technologies Used

Python

OpenCV

MediaPipe

TensorFlow / Deep Learning Model

NumPy

Pygame (for audio alerts)

Applications

Driver safety systems

Accident prevention systems

Smart vehicle monitoring

Fleet management systems

Advantages

Real-time detection

Non-intrusive (no wearable sensors required)

Low computational cost

Easy to deploy

Improves road safety

Future Improvements

Integration with IoT devices

Mobile application support

Cloud-based monitoring

More advanced deep learning models

Integration with vehicle systems

Conclusion

This project provides an efficient and cost-effective solution to detect driver drowsiness and distraction in real time. It helps prevent accidents and enhances road safety using AI and computer vision technologies.
