🧠 CIFAR-10 Image Classification using Deep Learning
This project focuses on building an image classification system using the CIFAR-10 dataset. The aim was to explore how different neural network architectures perform on a standard benchmark image dataset, starting with a basic model and advancing to ResNet50 with transfer learning.

📌 Features
✅ Trained on CIFAR-10, a popular dataset of 60,000 images across 10 categories.

✅ Implemented a basic neural network using Keras.

✅ Integrated ResNet50, a deep convolutional neural network, using transfer learning.

✅ Data preprocessing with normalization and resizing.

✅ Performance evaluation using accuracy metrics and training curves.

✅ Deployed using Google Colab with support for GPU acceleration.

✅ Applied techniques like Dropout, Batch Normalization, and RMSprop optimizer to improve results.

📁 Dataset
Name: CIFAR-10

Size: 60,000 images (32x32 pixels)

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

Source: CIFAR-10 Dataset

🛠️ Tools & Technologies
Python

TensorFlow & Keras

NumPy, Pandas, Matplotlib

Google Colab

Kaggle API

🔄 Data Preprocessing
Normalized pixel values to [0, 1] by dividing by 255.

Converted images and labels to NumPy arrays.

Resized images to 256x256 for ResNet50 input requirements.

🤖 Models Used
Basic Neural Network

Layers: Flatten → Dense → Dropout → Dense (Output)

Optimizer: Adam

Accuracy: ~55–60%

ResNet50 with Transfer Learning

Pre-trained on ImageNet

Layers frozen except final classifier

Optimizer: RMSprop

Accuracy: ~80–85%

📉 Evaluation
Used model.evaluate() on test data.

Visualized training using accuracy/loss curves.

Applied 10% validation split during training.

⚠️ Challenges Faced
Resizing from 32x32 to 256x256 for ResNet50 compatibility.

Overfitting in early stages, handled using Dropout and BatchNorm.

Managing large model size and GPU memory.

🎓 What I Learned
Hands-on implementation of CNNs and transfer learning.

Image preprocessing and data pipeline design.

Model tuning, visualization, and evaluation techniques.

Best practices in using TensorFlow/Keras.

🔧 Future Improvements
Add data augmentation to improve generalization.

Fine-tune layers of ResNet50 for better performance.

Try other models like InceptionV3 or EfficientNet.

Integrate early stopping and learning rate scheduling.
📊 Sample Results
Model	Accuracy
Basic NN	~58%
ResNet50	~83%

📬 Contact
For any queries, reach out via LinkedIn or email me at at9120140@gmail.com
