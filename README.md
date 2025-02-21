# Hill Climb Racing Gesture Control

## 🚀 Overview
This project allows you to control the **Hill Climb Racing** game using **hand gestures** detected through a trained **Machine Learning model**. It uses **OpenCV, MediaPipe, and Scikit-Learn** to detect and classify hand gestures, which are then mapped to game controls.

## 🎮 Features
- **Real-time hand gesture recognition** 📷
- **ML-based gesture classification** 🤖
- **Hands-free control for Hill Climb Racing** 🚗
- **Custom dataset training for better accuracy** 🎯

## 🛠️ Tech Stack
- **Python** 🐍
- **OpenCV** (Computer Vision)
- **MediaPipe** (Hand Tracking)
- **Scikit-Learn** (ML Model Training)
- **PyAutoGUI** (Simulating Keypresses)

## 📂 Project Structure
```
Hill Climb Racing Gesture Control/
│── data/                  # Dataset for training
│── models/                # Saved ML models
│── scripts/
│   ├── collect_data.py    # Collects landmark data for training
│   ├── train_model.py     # Trains the ML model
│   ├── gesture_control.py # Runs real-time gesture control
│── README.md              # Project documentation
│── requirements.txt       # Dependencies
```

## 📝 How to Install & Run

### 1️⃣ Clone the Repository
```sh
 git clone https://github.com/yourusername/HillClimbGestureControl.git
 cd HillClimbGestureControl
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Collect Training Data
Run the script to collect hand landmark data:
```sh
python scripts/collect_data.py
```
Perform different gestures and save the data.

### 4️⃣ Train the Model
```sh
python scripts/train_model.py
```

### 5️⃣ Run Gesture Control
```sh
python scripts/gesture_control.py
```
Make sure **Hill Climb Racing** is open while running the script.

## 📌 Controls
| Gesture        | Action         |
|---------------|---------------|
| Open Hand     | Accelerate 🚀  |
| Closed Fist   | Brake ⏸️      |
| Thumbs Up     | Boost 🔥       |

## 📜 License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

## 🤝 Contributing
Feel free to fork this project and improve it! 🚀 Submit a **Pull Request** with your improvements.

## 🔗 Contact
📧 **Anish Kanojia** - [GitHub](https://github.com/AnishKanojia)
