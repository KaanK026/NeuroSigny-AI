# NeuroSigny-AI 
**Real-Time ASL to English Translation for Virtual Meetings**

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-4.9-blue?style=flat-square&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![AWS](https://img.shields.io/badge/AWS-Cloud-orange?style=flat-square&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![CSS](https://img.shields.io/badge/CSS-3-blue?style=flat-square&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![React](https://img.shields.io/badge/React-18-blue?style=flat-square&logo=react&logoColor=white)](https://reactjs.org/)
[![HTML](https://img.shields.io/badge/HTML-5-orange?style=flat-square&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview
**NeuroSigny-AI** is an innovative tool designed to bridge communication gaps by translating **American Sign Language (ASL) into English in real-time** during video calls.  
It allows individuals with hearing disabilities to communicate seamlessly with other participants, enhancing accessibility and inclusion in virtual meetings.

---

## 💡 Motivation
Despite advances in AI, real-time ASL translation in video calls is still limited. AI is a very strong tool that can support people with disabilities in working environments and creating equal opportunities.
The goal of NeuroSigny-AI is to **enable smooth and natural conversations**, making digital communication more inclusive and accessible.

---

## ⚡ Key Features
- **Real-Time Translation**: Converts ASL gestures into English text instantly.
- **Video Call Integration**: Compatible with popular video conferencing platforms.
- **User-Friendly Interface**: Minimal setup, intuitive use.
- **Open Source**: Freely available for research and improvement.

---

## 🏗️ Architecture

### Model Choices
The application allows users and developers to choose a model according to their needs. Each model has been chosen for a particular reason and analyzed for its properties.

**ResNet18**:  
- Chosen for its **proven accuracy** on image classification tasks while remaining relatively lightweight.  
- Its **residual connections** help prevent vanishing gradients, making training more stable.  
- Ideal for users who want a **balance between performance and computational efficiency**.

**MobileNetV3-large**:  
- Selected for its **high efficiency on mobile and edge devices**.  
- Uses **depthwise separable convolutions** and optimized architecture for speed.  
- Perfect for users who need **fast inference with limited hardware resources**.

**Custom CNN Model**:  
- Designed for **experimentation and flexibility**.  
- Allows developers to **tweak the architecture** (number of layers, filters, etc.) to suit custom datasets.  
- Serves as a **lightweight alternative** for quick prototyping or smaller-scale tasks.


### Hyperparameters and Tuning

- **Mixup & CutMix** Encourages models to learn more generalized features rather than overfitting to the training set.
- **Adam Optimizer:** Fast convergence for ResNet18 & Custom CNN
- **Sdg Optimizer:** Stable training for MobileNetV3-large
- **Scheduler:** ReduceLROnPlateau helps refine learning toward convergence. 


<img width="1808" height="1202" alt="analysis" src="https://github.com/user-attachments/assets/1b85bddd-a79e-4fe4-b61d-cfd2ca9b1636" />
Image can be seen in full resolution on clicking.


## 📊Demonstration
### Backend
![backend_git-ezgif com-optimize (1)](https://github.com/user-attachments/assets/745818d5-756e-4c18-83bd-27685eb06559)

### Frontend
![frontend_git](https://github.com/user-attachments/assets/cd6e6640-ad2f-49ff-83af-00df7ae683c9)



---

## 🧠 Technical Overview
1. **Input**: Camera feed from the user.
2. **Preprocessing**: Hand keypoints extraction using OpenCV and Mediapipe.
3. **ML Model**: Custom-trained and Pre-trained Pytorch models to recognize ASL gestures.
4. **Postprocessing**: Converts gestures to text and overlays in the video call in real-time.
5. **Frontend Integration**: React-based interface

### Tech Stack
- **Backend**: Python, Pytorch, Scikit-learn, OpenCV, Mediapipe, FastAPI
- **Frontend**: TypeScript, React, CSS, HTML
- **Cloud**: AWS S3


---

##  Limitations
The project currently does not cover all ASL expressions. Instead, it is based on constructing words by combining individual letters in ASL. 

## 🗂️ Project Structure
- `backend/` – ML models & Vision logic
  - `src/` 
    - `models/` – Model Architectures 
    - `trainings/` – Training Logics
    - `utils/` -Utils
    - `app.py` – API
    - `asl_prediction_ai.py` - Model Evaluation and Vision
    - `config.py` - Configurations
  - `requirements.txt`  
- `frontend/` – UI & video call integration  
  - `public/` 
  - `src/`   
    - `App.tsx` – Frontend entry point 
    - `App.css` – Frontend Styling
  - `package.json` 
- `.gitignore`,`README.md`

---

## 🛠️ Installation & Quick Start

**Steps:**
```bash
git clone https://github.com/KaanK026/NeuroSigny-AI.git
cd NeuroSigny-AI
# Backend
cd backend
pip install -r requirements.txt
## Frontend
cd ../frontend
npm install
npm start
```

---

## 🤝 Contributing
Contributions are welcome! To contribute:  
1. Fork the repository  
2. Create a branch 
3. Make changes and commit
4. Push Branch
5. Open a Pull Request with a description of your changes

## 📄 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact
**Kaan** – [kaankocaer026@gmail.com](mailto:your-email@example.com)  
GitHub: [https://github.com/KaanK026](https://github.com/KaanK026)  

---

*Made by Kaan Kocaer – Politecnico di Torino Computer Engineering Exp. Grad: 2027 *
