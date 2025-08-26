# NeuroSigny-AI 🚀
**Real-Time ASL to English Translation for Video Calls**

[![Python](https://img.shields.io/badge/Python-3.8+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Project Overview
**NeuroSigny-AI** is an innovative tool designed to bridge communication gaps by translating **American Sign Language (ASL) into English in real-time** during video calls.  
It allows Deaf individuals to communicate seamlessly with hearing participants, enhancing accessibility and inclusion in virtual meetings.

---

## 💡 Motivation
Despite advances in AI, real-time ASL translation in video calls is still limited.  
The goal of NeuroSigny-AI is to **enable smooth and natural conversations**, making digital communication more inclusive and accessible.

---

## ⚡ Key Features
- **Real-Time Translation**: Converts ASL gestures into English text instantly.
- **Video Call Integration**: Compatible with popular video conferencing platforms.
- **User-Friendly Interface**: Minimal setup, intuitive use.
- **Open Source**: Freely available for research and improvement.

---

## 🧠 Technical Overview
### Architecture
1. **Input**: Camera feed from the user.
2. **Preprocessing**: Hand keypoints extraction using OpenCV.
3. **ML Model**: Custom-trained TensorFlow model to recognize ASL gestures.
4. **Postprocessing**: Converts gestures to text and overlays in the video call in real-time.
5. **Frontend Integration**: React-based interface with WebRTC for video streaming.

### Tech Stack
- **Backend**: Python, TensorFlow, OpenCV
- **Frontend**: TypeScript, React
- **Video Integration**: WebRTC

---

## 🗂️ Project Structure
