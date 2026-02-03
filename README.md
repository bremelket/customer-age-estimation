# Customer Age Estimation from Photos using Computer Vision

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-red.svg)](https://keras.io/)

**Deep learning model estimating customer age from facial images for retail analytics and age-restricted product compliance.**

**Author:** Ekaterina Bremel  
**Project Type:** Computer Vision, Deep Learning  
**Status:** Completed

---

## 🎯 Project Overview

Developed a Convolutional Neural Network (CNN) using ResNet50 architecture to estimate customer age from facial images captured at checkout. The system supports retail analytics (targeted marketing by age group) and compliance monitoring for age-restricted product sales.

### Business Impact
- **Goal:** Automate age estimation for retail operations
- **Use Cases:** 
  - Targeted product recommendations based on age demographics
  - Age verification assistance for alcohol/tobacco sales
  - Customer analytics and market segmentation
- **Result:** Achieved MAE of 5.89 years, enabling reliable age group categorization

---

## 📊 Dataset Description

**Image Dataset:**
- Facial photos of customers at various ages
- Labeled with actual age (ground truth)
- Diverse demographics and lighting conditions
- Preprocessed and augmented for better model performance

**Data Characteristics:**
- Images captured at checkout points
- Real-world retail environment conditions
- Age range: Multiple age groups from young adults to seniors
- Challenge: Limited data for minors (under 18), affecting accuracy for age verification use case

---

## 🛠️ Technical Approach

### 1. Exploratory Data Analysis
- Analyzed age distribution in dataset
- Examined image quality and preprocessing requirements
- Identified data imbalances (fewer samples of minors)
- Visualized sample images across age ranges

### 2. Data Preprocessing & Augmentation
- Image normalization and resizing
- Data augmentation techniques:
  - Random rotation
  - Horizontal flipping
  - Brightness/contrast adjustments
- Train/validation/test split

### 3. Model Architecture
**Base Model:** ResNet50 (Pre-trained on ImageNet)
- **Architecture:** Deep Convolutional Neural Network with residual connections
- **Transfer Learning:** Leveraged pre-trained ResNet50 weights
- **Fine-tuning:** Adapted final layers for age regression task
- **Output:** Continuous age prediction (regression problem)

**Why ResNet50?**
- Proven performance on image recognition tasks
- Residual connections prevent vanishing gradients
- Pre-trained weights accelerate training
- Excellent feature extraction capabilities

### 4. Training Process
- **GPU Acceleration:** Utilized GPU for faster training
- **Optimizer:** Adam optimizer
- **Loss Function:** Mean Absolute Error (MAE)
- **Epochs:** 10 training epochs
- **Batch Size:** Optimized for GPU memory
- **Early Stopping:** Monitored validation performance

### 5. Model Evaluation

**Performance Metrics:**
- **Initial MAE:** 10.0 years (epoch 1)
- **Final MAE:** 2.76 years (epoch 10)
- **Test Set MAE:** 5.89 years

**Improvement:** 72% reduction in error over training period

**Performance Analysis:**
- ✅ Reliable for broad age group categorization (8-12 year ranges)
- ✅ Suitable for marketing segmentation (Gen Z, Millennials, Gen X, Boomers)
- ⚠️ Limited accuracy for minor age verification due to sparse training data
- ✅ Performs well in typical retail lighting conditions

---

## 🚀 Key Results

✅ **Test MAE: 5.89 years** - Accurate enough for age group classification  
✅ **Training MAE: 2.76 years** - Strong learning from training data  
✅ **72% error reduction** - Significant improvement over 10 epochs  
✅ **Production-ready** - Deployable for retail analytics use cases  
⚠️ **Limitation:** Less reliable for precise age verification of minors

---

## 💻 Technologies Used

**Deep Learning Framework:**
- TensorFlow 2.x
- Keras API
- GPU acceleration (CUDA)

**Computer Vision:**
- ResNet50 architecture
- Transfer learning
- Image preprocessing & augmentation

**Data Science Stack:**
- Python 3.8+
- NumPy - Numerical operations
- Pandas - Data manipulation
- Matplotlib - Visualization

**Development:**
- Jupyter Notebook
- Google Colab (GPU runtime)

---

## 📁 Project Structure

```
customer-age-estimation/
├── index.html              # Full analysis notebook (HTML export)
├── README.md              # This file
└── data/                  # (Data not included - privacy)
    └── photos/            # Customer facial images with age labels
```

---

## 🔍 View Full Analysis

**[👉 View Interactive Notebook (HTML)](index.html)**

The complete analysis includes:
- Exploratory data analysis with visualizations
- Data preprocessing pipeline
- Model architecture details
- Training curves (loss over epochs)
- Performance evaluation
- Sample predictions with actual vs. predicted ages

---

## 📈 Sample Visualizations

The project includes:
- Age distribution histograms
- Sample images with predictions
- Training/validation loss curves
- MAE improvement over epochs
- Confusion matrix for age groups
- Model prediction examples

---

## 🎓 Skills Demonstrated

**Deep Learning:**
- Convolutional Neural Networks (CNNs)
- Transfer learning with pre-trained models
- ResNet50 architecture
- Image classification/regression
- Model fine-tuning

**Computer Vision:**
- Image preprocessing
- Data augmentation techniques
- Feature extraction
- GPU-accelerated training

**Model Development:**
- Training deep neural networks
- Hyperparameter optimization
- Overfitting prevention
- Performance evaluation
- Production deployment readiness

**Business Application:**
- Understanding retail use cases
- Age demographics analysis
- Compliance requirements
- Practical limitations awareness

---

## 🌟 Real-World Applications

**Retail Analytics:**
- Customer demographics tracking
- Store traffic analysis by age group
- Product placement optimization

**Marketing:**
- Age-targeted promotions
- Personalized recommendations
- Campaign effectiveness measurement

**Compliance Support:**
- Age verification assistance (not replacement) for restricted products
- Cashier alerts for potential underage sales
- Audit trail for compliance reporting

**Note:** System is designed as a support tool, not sole verification method for legal age restrictions.

---

## 🔮 Future Improvements

**Model Enhancements:**
- Collect more data for underrepresented age groups (especially minors)
- Experiment with other architectures (EfficientNet, Vision Transformer)
- Ensemble multiple models for improved accuracy
- Multi-task learning (age + gender + ethnicity)

**Production Features:**
- Real-time inference API
- Confidence scores for predictions
- Alert system for uncertain predictions
- A/B testing framework

**Technical Optimization:**
- Model quantization for faster inference
- Edge deployment (run on cameras directly)
- Batch processing for analytics

---

## ⚠️ Ethical Considerations

**Privacy:**
- Images used only for age estimation
- No personal identification or storage
- Compliance with data protection regulations

**Bias Awareness:**
- Model performance may vary across demographics
- Continuous monitoring for fairness
- Regular retraining with diverse data

**Appropriate Use:**
- Support tool, not replacement for human judgment
- Clear communication of limitations
- Responsible deployment practices

---

## 📧 Contact

**Ekaterina Bremel**
- LinkedIn: [Ekaterina Bremel](https://www.linkedin.com/in/ekaterina-bremel-65b1b1238/)
- Email: bremelket@gmail.com
- GitHub: [@bremelket](https://github.com/bremelket)

---

## 📝 License

Copyright © 2026 Ekaterina Bremel. All rights reserved.

This project is available for viewing as part of my portfolio. Unauthorized copying, modification, distribution, or use of this code is prohibited.

---

**⭐ If you find this project interesting, please consider giving it a star!**
