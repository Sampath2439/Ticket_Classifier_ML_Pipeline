# 🎬 Demo Video Script - Customer Support Ticket Classification System

## 📋 Demo Overview
This script outlines the key points to cover in the demo video for the Customer Support Ticket Classification System.

## 🎯 Demo Structure (5-7 minutes)

### 1. Introduction (30 seconds)
**Script:**
"Hello! Today I'll demonstrate a comprehensive machine learning pipeline for customer support ticket classification. This system automatically categorizes tickets by issue type and urgency level while extracting key entities like products, dates, and order numbers."

**Show:**
- Project overview in README.md
- File structure

### 2. Dataset Overview (45 seconds)
**Script:**
"We're working with 1000 customer support tickets containing 7 different issue types like Billing Problems, Product Defects, and Installation Issues, with 3 urgency levels across 10 different products."

**Show:**
- Open tickets_dataset.csv
- Scroll through data to show variety
- Point out missing values and data challenges

### 3. Command Line Demo (90 seconds)
**Script:**
"Let's start with the command-line interface. I'll run the main pipeline which loads the data, trains two Random Forest classifiers, and generates performance metrics."

**Show:**
- Run `python ticket_classifier.py`
- Highlight key metrics:
  - Issue Type Classifier: 94.1% accuracy
  - Urgency Level Classifier: 37.9% accuracy (explain class imbalance)
- Show demo predictions with entity extraction
- Display generated visualizations

### 4. Web Interface Demo (2-3 minutes)
**Script:**
"Now let's explore the interactive web interface built with Gradio. This provides a user-friendly way to interact with our models."

**Show:**
#### Model Training Tab:
- Click "Train Models" button
- Show training progress and results

#### Single Ticket Prediction:
- Enter example ticket: "My SmartWatch V2 is broken and stopped working after 3 days. Order #12345. This is urgent!"
- Click Predict
- Explain results:
  - Issue Type: Product Defect (high confidence)
  - Urgency Level: Medium
  - Extracted entities: SmartWatch V2, #12345, complaint keywords
- Try 2-3 more examples from the provided examples

#### Batch Processing:
- Upload sample_tickets.csv
- Process batch
- Show results table with all predictions
- Explain batch summary statistics

### 5. Technical Architecture (60 seconds)
**Script:**
"Let me explain the technical approach. We use traditional NLP techniques with TF-IDF vectorization combined with engineered features like sentiment analysis and text statistics. The models are Random Forest classifiers chosen for their interpretability and robustness."

**Show:**
- Code structure in ticket_classifier.py
- Highlight key methods:
  - Text preprocessing
  - Feature engineering
  - Model training
  - Entity extraction

### 6. Key Features & Benefits (45 seconds)
**Script:**
"Key features include comprehensive text preprocessing, multi-task learning for both classification tasks, rule-based entity extraction, confidence scoring, and both single and batch processing capabilities."

**Show:**
- Feature list in README
- Performance metrics
- Visualization outputs

### 7. Limitations & Future Work (30 seconds)
**Script:**
"Current limitations include rule-based entity extraction and moderate urgency prediction accuracy due to class imbalance. Future improvements could include deep learning models, advanced NER, and active learning capabilities."

**Show:**
- Limitations section in README
- Confusion matrices showing areas for improvement

### 8. Conclusion (15 seconds)
**Script:**
"This system demonstrates a complete ML pipeline from data preprocessing to deployment, providing practical value for customer support automation. Thank you for watching!"

**Show:**
- Final results summary
- Generated files

## 🎥 Recording Tips

### Technical Setup:
- Record in 1080p resolution
- Use clear audio (consider external microphone)
- Screen recording software: OBS Studio, Camtasia, or built-in tools
- Prepare all files and close unnecessary applications

### Presentation Tips:
- Speak clearly and at moderate pace
- Use mouse highlighting for important elements
- Pause briefly between sections
- Show actual results, not just code
- Explain technical terms briefly

### Demo Flow:
1. Start with file overview
2. Run command-line demo first (shows full pipeline)
3. Then demonstrate web interface
4. End with technical discussion

## 📝 Key Points to Emphasize

### Technical Excellence:
- **94.1% accuracy** on issue type classification
- Comprehensive **text preprocessing** pipeline
- **Multi-task learning** approach
- **Entity extraction** with multiple types
- **Robust error handling** and validation

### Practical Value:
- **Real-world dataset** with missing values
- **User-friendly interface** for non-technical users
- **Batch processing** for operational efficiency
- **Confidence scores** for decision support
- **Visualization** for model interpretation

### Code Quality:
- **Modular design** with clear separation of concerns
- **Comprehensive testing** with test_pipeline.py
- **Detailed documentation** and README
- **Error handling** and edge cases
- **Professional code structure**

## 🚀 Demo Checklist

### Before Recording:
- [ ] All dependencies installed
- [ ] Test pipeline runs successfully
- [ ] Gradio app launches without errors
- [ ] Sample files prepared
- [ ] Browser bookmarks cleared
- [ ] Desktop cleaned up

### During Recording:
- [ ] Introduce the project clearly
- [ ] Show actual results, not just code
- [ ] Explain key metrics and their significance
- [ ] Demonstrate both interfaces
- [ ] Highlight unique features
- [ ] Address limitations honestly

### After Recording:
- [ ] Review for clarity and completeness
- [ ] Check audio quality
- [ ] Verify all key points covered
- [ ] Upload to appropriate platform
- [ ] Share access link

## 📊 Expected Results to Show

### Model Performance:
- Issue Type Accuracy: ~94%
- Urgency Level Accuracy: ~38% (explain why)
- Confusion matrices
- Classification reports

### Entity Extraction Examples:
- Products: SmartWatch V2, PhotoSnap Cam, etc.
- Order Numbers: #12345, #67890, etc.
- Dates: "15 March", "20 April", etc.
- Complaint Keywords: broken, defective, late, etc.

### Visualizations:
- Data distribution charts
- Model performance comparisons
- Sentiment analysis plots
- Feature importance (if time permits)

---

**Total Estimated Time: 5-7 minutes**
**Target Audience: Technical reviewers, potential users, stakeholders**
**Goal: Demonstrate comprehensive ML pipeline with practical value**
