# 🎫 Customer Support Ticket Classification System

A comprehensive machine learning pipeline for automatically classifying customer support tickets by issue type and urgency level, with intelligent entity extraction capabilities.

## 🎯 Project Overview

This system addresses the challenge of efficiently categorizing and prioritizing customer support tickets using traditional NLP and machine learning techniques. It provides both a command-line interface and an interactive web application for real-time predictions.

## ✨ Features

### 🔧 Core Functionality
- **Multi-Task Classification**: Simultaneous prediction of issue type and urgency level
- **Entity Extraction**: Automatic identification of products, dates, order numbers, and complaint keywords
- **Confidence Scoring**: Prediction confidence levels for better decision-making
- **Batch Processing**: Handle multiple tickets efficiently

### 🤖 Machine Learning Pipeline
- **Text Preprocessing**: Comprehensive cleaning, tokenization, and lemmatization
- **Feature Engineering**: TF-IDF vectorization + additional features (sentiment, text length, etc.)
- **Model Training**: Random Forest classifiers with cross-validation
- **Performance Evaluation**: Detailed metrics and confusion matrices

### 🌐 Interactive Interface
- **Gradio Web App**: User-friendly interface for real-time predictions
- **Single Ticket Mode**: Individual ticket classification
- **Batch Upload**: CSV file processing for multiple tickets
- **Visualization Dashboard**: Data exploration and model performance charts

## 📊 Dataset Information

- **Size**: 1000 customer support tickets
- **Issue Types**: 7 categories (Billing Problem, Product Defect, Installation Issue, etc.)
- **Urgency Levels**: 3 levels (Low, Medium, High)
- **Products**: 10 different products (SmartWatch V2, PhotoSnap Cam, etc.)
- **Features**: ticket_id, ticket_text, issue_type, urgency_level, product

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project files**
   ```bash
   # Ensure you have these files in your directory:
   # - ticket_classifier.py
   # - gradio_app.py
   # - tickets_dataset.csv
   # - requirements.txt
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if not already downloaded)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

### Running the Application

#### Option 1: Command Line Interface
```bash
python ticket_classifier.py
```
This will:
- Load and preprocess the dataset
- Train both classification models
- Generate performance metrics and visualizations
- Run demo predictions on sample tickets

#### Option 2: Web Interface (Recommended)
```bash
python gradio_app.py
```
Then open your browser and navigate to the provided URL (typically `http://localhost:7860`)

## 🎮 Using the Web Interface

### 1. Model Training
- Navigate to the "🤖 Model Training" tab
- Click "🚀 Train Models" to train the classifiers
- Wait for training completion (shows accuracy metrics)

### 2. Single Ticket Prediction
- Go to "🎯 Single Ticket Prediction" tab
- Enter a ticket description in the text area
- Click "🔮 Predict" to get results
- View issue type, urgency level, and extracted entities

### 3. Batch Processing
- Switch to "📊 Batch Processing" tab
- Upload a CSV file with a 'ticket_text' column
- Click "📈 Process Batch" to process all tickets
- Download results as needed

## 📈 Model Performance

The system achieves strong performance across both classification tasks:

### Issue Type Classification
- **Algorithm**: Random Forest with 100 estimators
- **Features**: TF-IDF (5000 features) + sentiment + text statistics
- **Evaluation**: Cross-validation with stratified splits

### Urgency Level Classification
- **Algorithm**: Random Forest with balanced class weights
- **Features**: Same feature set as issue classification
- **Handling**: Addresses class imbalance in urgency levels

### Entity Extraction
- **Products**: Rule-based matching against known product list
- **Dates**: Regex patterns for multiple date formats
- **Order Numbers**: Pattern matching for order IDs (#XXXXX)
- **Complaint Keywords**: Predefined keyword dictionary

## 🔧 Technical Architecture

### Data Preprocessing Pipeline
1. **Text Cleaning**: Lowercase conversion, special character removal
2. **Tokenization**: Word-level tokenization using NLTK
3. **Stopword Removal**: English stopwords filtering
4. **Lemmatization**: Word normalization using WordNet
5. **Feature Extraction**: TF-IDF + additional engineered features

### Feature Engineering
- **TF-IDF Vectors**: Unigrams and bigrams (max 5000 features)
- **Text Statistics**: Length, word count
- **Sentiment Analysis**: Polarity and subjectivity scores
- **Urgency Indicators**: Presence of urgent keywords

### Model Architecture
```
Input Text → Preprocessing → Feature Engineering → Classification Models
                                                 ├── Issue Type Classifier
                                                 └── Urgency Level Classifier

Input Text → Entity Extraction → Structured Output
```

## 📁 Project Structure

```
├── ticket_classifier.py              # Main ML pipeline and training
├── gradio_app.py                     # Web interface application
├── test_pipeline.py                  # Comprehensive testing suite
├── tickets_dataset.csv               # Training dataset (1000 tickets)
├── sample_tickets.csv                # Sample data for batch testing
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Git ignore rules
├── README.md                         # This documentation
├── DEMO_SCRIPT.md                    # Video recording guide
├── PROJECT_SUMMARY.md                # Technical project summary
└── Generated Visualizations:
    ├── ticket_analysis_dashboard.png
    ├── issue_type_confusion_matrix.png
    └── urgency_confusion_matrix.png
```

## 🎯 Key Design Choices

### 1. **Traditional ML Approach**
- Used Random Forest for interpretability and robustness
- TF-IDF for proven text representation effectiveness
- Rule-based entity extraction for precision

### 2. **Feature Engineering Strategy**
- Combined text features with metadata (length, sentiment)
- Balanced approach between complexity and performance
- Engineered urgency indicators for better priority detection

### 3. **Handling Missing Data**
- Graceful handling of missing text and labels
- Separate training for each classification task
- Robust prediction pipeline with error handling

### 4. **User Experience Focus**
- Interactive web interface for non-technical users
- Batch processing for operational efficiency
- Clear confidence scores for decision support

## 🔍 Model Evaluation

### Metrics Used
- **Accuracy**: Overall classification performance
- **Precision/Recall/F1**: Per-class performance metrics
- **Confusion Matrix**: Detailed error analysis
- **Cross-Validation**: Robust performance estimation

### Visualization Outputs
- Data distribution charts
- Model performance comparisons
- Confusion matrices for error analysis
- Sentiment analysis scatter plots

## 🚧 Limitations & Future Improvements

### Current Limitations
- Rule-based entity extraction may miss complex patterns
- Limited to predefined product and keyword lists
- Performance depends on training data quality
- No real-time model updates

### Potential Enhancements
- **Deep Learning**: Implement BERT/RoBERTa for better text understanding
- **Named Entity Recognition**: Use spaCy or similar for advanced entity extraction
- **Active Learning**: Incorporate user feedback for model improvement
- **Multi-language Support**: Extend to non-English tickets
- **Real-time Training**: Implement online learning capabilities

## 🤝 Contributing

This project was developed as part of an AI development assignment. For improvements or modifications:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is developed for educational and demonstration purposes.

## 🙋‍♂️ Support

For questions or issues:
- Check the troubleshooting section below
- Review the code documentation
- Test with the provided example tickets

## 🔧 Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('all')
   ```

2. **Memory Issues with Large Datasets**
   - Reduce TF-IDF max_features parameter
   - Process data in smaller batches

3. **Gradio Interface Not Loading**
   - Check if port 7860 is available
   - Try different port: `interface.launch(server_port=8080)`

4. **Poor Prediction Performance**
   - Ensure model is trained before predictions
   - Check input text quality and length
   - Verify dataset integrity

---

**Built with ❤️ using Python, scikit-learn, NLTK, and Gradio**
**Made by Batchu Gnana Sampath 😉**
