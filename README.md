# Text Classification with SVM and Naive Bayes
Comparing the performance of Naive Bayes and Support Vector Machine for multidimensional text classification
## Overview
This repository contains code and resources for a text classification project using Support Vector Machine (SVM) and Naive Bayes algorithms. The goal of this project is to accurately classify text data into predefined categories, demonstrating the effectiveness of these machine learning models in natural language processing tasks.

## Features
- Implementation of SVM and Naive Bayes algorithms for text classification.
- Preprocessing pipelines for text data.
- Evaluation metrics to assess model performance.
- Sample datasets for training and testing.

## Requirements
Python 3.x
scikit-learn
pandas
NLTK or similar NLP libraries (for text preprocessing)

## Installation
#### Clone the repository to your local machine:
git clone https://github.com/wisdomobinna/textClassification.git
cd textClassification
#### Install required Python packages:
pip install -r requirements.txt

## Usage
To run the text classification using SVM or Naive Bayes, execute the following command:
python sentiment_analysis.py --model [svm|naive_bayes]

## Data
- text_classification dataset: This dataset is used for research or training of natural language processing (NLP) models. The dataset may include various types of conversations such as casual or formal discussions, interviews, customer service interactions, or social media conversations (3k sentences).
- chat_dataset

## Scripts and Notebooks
- textClassification.py: Main script for training and evaluating models.
- preprocessing.py: Contains text preprocessing functions.
- evaluation.py: Script for evaluating model performance.
- Jupyter notebooks with exploratory data analysis and model training walkthroughs.

## Models
- SVM: Implemented using scikit-learn's SVC class.
- Naive Bayes: Implemented using scikit-learn's MultinomialNB.

## Results
The performance of the text classification models - Naive Bayes and Support Vector Machine (SVC) - was evaluated on a test dataset. The key metrics considered were accuracy, precision, recall, and F1-score for each sentiment class (-1, 0, 1).
#### Naive Bayes Performance

- **Accuracy**: 77.78%

| Sentiment | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| -1        | 0.70      | 0.96   | 0.81     | 54      |
| 0         | 1.00      | 0.37   | 0.54     | 30      |
| 1         | 0.88      | 0.85   | 0.86     | 33      |
| **Overall** | **0.83** | **0.78** | **0.76** | **117** |

- Observations: The model showed high recall for negative sentiment (-1) but lower recall for neutral sentiment (0). This indicates a strong ability to identify negative sentiments but a potential overfitting to this class.

#### SVM (Support Vector Machine) Performance

- **Accuracy**: 86.32%

| Sentiment | Precision | Recall | F1-score | Support |
|-----------|-----------|--------|----------|---------|
| -1        | 0.89      | 0.94   | 0.92     | 54      |
| 0         | 0.91      | 0.67   | 0.77     | 30      |
| 1         | 0.79      | 0.91   | 0.85     | 33      |
| **Overall** | **0.87** | **0.86** | **0.86** | **117** |

- Observations: The SVM model outperformed the Naive Bayes in overall accuracy and balanced performance across all classes. It was particularly effective in identifying negative and positive sentiments with high precision and recall.

Overall, the SVM model demonstrated superior performance in terms of accuracy and balanced metrics across different sentiment classes compared to the Naive Bayes model. However, the choice between these models may depend on specific use cases and requirements, such as interpretability and computational efficiency

### Contributing
Contributions to this project are welcome. Please read the CONTRIBUTING.md file for guidelines on how to contribute.

License
This project is licensed under the MIT License - see the LICENSE.md file for details.

### Acknowledgments
- Kaggle: For providing a rich variety of datasets crucial for training and testing our models.
- Python Community: Immense gratitude for developing Python, which made this project possible with its powerful and versatile libraries.
- Jupyter Project: For their invaluable Jupyter Notebook tool, facilitating an interactive environment for coding and analysis.
- 
### Contact
For any queries regarding this project, please contact Wisdom Obinna at wisdom.k.obinna@gmail.com.

