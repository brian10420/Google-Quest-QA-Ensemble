# Data Directory

This directory should contain the Google QUEST Challenge competition data files.

## Required Files

Place the following files in this directory:

- `train.csv` - Training data with question-answer pairs and labels
- `test.csv` - Test data for predictions
- `sample_submission.csv` - Submission format example

## Download Instructions

1. **Visit Kaggle Competition Page**
   - URL: https://www.kaggle.com/competitions/google-quest-challenge/data
   - You need to join the competition to access the data

2. **Download Data**
   ```bash
   # Using Kaggle API (recommended)
   kaggle competitions download -c google-quest-challenge
   
   # Unzip
   unzip google-quest-challenge.zip -d data/
   ```

3. **Verify Files**
   ```bash
   ls -lh data/
   # Should show:
   # - train.csv (~6MB)
   # - test.csv (~500KB)
   # - sample_submission.csv (~100KB)
   ```

## Dataset Statistics

- **Training samples**: 6,079 Q&A pairs
- **Test samples**: ~476 Q&A pairs
- **Features**:
  - `qa_id`: Unique identifier
  - `question_title`: Question headline
  - `question_body`: Full question text
  - `answer`: Answer text
  - `category`: Domain category (5 categories)
  - `host`: Source website
  
- **Targets**: 30 quality scores (0-1 range)
  - 21 question quality metrics
  - 9 answer quality metrics

## Data Privacy

⚠️ **Important**: Do not commit data files to GitHub
- Already excluded in `.gitignore`
- Data is copyrighted by Kaggle/Google

## Data Preprocessing

See training scripts in `../training/` for preprocessing examples:
- Text cleaning (code/math/URL normalization)
- Tokenization strategies for different models
- Category encoding

---

For questions about the data format, refer to the [competition overview](https://www.kaggle.com/competitions/google-quest-challenge/overview).
