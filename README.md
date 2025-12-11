# 🎯 Propensity Score Matching: A Complete Tutorial

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Course](https://img.shields.io/badge/Course-INFO%207390-red.svg)](https://www.northeastern.edu/)

> **From Correlation to Causation in Observational Data**

A comprehensive educational package teaching Propensity Score Matching (PSM) for causal inference. Created for INFO 7390: Advanced Data Science and Architecture at Northeastern University.

---

## 📺 Video Tutorial

**🔗 Watch the Tutorial:** [https://youtu.be/4yfYS5VX7dE]

Duration: ~10 minutes | Structure: Explain → Show → Try

---

## 📚 Concept Overview

### What is Propensity Score Matching?

Propensity Score Matching (PSM) is a statistical technique used to estimate **causal effects from observational data**. When randomized experiments aren't possible, PSM helps reduce selection bias by matching treated and control units with similar probabilities of receiving treatment.

### The Problem PSM Solves
```
❌ Naive Comparison:
   "Patients who took the medication had better outcomes"
   But... were healthier patients more likely to take it?

✅ PSM Solution:
   Match patients with SAME probability of treatment
   Compare outcomes between matched pairs
   Remove selection bias from measured confounders
```

### Real-World Applications

| Domain | Example |
|--------|---------|
| Healthcare | Evaluating treatment effectiveness from EHR data |
| Marketing | Measuring campaign ROI without A/B testing |
| Policy | Assessing job training program impact |
| Technology | Estimating feature impact from observational logs |

---

## 🎯 Learning Objectives

By completing this tutorial, you will be able to:

✅ **Explain** when propensity score matching is appropriate and its limitations

✅ **Implement** a complete PSM pipeline in Python from scratch

✅ **Diagnose** covariate balance and overlap issues using SMD and Love plots

✅ **Interpret** treatment effect estimates (ATT, ATE) correctly

✅ **Apply** computational skepticism to causal claims from observational studies

---

## 📁 Repository Structure
```
psm-teaching-project/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── LICENSE                                # MIT License
├── notebooks/
│   ├── 01_psm_complete_tutorial.ipynb    # Main tutorial (68 cells)
│   └── 02_starter_template.ipynb         # Practice notebook with TODOs
└── data/
    └── synthetic_cardiac_rehab.csv       # Sample dataset
```

---

## 🚀 Installation Instructions

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab

### Step 1: Clone the Repository
```bash
git clone https://github.com/NikshipthNarayan/psm-teaching-project.git
cd psm-teaching-project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # On Mac/Linux
venv\Scripts\activate           # On Windows
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter Notebook
```bash
jupyter notebook
```

### Step 5: Open the Tutorial
Navigate to `notebooks/` and open `01_psm_complete_tutorial.ipynb`

---

## 💻 Usage Examples

### Quick Start: Run the Complete Tutorial
```python
# Open notebooks/01_psm_complete_tutorial.ipynb
# Run all cells to see PSM in action!
```

### Using the Code in Your Own Projects
```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load your data
df = pd.read_csv('data/synthetic_cardiac_rehab.csv')

# Define covariates
covariates = ['age', 'severity', 'prior_admits', 'insurance', 'motivation']

# Step 1: Estimate Propensity Scores
X = df[covariates]
y = df['treatment']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

df['propensity_score'] = model.predict_proba(X_scaled)[:, 1]

# Step 2: Perform Matching (see tutorial for full implementation)

# Step 3: Assess Balance (SMD < 0.1 is excellent)

# Step 4: Estimate ATT
# ATT = mean(outcome|treated, matched) - mean(outcome|control, matched)
```

---

## 📊 Sample Dataset

The `data/synthetic_cardiac_rehab.csv` file contains 2,000 synthetic patient records:

| Variable | Description | Type |
|----------|-------------|------|
| `patient_id` | Unique identifier | Integer |
| `age` | Patient age (45-85) | Float |
| `severity` | Disease severity (1-10) | Float |
| `prior_admits` | Previous hospitalizations (0-5) | Integer |
| `insurance` | Has premium insurance (0/1) | Binary |
| `motivation` | Health motivation score (1-10) | Float |
| `treatment` | Enrolled in cardiac rehab (0/1) | Binary |
| `readmitted` | Readmitted within 30 days (0/1) | Binary |

**Note:** This is synthetic data with a built-in true treatment effect of approximately -15 percentage points.

---

## 📖 Tutorial Contents

### Main Tutorial (`01_psm_complete_tutorial.ipynb`)

| Section | Topics |
|---------|--------|
| 1. Introduction | Why causal inference matters, selection bias |
| 2. Fundamental Problem | Potential outcomes, counterfactuals |
| 3. What is PSM? | Propensity score theorem |
| 4. Math Foundations | Logistic regression, SMD |
| 5. Key Assumptions | Unconfoundedness, positivity, SUTVA |
| 6. Implementation | Full PSM pipeline from scratch |
| 7. Diagnostics | Balance assessment, Love plots |
| 8. Treatment Effects | ATT estimation with confidence intervals |
| 9. Sensitivity Analysis | Robustness checks |
| 10-11. Exercises | Beginner, intermediate, advanced |
| 12. Debugging Guide | Common mistakes and solutions |
| 13. References | Papers, books, libraries |

### Starter Template (`02_starter_template.ipynb`)

Practice notebook with:
- ✅ Setup code provided
- ✅ TODO sections for you to complete
- ✅ Hints and expected outputs
- ✅ Reflection questions

---

## 🔗 References

### Key Papers
1. Rosenbaum & Rubin (1983). "The central role of the propensity score in observational studies for causal effects."
2. Austin (2011). "An Introduction to Propensity Score Methods."

### Recommended Books (Free Online)
- [Causal Inference: The Mixtape](https://mixtape.scunning.com/) - Scott Cunningham
- [The Effect](https://theeffectbook.net/) - Nick Huntington-Klein

### Python Libraries for Production
- `pymatch` - Full PSM pipeline
- `DoWhy` - Microsoft's causal inference library
- `causalinference` - Causal inference toolkit

---

## 👤 Author

**Nikshipth Narayan Bondugula**

- 🎓 M.S. Information Systems, Northeastern University
- 📚 Course: INFO 7390 - Advanced Data Science and Architecture
- 📅 Date: December 2025

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### AI Assistance Disclosure
This tutorial was developed with assistance from **Claude (Anthropic)** for:
- Code debugging and optimization
- Generating synthetic datasets
- Creating explanatory diagrams
- Proofreading and formatting

All pedagogical approaches, examples, and educational design represent original work by the author.

---

<p align="center">
  <b>⭐ If you found this helpful, please star the repository! ⭐</b>
</p>
```

---

## 📄 2. requirements.txt
```
# PSM Teaching Project - Python Dependencies
# Install with: pip install -r requirements.txt

numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
scipy>=1.7.0
jupyter>=1.0.0
notebook>=6.4.0
```

---

## 📄 3. LICENSE
```
MIT License

Copyright (c) 2025 Nikshipth Narayan Bondugula

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
