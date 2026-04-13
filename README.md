# ML Math Toolkit

A complete Streamlit-based mathematics and machine learning calculator for quick experimentation, concept learning, and interview prep.

This app combines multiple domains in one place:
- Basic arithmetic
- Statistics
- Probability and distributions
- Linear algebra
- Calculus
- ML loss and activation functions

It is designed for students, beginners in AI/ML, and anyone who wants an interactive way to compute and visualize core formulas.

## Features

### 1) Basic Arithmetic
- Addition, subtraction, multiplication, division
- Power and modulo operations

### 2) Statistics
- Mean, median, mode
- Variance, standard deviation
- Min, max, range
- Percentile, IQR
- Skewness, kurtosis
- Histogram visualization with mean line

### 3) Probability
- Basic probability and complement
- Union, intersection, conditional probability
- Bayes theorem
- Binomial, Poisson, exponential distributions
- Gaussian PDF and normal Z-score calculator
- Built-in visual plots for distributions

### 4) Linear Algebra (Matrix)
- Matrix addition, subtraction, multiplication
- Transpose, determinant, inverse
- Rank, trace
- L1, L2, and Frobenius norms
- Eigenvalues/eigenvectors
- SVD decomposition
- Dot product

### 5) Calculus
- First derivative
- Second derivative
- Partial derivative
- Gradient vector
- Definite integral
- Gradient descent simulation
- Straight line equation visualizer

### 6) ML Functions

Loss functions:
- MSE
- MAE
- RMSE
- Binary cross-entropy
- Categorical cross-entropy

Activation functions:
- Sigmoid
- ReLU
- Leaky ReLU
- Tanh
- Softmax
- ELU

## Tech Stack

- Python
- Streamlit
- NumPy
- SciPy
- SymPy
- Matplotlib

## Project Structure

```
ML-Math-Toolkit/
|- app.py
|- requirements.txt
|- README.md
|- LICENSE
```

## Installation and Run

1. Clone the repository:

```bash
git clone https://github.com/TasnimTaz/ML-Math-Toolkit.git
cd ML-Math-Toolkit
```

2. (Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the app:

```bash
streamlit run app.py
```

## Input Format Notes

- For list-based fields, use comma-separated values.
	- Example: `1, 2, 3, 4`
- For matrix input, use:
	- Rows separated by semicolons
	- Values separated by commas
	- Example: `1,2;3,4`

## Why This Project

Most learning tools separate mathematics topics into different notebooks or websites. This project brings major AI/ML mathematical operations into a single interactive app so users can:
- compute faster,
- validate formulas,
- and understand behavior visually.

## Roadmap

Possible future improvements:
- Add confusion matrix and classification metrics
- Add optimization visualizers (GD vs Adam)
- Add symbolic simplification and step-by-step solving
- Add export/download of generated results

## License

This project is available under the license included in this repository.
