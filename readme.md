# Text_Generation_with_LSTM

## Overview  
This repository implements a text​‑generation project using an LSTM (Long Short​‑Term Memory) neural network. The goal is to train a model to generate coherent text sequences by learning patterns from the book ‘Alice in Wonderland’ written by Lewis Carroll, which was published in the public domain under the Gutenberg Project.

## Contents  
- `Text_Generation_with_LSTM.ipynb` – A Jupyter notebook that includes the whole workflow: data loading → preprocessing → model definition (LSTM) → training → text generation.  
- `Alice_in_Wonderland.txt` – The raw text file used as the training corpus.
- `model_weights.h5` – Saved weights of the trained LSTM model (optional).  
- ‘My_Alice_Tokenizer’ - Tokenizer prepared with keras, which assigns each token an index based on its frequency in the book.

## Getting Started

### Prerequisites  
Make sure you have:  
- Python 3.7+ installed  
- Jupyter Notebook or JupyterLab  
- Required Python libraries installed (for example: `numpy`, `pandas`, `tensorflow`, `keras`, etc.)  

### Installation  
1. Clone this repository:  
   ```bash
   git clone https://github.com/Roi-Belson/Text_Generation_with_LSTM.git
   cd Text_Generation_with_LSTM
   ```  
2. Install dependencies (you can create a virtual environment first if desired):  
   ```bash
   pip install -r requirements.txt
   ```

## Datasets  
- **Alice in Wonderland**: The text of Alice in Wonderland was downloaded from the Gutenberg Project. It can be found and downloaded at: https://www.gutenberg.org/ebooks/11

