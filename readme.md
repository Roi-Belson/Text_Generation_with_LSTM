# Text_Generation_with_LSTM

## Overview  
This repository implements a text​‑generation project using an LSTM (Long Short​‑Term Memory) neural network. The goal is to train a model to generate coherent text sequences by learning patterns from a corpus of text data.

## Contents  
- `Text_Generation_with_LSTM.ipynb` – A Jupyter notebook that includes the whole workflow: data loading → preprocessing → model definition (LSTM) → training → text generation.  
- `dataset.txt` – The raw text file used as the training corpus (you may replace this with your own).  
- `model_weights.h5` – Saved weights of the trained LSTM model (optional).  
- `generate_text.py` – A simple Python script to generate new text using the trained model.  
- `.gitattributes` – (Optional) track large files if you use them (e.g., large datasets or models).  

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

## Usage

### Training the Model
1. Open `Text_Generation_with_LSTM.ipynb`.  
2. Load your dataset or use the provided `dataset.txt`.  
3. Run the notebook cells sequentially to preprocess the text, create sequences, build the LSTM model, and train it.  
4. Save the trained model weights (`model.save('model_weights.h5')`) if needed.

### Generating Text
Use the `generate_text.py` script to generate new text:
```bash
python generate_text.py
```
Adjust the following parameters in the script:  
- Seed text – starting phrase for generation  
- Sequence length – input length for the model  
- Temperature/diversity – controls randomness of generated text  
- Number of characters/words – length of generated output  

## Contributing
Contributions are welcome! You can:  
- Experiment with different LSTM configurations  
- Try GRU or Transformer architectures  
- Use larger or more diverse datasets  
- Add evaluation metrics (perplexity, BLEU, etc.)

## License
This project is licensed under the MIT License. See `LICENSE` for details.