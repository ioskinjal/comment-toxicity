# 🛡️ Comment Toxicity Detection

A simple ML + Streamlit app to detect toxic comments.  
Trains a model, predicts single comments, and supports CSV bulk predictions.

---

## ⚙️ Setup
```bash
git clone https://github.com/your-username/comment-toxicity.git
cd comment-toxicity
python3 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

📊 Train Model

python train_model.py

🖥️ Run App
streamlit run app.py

🚀 Tech
	•	Python, scikit-learn, TensorFlow/PyTorch
	•	NLP (TF-IDF / BERT)
	•	Streamlit

    