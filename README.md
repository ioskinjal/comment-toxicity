# 🛡️ Comment Toxicity Detection

A simple ML + Streamlit app to detect toxic comments.  
Trains a model, predicts single comments, and supports CSV bulk predictions.

---

## ⚙️ Setup
```bash
git clone https://github.com/ioskinjal/comment-toxicity.git
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

	---

## 🔹 Step 3: Commit & Push
Now update GitHub with the new files:
```bash
git add .gitignore README.md
git commit -m "Added .gitignore and README.md"
git push origin main
