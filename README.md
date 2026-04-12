# 🎙️ Speech Analysis: Pause & Repetition Detection #
A Python-based system that analyzes speech audio files to detect:

🟢 Pause segments (silent regions in speech)    
🔁 Repetition patterns (e.g., "ba-ba-ball", "I-I-I want")

This project focuses on signal processing techniques (not machine learning) to understand speech patterns in an interpretable and lightweight way.

🚀 What it does  
.Detects silent regions in speech and returns their timestamps    
.Calculates total pause duration   
.Identifies repeated speech segments using audio features    
.Outputs a clean, readable analysis report    

⚙️ How it works    
🔹1. Audio Preprocessing    
 .Load audio using librosa   
 .Convert to mono and resample (16kHz)   
 .Normalize amplitude  
 .Optional noise reduction using STFT
🔹2. Feature Extraction  
  .RMS Energy → used for detecting silence   
  .MFCC (Mel-Frequency Cepstral Coefficients) → used for speech pattern analysis    
🔹3. Pause Detection  
  .Compute RMS energy across frames  
  .Frames below a threshold are marked as silent  
  .Consecutive silent frames → pause segments  
  .Convert frame indices to timestamps  
🔹4. Repetition Detection  
  .Extract MFCC features from sliding windows  
  .Compute cosine similarity between adjacent windows  
  .High similarity → potential repetition  
  .Group similar segments into repetition events  

💡 Why this approach  
This project uses signal processing techniques instead of machine learning to keep the system:  
.Simple  
.Interpretable  
.Lightweight  
.RMS energy is effective for detecting silence since pauses correspond to low signal energy  
.MFCC captures important speech characteristics  
.Cosine similarity helps identify repeated patterns in speech  

▶️ How to run   

1. Install dependencies 
pip install -r requirements.txt 
2. Generate a sample audio file 
python generate_sample.py 
3. Run the analysis 
python main.py sample_audio/sample.wav 
4. Optional arguments 
python main.py sample_audio/sample.wav --threshold 0.03 
python main.py sample_audio/sample.wav --sim 0.88 
python main.py sample_audio/sample.wav --no-noise-reduction 

📊 Example Output  
Step 2: Detecting pauses...  
Detected 11 pauses  
Step 3: Detecting repetitions...  
Detected 6 repetitions  
Pause Detection:  
[3.82s - 4.42s]   
[6.77s - 8.05s] ...  

Total pause time: 6.43s   
Repetition Detection:  
Repetitions found: 6  
At positions: 0.00s, 1.60s, 3.80s...  


📁 Project Structure 
speech_analysis/ 
│── main.py 
│── utils.py 
│── pause_detection.py 
│── repetition_detection.py 
│── generate_sample.py 
│── sample_audio/ 
│── requirements.txt 
│── README.md 

📦 Requirements 
.Python 3.9+ 
.librosa 
.numpy 
.scipy 
.soundfile 

Install all with: 

pip install -r requirements.txt 

⚠️ Limitations 
.Sensitive to background noise and recording quality 
.Repetition detection may not work well for very fast speech 
.Threshold values may require tuning for different datasets 

🔮 Future Improvements 
.Improve noise reduction techniques 
.Use machine learning models for better accuracy 
.Real-time speech analysis support 
.Visualization of pauses and repetitions 

