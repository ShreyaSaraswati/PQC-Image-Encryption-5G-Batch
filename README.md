#  Post-Quantum Image Encryption over Simulated 5G Network

This project demonstrates **secure image transmission** using **Post-Quantum Cryptography (PQC)** integrated with a **simulated 5G network** model.  
It encrypts images, simulates their transmission under 5G parameters (latency, noise, bandwidth), and decrypts them to evaluate quality using metrics such as **MSE**, **PSNR**, and **SSIM**.

---

##  Features
- **Post-Quantum Cryptography (Kyber)** for encryption & decryption  
- **Batch-wise image processing** (151 images tested)  
- **5G network simulation** to emulate bandwidth, latency, and noise conditions  
- **Image quality evaluation** using:
  - Mean Squared Error (MSE)
  - Peak Signal-to-Noise Ratio (PSNR)
  - Structural Similarity Index (SSIM)
- **Matplotlib visualizations** for original, encrypted, and decrypted comparisons  
- **Animated 5G transmission demo** (Tkinter-based)  
- **CSV logging** of results and summary metrics  

---

##  Tech Stack
- **Language:** Python 3.10  
- **Libraries Used:**  
  `numpy`, `matplotlib`, `tqdm`, `Pillow`, `cryptography`, `pycryptodome`  

---

##  Working
1. Each image is **encrypted** using Kyber PQC scheme.  
2. The **cipher data** is transmitted through a **simulated 5G channel** that adds noise, latency, and variable bandwidth.  
3. Upon reception, the image is **decrypted** and compared against the original.  
4. The script computes and logs **MSE**, **PSNR**, and **SSIM** metrics.  
5. A final summary and visualization compare transmission performance across the dataset.

---

##  Example Metrics
| Metric | Ideal Range   | Project Results |
|--------|---------------|-----------------|
| MSE    | 0 – 10        | ~5.5 – 8.2      |
| PSNR   | >40 dB        | 42 – 50 dB      |
| SSIM   | 0.90 – 1.00   | ~0.95 – 0.99    |

---

##  5G Simulation Parameters
| Parameter  | Description                        | Typical Value |
|------------|------------------------------------|---------------|
| Bandwidth  | Simulated 5G throughput            | 100 Mbps      |
| Latency    | Channel propagation delay          | 1–10 ms       |
| Noise      | Gaussian interference (randomized) | 0.01–0.05 dB  |

---

##  Running the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/PQC-Image-Encryption-5G-Batch.git
   cd PQC-Image-Encryption-5G-Batch
2. Create a virtual environment:
    python -m venv venv
    venv\Scripts\activate
3. Install dependencies:
    pip install -r requirements.txt
4. Run the main script:
    python main.py

---

## Team Members:
  Shreya Saraswati
  
  Bhoomika B V
  
  Deeksha P T
  
  Yashodha B T

---

## Acknowledgement
  This project is carried out as part of our **Major Project work under Visvesvaraya Technological University (VTU)** for the partial fulfillment of the requirements for the **award of the Bachelor’s Degree in Engineering (ECE), 2026 batch.**
