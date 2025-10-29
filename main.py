"""
pqc_batch_pipeline.py

Batch-mode PQC image encryption simulation:
  1) Encrypt all images (store .bin + metadata)
  2) Transmit all encrypted files (copy)
  3) Decrypt all transmitted files, add mild noise, compute metrics
  4) Save CSV, print summary table + averages
  5) Visualize 5 random examples and a metrics chart

Requirements:
  - kyber_module with generate_keypair(), encapsulate(), decapsulate()
  - Pillow, numpy, pandas, matplotlib, tqdm, pycryptodome, scikit-image
"""

import os
import sys
import time
import shutil
import random
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from simulate_5g_transmission import FiveGNetworkSimulator
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="tkinter")

if sys.prefix == sys.base_prefix:
    print("  Virtual environment not detected. Please activate it with:")
    print("   venv\\Scripts\\activate (Windows)  or  source venv/bin/activate (Linux/macOS)")
    print("Continuing anyway...\n")

# Add project root for kyber_module import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from kyber_module import generate_keypair, encapsulate, decapsulate

# ----------------- CONFIG -----------------
STD_DEV_NOISE = 1.0           # post-decryption noise std dev (0 => near-lossless)
SAMPLE_TO_VISUALIZE = 5       # number of random images to display at end
RANDOM_SEED = 42              # reproducible random sampling
SAVE_METRICS_CSV = True
SAVE_METRICS_CSV_PATHNAME = "image_metrics.csv"  # saved inside decrypted folder
# ------------------------------------------

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------- PATHS -----------------
ROOT = os.path.dirname(__file__)
DATASET_DIR = os.path.join(ROOT, "dataset")
ENCRYPTED_DIR = os.path.join(ROOT, "encrypted")
TRANSMIT_DIR = os.path.join(ROOT, "transmitted")
DECRYPTED_DIR = os.path.join(ROOT, "decrypted")

os.makedirs(ENCRYPTED_DIR, exist_ok=True)
os.makedirs(TRANSMIT_DIR, exist_ok=True)
os.makedirs(DECRYPTED_DIR, exist_ok=True)

# ----------------- HELPERS -----------------
def add_gaussian_noise_uint8(image_array, std_dev=1.0):
    """Add Gaussian noise on 0-255 uint8 image, clip and return uint8."""
    noise = np.random.normal(0.0, std_dev, image_array.shape).astype(np.float32)
    out = np.clip(image_array.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out

def ciphertext_to_visual(cipher_bytes, shape):
    """Map ciphertext bytes to a uint8 array visual with same shape as image."""
    if cipher_bytes is None or len(cipher_bytes) == 0:
        return np.random.randint(0, 256, shape, dtype=np.uint8)
    arr = np.frombuffer(cipher_bytes, dtype=np.uint8)
    needed = int(np.prod(shape))
    if arr.size < needed:
        reps = int(np.ceil(needed / arr.size))
        arr = np.tile(arr, reps)
    arr = arr[:needed].reshape(shape)
    return arr

def plot_images_white(original, enc_vis, decrypted, mse, psnr, ssim, filename=None):
    """White-themed plotting (Original / Encrypted visual / Decrypted)."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "font.size": 10
    })
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"{filename or 'Image'} | MSE: {mse:.6f} | PSNR: {psnr:.2f} dB | SSIM: {ssim:.6f}",
        fontsize=12, fontweight="bold", color="black"
    )
    titles = ["Original", "Encrypted (visual)", "Decrypted"]
    imgs = [original, enc_vis, decrypted]
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=10, color="black")
        ax.axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

# ----------------- COLLECT FILES -----------------
image_files = sorted([f for f in os.listdir(DATASET_DIR)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])
num_images = len(image_files)
if num_images == 0:
    raise RuntimeError("No images found in dataset folder.")

print("\n=== 🔒 ENCRYPTION PHASE STARTED ===")
encryption_metadata = {}   # filename -> {"aes_key":..., "iv":..., "shape":..., "cipher_len":...}

# ----------------- 1) ENCRYPTION (BATCH) -----------------
for filename in tqdm(image_files, desc="Encrypting", ncols=80):
    img_path = os.path.join(DATASET_DIR, filename)
    pil = Image.open(img_path).convert("RGB")
    img_arr = np.array(pil, dtype=np.uint8)
    shape = img_arr.shape
    # Kyber keypair + shared secret -> AES key
    public_key, secret_key = generate_keypair()
    _, shared_secret = encapsulate(public_key)
    aes_key = shared_secret[:32] if len(shared_secret) >= 32 else shared_secret.ljust(32, b'\0')
    # AES-CBC encryption with random IV
    cipher_enc = AES.new(aes_key, AES.MODE_CBC)
    iv = cipher_enc.iv
    ciphertext = cipher_enc.encrypt(pad(img_arr.tobytes(), AES.block_size))
    # Save encrypted (iv + ciphertext)
    enc_path = os.path.join(ENCRYPTED_DIR, filename + ".bin")
    with open(enc_path, "wb") as f:
        f.write(iv + ciphertext)
    # store metadata
    encryption_metadata[filename] = {
        "aes_key": aes_key,
        "iv": iv,
        "shape": shape,
        "cipher_len": len(ciphertext)
    }

print("=== 🔒 ENCRYPTION PHASE COMPLETE ===")

# ----------------- 2) TRANSMISSION (5G NETWORK SIMULATION) -----------------

# Initialize 5G network simulator
network = FiveGNetworkSimulator()
#network.animate_live(interval_ms=300)
print("\n===  SIMULATED 5G TRANSMISSION PHASE STARTED ===")


# Launch live animation
#network.animate_live(interval_ms=300)

transmission_stats = []

for filename in tqdm(image_files, desc="Transmitting over 5G", ncols=90):
    src = os.path.join(ENCRYPTED_DIR, filename + ".bin")
    dst = os.path.join(TRANSMIT_DIR, filename + ".bin")

    with open(src, "rb") as f:
        data = f.read()

    stats = network.transmit(data, filename)
    transmission_stats.append(stats)

    with open(dst, "wb") as f:
        f.write(data)

print("\n=== ✅ SECURE TRANSMISSION OVER 5G NETWORK COMPLETE ===")

# --- Transmission Summary ---
if transmission_stats:
    avg_delay = np.mean([s["delay_s"] for s in transmission_stats])
    loss_rate = np.mean([1 if s["lost"] else 0 for s in transmission_stats]) * 100
    total_data_mb = np.sum([s["data_size_kb"] for s in transmission_stats]) / 1024
    avg_signal = np.mean([s["signal_quality_db"] for s in transmission_stats]) 

    print("\n📊 5G Transmission Summary:")
    print(f"• Total Images Transmitted : {len(transmission_stats)}")
    print(f"• Total Data Sent          : {total_data_mb:.2f} MB")
    print(f"• Average Delay per Image  : {avg_delay:.4f} s")
    print(f"• Average Signal Quality   : {avg_signal:.2f} dB")
    print(f"• Packet Loss Rate         : {loss_rate:.2f} %")
    print(f"• Effective Throughput     : {(total_data_mb / (avg_delay * len(transmission_stats))):.2f} MB/s")

    print("\n⚡ Transmission Simulation Done. Proceeding to Decryption & Metrics Phase...\n")
else:
    print("⚠️ No transmission stats available. Skipping summary.")

avg_delay = np.mean([s["delay_s"] for s in transmission_stats])
avg_signal = np.mean([s["signal_quality_db"] for s in transmission_stats])
print(f"\n📊 5G Network Summary:")
print(f"   • Average Transmission Delay: {avg_delay:.6f} s")
print(f"   • Average Signal Quality: {avg_signal:.2f} dB")
print(f"   • Total Files Transmitted: {len(transmission_stats)}")

# Keep animation window open until user closes
plt.show()


# ----------------- 3) DECRYPTION + METRICS (BATCH) -----------------
print("\n=== 🔓 DECRYPTION & METRICS PHASE STARTED ===")
metrics_list = []   # rows: [filename, mse, psnr, ssim]
for filename in tqdm(image_files, desc="Decrypting", ncols=80):
    meta = encryption_metadata[filename]
    aes_key = meta["aes_key"]
    # read transmitted file
    with open(os.path.join(TRANSMIT_DIR, filename + ".bin"), "rb") as f:
        iv_read = f.read(16)
        ciphertext_read = f.read()
    # decrypt
    cipher_dec = AES.new(aes_key, AES.MODE_CBC, iv=iv_read)
    try:
        decrypted_bytes = unpad(cipher_dec.decrypt(ciphertext_read), AES.block_size)
    except ValueError:
        # fallback: if unpad fails, try truncation (shouldn't normally happen)
        decrypted_bytes = cipher_dec.decrypt(ciphertext_read)[: np.prod(meta["shape"])]
    # reshape
    decrypted_arr = np.frombuffer(decrypted_bytes[: np.prod(meta["shape"])], dtype=np.uint8).reshape(meta["shape"])
    # add mild realistic noise (post-decryption)
    decrypted_noisy = add_gaussian_noise_uint8(decrypted_arr, std_dev=STD_DEV_NOISE)
    # save decrypted image
    dec_save_path = os.path.join(DECRYPTED_DIR, filename)
    Image.fromarray(decrypted_noisy).save(dec_save_path)
    # compute metrics vs original (use original loaded from dataset for exact comparison)
    orig_arr = np.array(Image.open(os.path.join(DATASET_DIR, filename)).convert("RGB"), dtype=np.uint8)
    mse_val = mean_squared_error(orig_arr, decrypted_noisy)
    psnr_val = peak_signal_noise_ratio(orig_arr, decrypted_noisy, data_range=255)
    ssim_val = structural_similarity(orig_arr, decrypted_noisy, channel_axis=2)
    metrics_list.append([filename, float(mse_val), float(psnr_val), float(ssim_val)])

print("=== 🔓 DECRYPTION & METRICS PHASE COMPLETE ===")

# ----------------- SAVE METRICS CSV -----------------
df = pd.DataFrame(metrics_list, columns=["Filename", "MSE", "PSNR", "SSIM"])
csv_path = os.path.join(DECRYPTED_DIR, SAVE_METRICS_CSV_PATHNAME)
if SAVE_METRICS_CSV:
    df.to_csv(csv_path, index=False)
    print(f"\n✅ Metrics CSV saved to: {csv_path}")

# ----------------- PRINT SUMMARY TABLE + AVERAGES -----------------
print("\n=== 🧾 SUMMARY METRICS ===")
pd.set_option("display.max_rows", 10)   # show first 10 rows by default in console
print(df.head(20).to_string(index=False))  # print first 20 rows for quick check (adjust as needed)

avg_mse = df["MSE"].mean()
avg_psnr = df["PSNR"].mean()
avg_ssim = df["SSIM"].mean()

print("\n=== AVERAGE METRICS (BATCH) ===")
print(f"Processed {len(df)} images.")
print(f"Average MSE : {avg_mse:.6f}")
print(f"Average PSNR: {avg_psnr:.4f} dB")
print(f"Average SSIM: {avg_ssim:.6f}")

# ----------------- VISUALIZE SAMPLE IMAGES -----------------
print(f"\n=== 🎨 Visualizing {SAMPLE_TO_VISUALIZE} random samples ===")
sample_files = random.sample(image_files, min(SAMPLE_TO_VISUALIZE, len(image_files)))
for filename in sample_files:
    orig = np.array(Image.open(os.path.join(DATASET_DIR, filename)).convert("RGB"), dtype=np.uint8)
    dec = np.array(Image.open(os.path.join(DECRYPTED_DIR, filename)).convert("RGB"), dtype=np.uint8)
    # build encrypted visual from encrypted file
    with open(os.path.join(ENCRYPTED_DIR, filename + ".bin"), "rb") as f:
        _iv = f.read(16)
        cbytes = f.read()
    enc_vis = ciphertext_to_visual(cbytes, orig.shape)
    # pick stored metric row
    row = df[df["Filename"] == filename].iloc[0]
    plot_images_white(orig, enc_vis, dec, row["MSE"], row["PSNR"], row["SSIM"], filename=filename)

# ----------------- METRICS CHART (ALL IMAGES) -----------------
print("\n=== 📈 Generating metrics chart for all images ===")
plt.figure(figsize=(12, 6))
x = np.arange(len(df))
plt.subplot(3, 1, 1)
plt.plot(x, df["MSE"], marker='o')
plt.title("MSE per image")
plt.ylabel("MSE")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x, df["PSNR"], marker='o', color='orange')
plt.title("PSNR per image (dB)")
plt.ylabel("PSNR (dB)")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, df["SSIM"], marker='o', color='green')
plt.title("SSIM per image")
plt.ylabel("SSIM")
plt.xlabel("Image index")
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n=== 🎉 BATCH PROCESS COMPLETE ===")

# ----------------- ANIMATED DEMONSTRATION -----------------
print("\n===  Generating animated PQC-5G transmission demo ===")

from matplotlib.animation import FuncAnimation

# Convert df (metrics table) into list of dicts for easy access
results = df.to_dict(orient="records")

# Pick a random image from processed results
sample_idx = random.randint(0, len(results) - 1)
sample = results[sample_idx]
filename = sample["Filename"]

# Read the corresponding images
orig = plt.imread(os.path.join(DATASET_DIR, filename))
dec = plt.imread(os.path.join(DECRYPTED_DIR, filename))

# Create a pseudo-encrypted visual from ciphertext for animation
with open(os.path.join(ENCRYPTED_DIR, filename + ".bin"), "rb") as f:
    _iv = f.read(16)
    cbytes = f.read()
enc_vis = ciphertext_to_visual(cbytes, orig.shape)

mse_val = sample["MSE"]
psnr_val = sample["PSNR"]
ssim_val = sample["SSIM"]

# --- Create animation ---
fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
ax.axis("off")

# Text for status and metrics
status_text = ax.text(0.5, -0.1, "", ha="center", va="center", transform=ax.transAxes, fontsize=12, color="black")
metrics_text = ax.text(0.5, -0.2, "", ha="center", va="center", transform=ax.transAxes, fontsize=10, color="black")

# Initialize image display
img_disp = ax.imshow(orig)
ax.set_title(f"PQC over 5G Demo: {filename}", fontsize=13, color="black")

# Helper: create blended transition frames
def transition_frames(img1, img2, steps=15):
    frames = []
    for i in range(steps):
        alpha = i / (steps - 1)
        blended = np.clip((1 - alpha) * img1 + alpha * img2, 0, 1)
        frames.append(blended)
    return frames

# Build animation sequences (normalize to 0-1 for blending)
frames_encrypt = transition_frames(orig / 255.0, enc_vis / 255.0)
frames_transmit = transition_frames(enc_vis / 255.0, enc_vis / 255.0)
frames_decrypt = transition_frames(enc_vis / 255.0, dec / 255.0)
frames_all = frames_encrypt + frames_transmit + frames_decrypt

# Update function for animation frames
def update(frame):
    if frame < len(frames_encrypt):
        img_disp.set_data(frames_encrypt[frame])
        status_text.set_text("🔒 Encrypting with PQC (Kyber)...")
        metrics_text.set_text("")
    elif frame < len(frames_encrypt) + len(frames_transmit):
        img_disp.set_data(frames_transmit[frame - len(frames_encrypt)])
        status_text.set_text("📡 Transmitting securely over 5G network...")
        metrics_text.set_text("")
    else:
        img_disp.set_data(frames_decrypt[frame - len(frames_encrypt) - len(frames_transmit)])
        status_text.set_text("🔓 Decrypting at Receiver Side...")
        metrics_text.set_text(f"MSE: {mse_val:.4f} | PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    return img_disp, status_text, metrics_text

# Run animation
ani = FuncAnimation(fig, update, frames=len(frames_all), interval=150, blit=False, repeat=False)

plt.tight_layout()
plt.show()
