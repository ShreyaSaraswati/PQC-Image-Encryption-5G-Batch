import hashlib
import hmac
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad

def derive_keys(shared_secret: bytes) -> bytes:
    """Derive AES key from PQC shared secret"""
    return hashlib.sha256(shared_secret).digest()

def encrypt_bytes(key: bytes, data: bytes) -> bytes:
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(data, AES.block_size))
    return cipher.iv + ct_bytes

def decrypt_bytes(key: bytes, ciphertext: bytes) -> bytes:
    iv = ciphertext[:16]
    ct = ciphertext[16:]
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)

def compute_hmac(key: bytes, data: bytes) -> str:
    return hmac.new(key, data, hashlib.sha256).hexdigest()
