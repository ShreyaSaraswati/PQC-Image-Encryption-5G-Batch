import os

# --- Simulated Kyber512 Post-Quantum KEM (for demo purposes) ---

KEY_SIZE = 32  # 256-bit key, equivalent to AES-256

def pqc_kem_keypair():
    """Generate Kyber-style keypair (public_key, secret_key)."""
    public_key = os.urandom(KEY_SIZE)
    secret_key = os.urandom(KEY_SIZE)
    return public_key, secret_key

def pqc_kem_encaps(public_key):
    """Encapsulate a shared secret using the public key."""
    shared_secret = os.urandom(KEY_SIZE)
    ciphertext = bytes([a ^ b for a, b in zip(public_key, shared_secret)])
    return ciphertext, shared_secret

def pqc_kem_decaps(ciphertext, secret_key):
    """Decapsulate shared secret using the secret key."""
    shared_secret = bytes([a ^ b for a, b in zip(ciphertext, secret_key)])
    return shared_secret
