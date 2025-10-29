# kyber_module.py
"""
Kyber module with two modes:
- If pqcrypto is installed, use pqcrypto.kem.kyber512
- Otherwise fallback to a simple demo Kyber512 class (not secure) for testing.
This module exposes:
- generate_keypair() -> (public, private)
- encapsulate(public) -> (kem_ct, shared_secret)
- decapsulate(kem_ct, private) -> shared_secret
- save_key(path, bytes), load_key(path)
"""

import os

# Try pqcrypto first
try:
    from pqcrypto.kem.kyber512 import generate_keypair as _pq_generate, encrypt as _pq_encapsulate, decrypt as _pq_decapsulate
    HAS_PQ = True
except Exception:
    HAS_PQ = False

if not HAS_PQ:
    # Fallback demo implementation (DO NOT USE IN PRODUCTION)
    class Kyber512Demo:
        def __init__(self):
            self.keysize = 32

        def keygen(self):
            key = os.urandom(self.keysize)
            return key, key

        def encapsulate(self, public_key):
            shared = os.urandom(self.keysize)
            ct = bytes([a ^ b for a, b in zip(public_key, shared)])
            return ct, shared

        def decapsulate(self, ct, private_key):
            shared = bytes([a ^ b for a, b in zip(ct, private_key)])
            return shared

    _demo = Kyber512Demo()

def generate_keypair():
    """Return (public, private)"""
    if HAS_PQ:
        pub, priv = _pq_generate()
        return pub, priv
    else:
        return _demo.keygen()

def encapsulate(public_key):
    """Return (kem_ct, shared_secret)"""
    if HAS_PQ:
        ct, shared = _pq_encapsulate(public_key)
        return ct, shared
    else:
        return _demo.encapsulate(public_key)

def decapsulate(kem_ct, private_key):
    """Return shared_secret"""
    if HAS_PQ:
        return _pq_decapsulate(kem_ct, private_key)
    else:
        return _demo.decapsulate(kem_ct, private_key)

def save_key(path, b):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)

def load_key(path):
    with open(path, "rb") as f:
        return f.read()
