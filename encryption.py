
import os
from pip._internal.utils import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend


def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a symmetric encryption key from a password."""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())


def encrypt_file(input_file: str, output_file: str, password: str):
    """Encrypt the model file using AES encryption."""
    salt = os.urandom(16)
    key = derive_key(password, salt)
    with open(input_file, "rb") as f:
        data = f.read()

    cipher = Cipher(algorithms.AES(key), modes.GCM(salt), backend=default_backend())
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(data) + encryptor.finalize()

    with open(output_file, "wb") as f:
        f.write(salt + encryptor.tag + encrypted_data)

def decrypt_file(input_file: str, output_file: str, password: str):
    """Decrypt the model file using AES decryption."""
    with open(input_file, "rb") as f:
        data = f.read()

    salt = data[:16]
    tag = data[16:32]
    encrypted_data = data[32:]

    key = derive_key(password, salt)
    cipher = Cipher(algorithms.AES(key), modes.GCM(salt, tag), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

    with open(output_file, "wb") as f:
        f.write(decrypted_data)