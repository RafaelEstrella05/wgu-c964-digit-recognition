import os
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend

def derive_key(password: str, salt: bytes) -> bytes:
    """Derive a symmetric encryption key from a password and salt.

    Args:
        password (str): The password to derive the key from.
        salt (bytes): A randomly generated salt for key derivation.

    Returns:
        bytes: The derived encryption key.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

def encrypt_file(input_file: str, output_file: str, password: str):
    """Encrypt a file using AES-GCM encryption.

    Args:
        input_file (str): Path to the plaintext input file.
        output_file (str): Path where the encrypted file will be saved.
        password (str): The password for encryption.
    """
    try:
        # Generate a random salt for key derivation
        salt = os.urandom(16)
        key = derive_key(password, salt)

        # Read the plaintext data from the input file
        with open(input_file, "rb") as f:
            data = f.read()

        # Set up AES-GCM encryption
        cipher = Cipher(algorithms.AES(key), modes.GCM(salt), backend=default_backend())
        encryptor = cipher.encryptor()
        encrypted_data = encryptor.update(data) + encryptor.finalize()

        # Write the encrypted file with salt and tag prepended
        with open(output_file, "wb") as f:
            f.write(salt + encryptor.tag + encrypted_data)
    except Exception as e:
        raise RuntimeError(f"Failed to encrypt file: {e}")

def decrypt_file(input_file: str, output_file: str, password: str):
    """Decrypt an AES-GCM encrypted file.

    Args:
        input_file (str): Path to the encrypted input file.
        output_file (str): Path where the decrypted file will be saved.
        password (str): The password for decryption.
    """
    try:
        # Read the encrypted file
        with open(input_file, "rb") as f:
            data = f.read()

        # Extract salt, tag, and encrypted data
        salt = data[:16]
        tag = data[16:32]
        encrypted_data = data[32:]

        # Derive the encryption key from the password and salt
        key = derive_key(password, salt)

        # Set up AES-GCM decryption
        cipher = Cipher(algorithms.AES(key), modes.GCM(salt, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()

        # Write the decrypted file
        with open(output_file, "wb") as f:
            f.write(decrypted_data)
    except Exception as e:
        raise RuntimeError(f"Failed to decrypt file: {e}")

if __name__ == "__main__":
    print("This module provides file encryption and decryption utilities.")
