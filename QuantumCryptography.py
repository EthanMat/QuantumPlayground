"""Quantum key distribution (QKD) BB84 simulation and secure string transmission.

This module simulates the basic BB84 protocol to establish a shared secret key
between Alice and Bob, then uses a one-time pad (OTP) to encrypt and decrypt
a text string.

Because this is a classical simulation, it models qubits as bit/basis choices,
and it can also simulate an eavesdropper (Eve) to show how interceptions
introduce errors.
"""

import random
import tkinter as tk
from tkinter import messagebox, scrolledtext
from typing import List, Tuple


def generate_random_bits(n: int) -> List[int]:
    return [random.randint(0, 1) for _ in range(n)]


def generate_random_bases(n: int) -> List[int]:
    # 0 = rectilinear basis (+), 1 = diagonal basis (x)
    return [random.randint(0, 1) for _ in range(n)]


def basis_symbol(basis: int) -> str:
    return '+' if basis == 0 else 'x'


def visualize_bb84(alice_bits: List[int], alice_bases: List[int], bob_bases: List[int], bob_results: List[int], show_bits: int = 64) -> None:
    display_bits = min(show_bits, len(alice_bits))
    alice_bits_line = ''.join(str(bit) for bit in alice_bits[:display_bits])
    alice_bases_line = ''.join(basis_symbol(b) for b in alice_bases[:display_bits])
    bob_bases_line = ''.join(basis_symbol(b) for b in bob_bases[:display_bits])
    bob_results_line = ''.join(str(bit) for bit in bob_results[:display_bits])
    matched_line = ''.join('|' if a == b else ' ' for a, b in zip(alice_bases[:display_bits], bob_bases[:display_bits]))

    print("--- BB84 transmission preview ---")
    print(f"Alice bits : {alice_bits_line}")
    print(f"Alice bases: {alice_bases_line}")
    print(f"Bob bases  : {bob_bases_line}")
    print(f"Bob meas.  : {bob_results_line}")
    print(f"Match      : {matched_line}")
    if len(alice_bits) > display_bits:
        print(f"... ({len(alice_bits) - display_bits} more bits omitted)")
    print("" )


def format_bb84_preview(alice_bits: List[int], alice_bases: List[int], bob_bases: List[int], bob_results: List[int], show_bits: int = 64) -> str:
    display_bits = min(show_bits, len(alice_bits))
    alice_bits_line = ''.join(str(bit) for bit in alice_bits[:display_bits])
    alice_bases_line = ''.join(basis_symbol(b) for b in alice_bases[:display_bits])
    bob_bases_line = ''.join(basis_symbol(b) for b in bob_bases[:display_bits])
    bob_results_line = ''.join(str(bit) for bit in bob_results[:display_bits])
    matched_line = ''.join('|' if a == b else ' ' for a, b in zip(alice_bases[:display_bits], bob_bases[:display_bits]))

    lines = [
        "--- BB84 transmission preview ---",
        f"Alice bits : {alice_bits_line}",
        f"Alice bases: {alice_bases_line}",
        f"Bob bases  : {bob_bases_line}",
        f"Bob meas.  : {bob_results_line}",
        f"Match      : {matched_line}",
    ]
    if len(alice_bits) > display_bits:
        lines.append(f"... ({len(alice_bits) - display_bits} more bits omitted)")
    return '\n'.join(lines)


def simulate_qkd_string_transfer_data(message: str, key_length: int = 256, sample_size: int = 20, eavesdrop: bool = False) -> str:
    alice_key, bob_key, success, alice_bits, alice_bases, bob_bases, bob_results = bb84_key_exchange(
        key_length, sample_size=sample_size, eavesdrop=eavesdrop)

    scenario = "with eavesdropper" if eavesdrop else "without eavesdropper"
    lines = [f"=== BB84 key exchange {scenario} ==="]
    lines.append(format_bb84_preview(alice_bits, alice_bases, bob_bases, bob_results, show_bits=48))
    lines.append(f"BB84 key exchange successful: {success}")
    lines.append(f"Sifted key length after sample removal: {len(alice_key)}")

    if not success or len(alice_key) == 0:
        lines.append("Cannot use the shared key. Eavesdropping detected or key length insufficient.")
        return '\n'.join(lines)

    required_bits = len(string_to_bits(message))
    if len(alice_key) < required_bits:
        lines.append("Shared key is too short for the message. Increase key_length.")
        return '\n'.join(lines)

    lines.append("\n--- One-Time Pad encryption ---")
    lines.append(f"Message: {message}")
    lines.append(f"OTP key bits used: {''.join(str(bit) for bit in alice_key[:required_bits])}")

    cipher_bits = otp_encrypt(message, alice_key)
    lines.append(f"Cipher bits: {''.join(str(bit) for bit in cipher_bits)}")
    decrypted = otp_decrypt(cipher_bits, bob_key[:required_bits])

    lines.append("\n--- QKD Secure Communication ---")
    lines.append(f"Decrypted message: {decrypted}")

    if decrypted == message:
        lines.append("Success: Bob recovered the original message.")
    else:
        lines.append("Failure: Bob could not recover the message correctly.")
    return '\n'.join(lines)


def simulate_qkd_string_transfer(message: str, key_length: int = 256, sample_size: int = 20, eavesdrop: bool = False) -> None:
    print(simulate_qkd_string_transfer_data(message, key_length, sample_size, eavesdrop))


def sift_key(alice_bases: List[int], bob_bases: List[int], raw_key: List[int]) -> List[int]:
    return [bit for bit, a_basis, b_basis in zip(raw_key, alice_bases, bob_bases) if a_basis == b_basis]


def sample_key_agreement(alice_key: List[int], bob_key: List[int], sample_size: int) -> Tuple[bool, List[int]]:
    sample_size = min(sample_size, len(alice_key), len(bob_key))
    if sample_size == 0:
        return False, []

    sample_indices = random.sample(range(len(alice_key)), sample_size)
    alice_sample = [alice_key[i] for i in sample_indices]
    bob_sample = [bob_key[i] for i in sample_indices]

    if alice_sample != bob_sample:
        return False, sample_indices

    return True, sample_indices


def remove_sample_bits(key: List[int], sample_indices: List[int]) -> List[int]:
    return [bit for idx, bit in enumerate(key) if idx not in sample_indices]


def bb84_key_exchange(num_bits: int, sample_size: int = 20, eavesdrop: bool = False) -> Tuple[List[int], List[int], bool, List[int], List[int], List[int], List[int]]:
    alice_bits = generate_random_bits(num_bits)
    alice_bases = generate_random_bases(num_bits)
    bob_bases = generate_random_bases(num_bits)

    if eavesdrop:
        eve_bases = generate_random_bases(num_bits)
        eve_results = [bit if a_basis == e_basis else random.randint(0, 1)
                       for bit, a_basis, e_basis in zip(alice_bits, alice_bases, eve_bases)]
        bob_results = [res if e_basis == b_basis else random.randint(0, 1)
                       for res, e_basis, b_basis in zip(eve_results, eve_bases, bob_bases)]
    else:
        bob_results = [bit if a_basis == b_basis else random.randint(0, 1)
                       for bit, a_basis, b_basis in zip(alice_bits, alice_bases, bob_bases)]

    alice_key = sift_key(alice_bases, bob_bases, alice_bits)
    bob_key = sift_key(alice_bases, bob_bases, bob_results)

    agreed, sample_indices = sample_key_agreement(alice_key, bob_key, sample_size)
    if not agreed:
        return alice_key, bob_key, False, alice_bits, alice_bases, bob_bases, bob_results

    alice_key = remove_sample_bits(alice_key, sample_indices)
    bob_key = remove_sample_bits(bob_key, sample_indices)
    return alice_key, bob_key, True, alice_bits, alice_bases, bob_bases, bob_results


def string_to_bits(message: str) -> List[int]:
    bits = []
    for char in message:
        code = ord(char)
        bits.extend([(code >> (7 - i)) & 1 for i in range(8)])
    return bits


def bits_to_string(bits: List[int]) -> str:
    if len(bits) % 8 != 0:
        raise ValueError("Number of bits must be a multiple of 8.")
    chars = []
    for i in range(0, len(bits), 8):
        byte = 0
        for bit in bits[i:i + 8]:
            byte = (byte << 1) | bit
        chars.append(chr(byte))
    return ''.join(chars)


def otp_encrypt(message: str, key: List[int]) -> List[int]:
    message_bits = string_to_bits(message)
    if len(key) < len(message_bits):
        raise ValueError("Key is too short for one-time pad encryption.")
    return [mb ^ kb for mb, kb in zip(message_bits, key)]


def otp_decrypt(cipher_bits: List[int], key: List[int]) -> str:
    if len(key) < len(cipher_bits):
        raise ValueError("Key is too short for one-time pad decryption.")
    message_bits = [cb ^ kb for cb, kb in zip(cipher_bits, key)]
    return bits_to_string(message_bits)


def simulate_qkd_string_transfer(message: str, key_length: int = 256, sample_size: int = 20, eavesdrop: bool = False) -> None:
    alice_key, bob_key, success, alice_bits, alice_bases, bob_bases, bob_results = bb84_key_exchange(
        key_length, sample_size=sample_size, eavesdrop=eavesdrop)

    scenario = "with eavesdropper" if eavesdrop else "without eavesdropper"
    print(f"\n=== BB84 key exchange {scenario} ===")
    visualize_bb84(alice_bits, alice_bases, bob_bases, bob_results, show_bits=48)
    print(f"BB84 key exchange successful: {success}")
    print(f"Sifted key length after sample removal: {len(alice_key)}")

    if not success or len(alice_key) == 0:
        print("Cannot use the shared key. Eavesdropping detected or key length insufficient.")
        return

    required_bits = len(string_to_bits(message))
    if len(alice_key) < required_bits:
        print("Shared key is too short for the message. Increase key_length.")
        return

    print("\n--- One-Time Pad encryption ---")
    print(f"Message: {message}")
    print(f"OTP key bits used: {''.join(str(bit) for bit in alice_key[:required_bits])}")

    cipher_bits = otp_encrypt(message, alice_key)
    print(f"Cipher bits: {''.join(str(bit) for bit in cipher_bits)}")
    decrypted = otp_decrypt(cipher_bits, bob_key[:required_bits])

    print("\n--- QKD Secure Communication ---")
    print(f"Decrypted message: {decrypted}")

    if decrypted == message:
        print("Success: Bob recovered the original message.")
    else:
        print("Failure: Bob could not recover the message correctly.")


def create_qkd_app() -> None:
    root = tk.Tk()
    root.title("BB84 QKD Visual Simulator")
    root.geometry("900x820")
    root.resizable(False, False)
    root.configure(bg="#101820")

    frame = tk.Frame(root, padx=16, pady=16, bg="#161f2c")
    frame.pack(fill=tk.BOTH, expand=True)

    title_label = tk.Label(
        frame,
        text="Quantum Key Distribution (BB84) Simulator",
        font=("Segoe UI", 18, "bold"),
        fg="#e0f7fa",
        bg="#161f2c",
    )
    title_label.grid(row=0, column=0, columnspan=4, pady=(0, 14), sticky="w")

    subtitle_label = tk.Label(
        frame,
        text="Visualize Alice, Bob, and Eve as they build a shared secret key.",
        font=("Segoe UI", 10),
        fg="#a2cfe4",
        bg="#161f2c",
    )
    subtitle_label.grid(row=1, column=0, columnspan=4, sticky="w")

    banner = tk.Frame(frame, bg="#30475e", height=2)
    banner.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 16))

    tk.Label(frame, text="Message:", fg="#d8dee9", bg="#161f2c", font=("Segoe UI", 10, "bold")).grid(row=3, column=0, sticky="nw")
    message_text = tk.Text(
        frame,
        width=72,
        height=4,
        wrap=tk.WORD,
        bg="#192734",
        fg="#eceff4",
        insertbackground="#88c0d0",
        relief=tk.FLAT,
        bd=0,
        padx=8,
        pady=8,
    )
    message_text.insert("1.0", "Hello from Alice to Bob via QKD!")
    message_text.grid(row=3, column=1, columnspan=3, sticky="w")

    tk.Label(frame, text="Key length:", fg="#d8dee9", bg="#161f2c", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, sticky="w", pady=(12, 0))
    key_length_var = tk.IntVar(value=512)
    tk.Spinbox(
        frame,
        from_=64,
        to=2048,
        increment=64,
        textvariable=key_length_var,
        width=10,
        bg="#1f2b38",
        fg="#eceff4",
        insertbackground="#88c0d0",
        relief=tk.FLAT,
        bd=1,
        justify=tk.CENTER,
    ).grid(row=4, column=1, sticky="w", pady=(12, 0))

    tk.Label(frame, text="Sample size:", fg="#d8dee9", bg="#161f2c", font=("Segoe UI", 10, "bold")).grid(row=4, column=2, sticky="w", pady=(12, 0))
    sample_size_var = tk.IntVar(value=50)
    tk.Spinbox(
        frame,
        from_=4,
        to=512,
        increment=2,
        textvariable=sample_size_var,
        width=10,
        bg="#1f2b38",
        fg="#eceff4",
        insertbackground="#88c0d0",
        relief=tk.FLAT,
        bd=1,
        justify=tk.CENTER,
    ).grid(row=4, column=3, sticky="w", pady=(12, 0))

    eavesdrop_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        frame,
        text="Simulate eavesdropper (Eve)",
        variable=eavesdrop_var,
        fg="#d8dee9",
        bg="#161f2c",
        activeforeground="#e0f7fa",
        activebackground="#161f2c",
        selectcolor="#30475e",
        font=("Segoe UI", 10),
        bd=0,
    ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(14, 0))

    output_label = tk.Label(frame, text="Simulation output:", fg="#e0f7fa", bg="#161f2c", font=("Segoe UI", 11, "bold"))
    output_label.grid(row=6, column=0, columnspan=4, sticky="w", pady=(18, 4))

    output_text = scrolledtext.ScrolledText(
        frame,
        width=100,
        height=24,
        wrap=tk.WORD,
        state=tk.DISABLED,
        bg="#0f1720",
        fg="#d8dee9",
        insertbackground="#88c0d0",
        relief=tk.FLAT,
        bd=1,
        padx=10,
        pady=10,
    )
    output_text.grid(row=7, column=0, columnspan=4, pady=(0, 0), sticky="nsew")

    frame.grid_columnconfigure(1, weight=1)
    frame.grid_columnconfigure(2, weight=0)
    frame.grid_columnconfigure(3, weight=0)

    output_text.tag_configure("header", foreground="#82aaff", font=("Segoe UI", 10, "bold"))
    output_text.tag_configure("success", foreground="#a3be8c")
    output_text.tag_configure("warning", foreground="#ebcb8b")
    output_text.tag_configure("error", foreground="#bf616a")
    output_text.tag_configure("plain", foreground="#d8dee9")

    def insert_colored_output(text: str) -> None:
        output_text.configure(state=tk.NORMAL)
        output_text.delete("1.0", tk.END)
        for line in text.splitlines():
            tag = "plain"
            if line.startswith("===") or line.startswith("---"):
                tag = "header"
            elif "Success:" in line:
                tag = "success"
            elif "Failure:" in line or "Cannot use" in line or "eavesdropping" in line.lower():
                tag = "error"
            elif "key length" in line.lower() or "short" in line.lower():
                tag = "warning"
            output_text.insert(tk.END, line + "\n", tag)
        output_text.configure(state=tk.DISABLED)

    def run_simulation() -> None:
        message = message_text.get("1.0", tk.END).strip()
        if not message:
            messagebox.showwarning("Input required", "Please enter a message to send.")
            return

        key_length = key_length_var.get()
        sample_size = sample_size_var.get()
        eavesdrop = eavesdrop_var.get()

        output = simulate_qkd_string_transfer_data(message, key_length, sample_size, eavesdrop)
        insert_colored_output(output)

    run_button = tk.Button(
        frame,
        text="Run QKD Simulation",
        command=run_simulation,
        width=22,
        bg="#4c79f2",
        fg="white",
        activebackground="#7ea6ff",
        activeforeground="#101820",
        relief=tk.FLAT,
        bd=0,
        padx=10,
        pady=10,
        font=("Segoe UI", 10, "bold"),
    )
    run_button.grid(row=5, column=2, columnspan=2, sticky="e", pady=(14, 0))

    badge = tk.Label(
        frame,
        text="Alice → BB84 → Bob",
        fg="#f5f0ff",
        bg="#30475e",
        font=("Segoe UI", 9, "bold"),
        padx=10,
        pady=6,
    )
    badge.grid(row=8, column=0, columnspan=4, pady=(14, 0), sticky="ew")

    root.mainloop()


def main() -> None:
    create_qkd_app()


if __name__ == "__main__":
    main()
