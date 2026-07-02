import torch
import flash_attention

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. This test is for later CUDA extension work.")
        return


    q = torch.randn(1, 1, 4, 8, device="cuda")
    k = torch.randn(1, 1, 4, 8, device="cuda")
    v = torch.randn(1, 1, 4, 8, device="cuda")

    try:
        flash_attention.forward(q, k, v)
    except RuntimeError as exc:
        print(f"Extension loaded, placeholder implementation raised: {exc}")


if __name__ == "__main__":
    main()
