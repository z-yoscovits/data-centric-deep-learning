from src.system import DigitClassifierSystem

system = DigitClassifierSystem.load_from_checkpoint(
    './artifacts/ckpts/train_flow/epoch=0-step=1500.ckpt')

print(system)