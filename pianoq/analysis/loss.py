import sys; sys.path.append("C:\\code")
from pianoq_results.fits_image import FITSImage

fb = FITSImage(r'G:\\My Drive\\Projects\\Quantum Piano\\Results\\Calibrations\\SPDC\\PPKTP\\New-2022-10\\f=250_before\\Temperature\\Image\\Preview_20221026_172826_0.05sec_Bin1_26.6C_gain0_T=29.2.fit')
fixed_b = fb.image[70:670, 400:1000] - fb.image[:50, :50].mean()
# sum_b = fixed_b[220:380, 200:350].sum()
sum_b = fixed_b.sum()


fa = FITSImage(r'G:\\My Drive\\Projects\\Quantum Piano\\Results\\Calibrations\\SPDC\\PPKTP\\New-2022-10\\f=250_before\\After fiber\\Preview_20221027_142158_10sec_Bin1_27.3C_gain0_filter10nm_no_PBS.fit')
fixed_a = fa.image[40:640, 510:1110] - fa.image[:50, :50].mean()
sum_a = fixed_a.sum()

sum_a = sum_a / 200  # exposure fix
transmission = sum_a / sum_b
print(transmission)
