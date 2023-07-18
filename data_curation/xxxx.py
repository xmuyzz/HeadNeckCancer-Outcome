from rembg import remove
from PIL import Image


input_path = '/mnt/kannlab_rfa/Zezhong/test/yzz.png'
output_path = '/mnt/kannlab_rfa/Zezhong/test/yzz_fix.png'

input = Image.open(input_path)
output = remove(input)
output.save(output_path)
