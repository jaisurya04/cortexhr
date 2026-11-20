"""Compare a screenshot with the approved reference (same 1365x768 size): prints MAE per pixel and SSIM, writes a side-by-side PNG.
   python scripts/compare_with_reference.py <screenshot.png> <side_by_side.png> [reference.png]  (needs Pillow, numpy, scikit-image)"""
import sys
from PIL import Image, ImageChops
import numpy as np
from skimage.metrics import structural_similarity as ssim
ref=Image.open(sys.argv[3] if len(sys.argv)>3 else 'outputs/console/reference_match/approved_reference.png').convert('RGB')
cur=Image.open(sys.argv[1]).convert('RGB')
assert ref.size==cur.size,(ref.size,cur.size)
a=np.asarray(ref).astype(float);b=np.asarray(cur).astype(float)
print('size',ref.size,'MAE per px',np.abs(a-b).mean().round(2),'SSIM',round(ssim(np.asarray(ref.convert('L')),np.asarray(cur.convert('L'))),4))
side=Image.new('RGB',(ref.width*2,ref.height));side.paste(ref,(0,0));side.paste(cur,(ref.width,0));side.save(sys.argv[2])
