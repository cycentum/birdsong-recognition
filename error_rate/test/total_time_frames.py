from pathlib import Path
import sys
import numpy as np

sys.path.append("../../fold/test")
from show_duration import readAnnotationXml

def readTotalFrameLengthFromErrorFile(file):
	length=None
	with open(file, "r") as f:
		for li,line in enumerate(f):
			if li==0: continue
			
			line=line.rstrip().split("\t")
			algorithm=line[0]
			training_duration=line[1]
			
			le=int(line[3].split()[3][:-1])
			if length is None: length=le
			assert length==le

	return length

if __name__=="__main__":
	
	dirErrorRate=Path("./..")
	dirSong=Path("path/to/song/data/downloaded/from/figshare")
	
	fft_size=512
	fft_step=32
	fft_overlap=fft_size-fft_step
	
	for bird in range(11):
		fileAnnotation=dirSong/("Bird"+str(bird))/"Annotation.xml"
		annotation=readAnnotationXml(fileAnnotation)
		
		totalSpectrogramLength=0
		for sequence in annotation:
			length=sequence["length"]
			spectrogramLength=int(np.ceil((length-fft_overlap)/fft_step))
			totalSpectrogramLength+=spectrogramLength
		
		fileErrorRate=dirErrorRate/("Bird"+str(bird)+".txt")
		lengthFromErrorFile=readTotalFrameLengthFromErrorFile(fileErrorRate)
		
		assert lengthFromErrorFile==totalSpectrogramLength
		
		print("Bird"+str(bird), totalSpectrogramLength)
		