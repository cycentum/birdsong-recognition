from pathlib import Path
from xml.etree import ElementTree

def readAnnotationXml(fileAnnotation):
	rootNode=ElementTree.parse(fileAnnotation).getroot()
	sequences=[]
	numSequence=int(rootNode[0].text)
	for sequenceNode in rootNode[1:]:
		waveFileName=sequenceNode[0].text
		position=int(sequenceNode[1].text)
		length=int(sequenceNode[2].text)
		numNote=int(sequenceNode[3].text)
		notes=[]
		sequence={"waveFileName":waveFileName,"position":position,"length":length,"notes":notes}
		sequences.append(sequence)
		for noteNode in sequenceNode[4:]:
			position=int(noteNode[0].text)
			length=int(noteNode[1].text)
			label=noteNode[2].text
			note={"position":position,"length":length,"label":label}
			sequence["notes"].append(note)
		assert len(sequence["notes"])==numNote
	assert len(sequences)==numSequence
	return sequences


if __name__=="__main__":
	
	dirFold=Path("./../fold")
	dirSong=Path("path/to/song/data/downloaded/from/figshare")
	
	fs=32000
	
	print("Bird", "Fold", "Set", "Length_(sample)", "Length_(minutes)", sep="\t")
	
	for bird in range(11):
		fileAnnotation=dirSong/("Bird"+str(bird))/"Annotation.xml"
		annotation=readAnnotationXml(fileAnnotation)
		
		for fold in range(3):
			
			for setName in ("training_2min", "training_8min", "validation"):
				file=dirFold/("Bird"+str(bird))/("fold"+str(fold))/(setName+".txt")
				
				sequences=[]
				with open(file, "r") as f:
					for line in f:
						line=line.rstrip()
						s=int(line)
						sequences.append(s)
				
				length=0
				for s in sequences:
					l=annotation[s]["length"]
					length+=l
				
				print(bird, fold, setName, length, length/fs/60, sep="\t")
