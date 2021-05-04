from pathlib import Path
import itertools

def readErrorRate(file):
	errorRate={}
	with open(file, "r") as f:
		for li,line in enumerate(f):
			if li==0: continue
			
			line=line.rstrip().split("\t")
			algorithm=line[0]
			training_duration=line[1]
			
			note_error_rate=float(line[2].split()[0])
			assert note_error_rate == int(line[2].split()[1][1:]) / int(line[2].split()[3][:-1])
			
			timing_error_rate=float(line[3].split()[0])
			assert timing_error_rate == int(line[3].split()[1][1:]) / int(line[3].split()[3][:-1])
			
			note_timing_error_rate=float(line[4].split()[0])
			assert note_timing_error_rate == int(line[4].split()[1][1:]) / int(line[4].split()[3][:-1])
			
			errorRate[algorithm, training_duration]={"note_error_rate":note_error_rate, "timing_error_rate":timing_error_rate, "note_timing_error_rate":note_timing_error_rate}

	return errorRate

if __name__=="__main__":

	dirErrorRate=Path("./../error_rate")
	
	errorRate={}
	for bird in range(11):
		file=dirErrorRate/("Bird"+str(bird)+".txt")
		errorRate[bird]=readErrorRate(file)
		
	print("algorithm", "training_duration", "note_error_rate", "timing_error_rate", "note_timing_error_rate", sep="\t")
	for algorithm, training_duration in itertools.product(("BD>LC>GS", "LC>BD&GS", "LC&GS>BD&GS"), ("2.0", "8.0")):
		average_note_error_rate=0
		average_timing_error_rate=0
		average_note_timing_error_rate=0
		for bird in range(11):
			average_note_error_rate+=errorRate[bird][algorithm, training_duration]["note_error_rate"]
			average_timing_error_rate+=errorRate[bird][algorithm, training_duration]["timing_error_rate"]
			average_note_timing_error_rate+=errorRate[bird][algorithm, training_duration]["note_timing_error_rate"]
		
		average_note_error_rate/=11
		average_timing_error_rate/=11
		average_note_timing_error_rate/=11
		
		print(algorithm, training_duration, average_note_error_rate, average_timing_error_rate, average_note_timing_error_rate, sep="\t")
		