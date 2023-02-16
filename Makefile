# Path: Makefile
dataset: process_emg.py 
	python process_emg.py

clean:
	rm -f *.json

reademg: emg_task0.py
	python emg_task0.py