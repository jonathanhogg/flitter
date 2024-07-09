
.PHONY: diagrams

diagrams: easings.png pseudorandoms.png waveforms.png

%.png: %.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<
