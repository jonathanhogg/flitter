
.PHONY: diagrams

diagrams: box_uvmap.png dummyshader.png easings.png petri.png pseudorandoms.png waveforms.png

petri.png: petri.fl
	flitter --offscreen --lockstep --fps=10 --runtime=10 --define "OUTPUT=$@" $<

%.png: %.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<
