
.PHONY: images

images: tutorial1.jpg tutorial2.jpg tutorial3.jpg tutorial4.jpg tutorial5.jpg tutorial5.gif tutorial6.jpg tutorial6.gif

%.jpg: %.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

%.gif: %.fl
	flitter --offscreen --lockstep --fps=10 --runtime=5 --define "OUTPUT=$@" $<
