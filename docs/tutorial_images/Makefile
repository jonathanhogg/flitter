
.PHONY: images

images: tutorial1.png tutorial2.png tutorial3.png tutorial4.png tutorial5.png tutorial5.gif tutorial6.png tutorial6.gif

%.png: %.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

%.gif: %.fl
	flitter --offscreen --lockstep --fps=10 --runtime=5 --define "OUTPUT=$@" $<
