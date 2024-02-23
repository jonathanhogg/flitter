
.PHONY: images

images: tutorial1.jpg tutorial2.jpg tutorial3.jpg tutorial4.jpg tutorial5.jpg

%.jpg: %.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 $<
