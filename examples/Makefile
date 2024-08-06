
.PHONY: all

all: bauble.png bounce.png canvas3d.png dots.png hoops.png linear.png \
	linelight.png physics.png smoke.png solidgeometry.png sphere.png \
	teaset.png textures.png translucency.png video.png

bauble.png: bauble.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

bounce.png: bounce.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

canvas3d.png: canvas3d.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

dots.png: dots.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

hoops.png: hoops.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

linear.png: linear.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

linelight.png: linelight.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

physics.png: physics.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

smoke.png: smoke.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

solidgeometry.png: solidgeometry.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

sphere.png: sphere.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

teaset.png: teaset.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

textures.png: textures.fl
	flitter --offscreen --lockstep --fps=2 --runtime=1 --define "OUTPUT=$@" $<

translucency.png: translucency.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

video.png: video.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<
