
.PHONY: all short long

all: short long

short: bauble.png canvas3d.png linelight.png solidgeometry.png teaset.png textures.png translucency.png

long: bounce.png hoops.png physics.png smoke.png video.png

bauble.png: bauble.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

bounce.png: bounce.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

canvas3d.png: canvas3d.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

hoops.png: hoops.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

linelight.png: linelight.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

physics.png: physics.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

smoke.png: smoke.fl
	flitter --offscreen --lockstep --fps=10 --runtime=1 --define "OUTPUT=$@" $<

solidgeometry.png: solidgeometry.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

teaset.png: teaset.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

textures.png: textures.fl
	flitter --offscreen --lockstep --fps=2 --runtime=1 --define "OUTPUT=$@" $<

translucency.png: translucency.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<

video.png: video.fl
	flitter --offscreen --lockstep --fps=1 --runtime=1 --define "OUTPUT=$@" $<
