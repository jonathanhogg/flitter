
# Controlling code with a Push 2

Assuming that you have an Ableton Push 2 connected,

```
flitter-push
```

will fire up the process that talks to it. This interfaces with the engine via
OSC messaging (on `localhost`) and is generally resilient to the engine starting
and stopping. You can also automatically start the Push 2 interface as a
managed subprocess of the engine by just adding the `--push` command-line
option to `flitter`.

Other than tempo control, you won't have much in the way of interface until you
specify one in the program itself. `!pad` and `!encoder` nodes at the top level
in the graph will configure pads and encoders on the Push 2. Again, really
you'll need to look at the examples.

The outputs from the pads and encoders are put into a special environment map
in the engine that is read from with `$(:some_key)`. This allows one to
parameterise the program and live manipulate it.

If multiple code files are specified on the command line of the engine, then
these will be loaded as multiple "pages". The previous and next page buttons on
the Push 2 can be used to switch between the files. The state of each page is
maintained when switching, including the current tempo and start time of the
beat clock, and the state values of all of the pads and encoders.
