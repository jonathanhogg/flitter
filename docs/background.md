
# Background

## Where did Flitter come from?

This is probably my third implementation of a variant of these ideas.
Originally, I developed a simple visuals system as an embedding in Python
(using a crazy system of `with` statements), which sent JSON graphs over a
WebSocket to a JavaScript web app that rendered the results in an HTML 2D
canvas. This was actually a three-layer system that involved running the Python
to build a graph, with the graph using a small expression language that allowed
evaluating the attribute values with respect to time, and then the final static
graph would be rendered. I quickly found this too limiting and wanted a more
expressive, single combined language.

This current version was initially developed over a furious fortnight in
October 2021 leading up to a live performance at the Purcell Room, London
Southbank Centre, doing visuals for Bishi at her 'Let My Country Awake' album
launch. The work was partially supported by Ableton, who gave me an artist
discount on a Push 2. I've been working on **Flitter** off-and-on since then,
developing it as a live tool.

While **Flitter** supports live-coding, that's not primarily why or how I
designed it. As a programmer, I enjoy using code to create visual artwork and I
am fascinated by how generating all of the visuals live affords me the ability
to improvise a performance. However, I'm interested in complex, narrative
visuals that do not lend themselves to coding in-the-moment. Thus, I have in no
way optimised the language for this purpose - meaning it is probably too verbose
for performative coding (and I haven't bothered building a mechanism for
overlaying code onto the output). That said, there are some interesting live
aspects to declarative programming, such as being able to restructure through
indentation.

I spend a huge amount of time in advance of a performance thinking and designing
the visuals. In this period I will constantly iterate on the code, and there I
really value the ability to immediately see the effects of code changes.
However, during a performance I prefer to use physical knobs and buttons to riff
on the themes I have developed in advance. So **Flitter** is designed to
interface with a MIDI surface and then provide different ways to parameterise
the running code, and manipulate and alter the graph that describes the visuals.

Nothing about **Flitter** is in any sense "finished". It is still a testbed for
my ideas. The language and graph semantics are in near constant flux and I
semi-frequently rethink how something works and introduce breaking changes. I've
put this on GitHub in case anyone finds something in this interesting. You'll
find that the frequency of my commits is in direct proportion to how close I am
to a gig.

## A note about the name

Much like the tortured path of the software, the name has evolved over time.
The *very* original version of this idea was called **flight**, which stood for
"functional lighting", as my first 2D visuals system was actually an extension
of a framework I wrote to control DMX lights. The name **Flitter** was a sort
of extension of this from the standard meaning of "flitter", which is to fly
back and forth. However, "flitter" has a number of other meanings in different
dialects of old Scots (I am a secret Scot), including "a small shiny piece of
metal" – like a sequin. I like the name encompassing both movement and light.
