
# Tutorial

This tutorial assumes that you have **Flitter** [installed](install.md) and
that you are comfortable running it from the command-line. We'll be building
scripts as we go along and, as **Flitter** will live reload code, you can
follow along in the editor of your choice making changes and seeing the result
instantly.

## Declarative Programming

**Flitter** is first-and-foremost a [declarative
language](https://en.wikipedia.org/wiki/Declarative_programming) for describing
visuals. Let's start with the simplest example:

```flitter
!window size=1920;1080
```

Save this to a file, say `tutorial.fl`, and execute it with:

```sh
flitter tutorial.fl
```

A 16:9 black window should open. The `size=1920;1080` declares that we want to
draw 1920 x 1080 graphics inside this window. The window won't necessarily
actually be 1920 x 1080, **Flitter** will resize it as necessary to fit on
screen neatly and the OS is responsible for the pixel dimensions of the
framebuffer that backs the window. However, the size here is an important hint
for things we will put inside this window.

At this point we should probably put something inside the window. Change the
code to read:

```{code-block} flitter
:emphasize-lines: 2

!window size=1920;1080
    !canvas
```

You can leave **Flitter** running in the terminal, changes to the file will
be immediately reloaded. Nothing will visually change about the window at the
moment, but this new line instructs **Flitter** to place a drawing canvas
inside the window.

The drawing canvas that we have placed inside the window will be *exactly*
1920 x 1080 pixels in size – the canvas inherits its size from the enclosing
window `size` attribute. We could override this by putting a `size=` attribute
after `!canvas`.

We should start actually drawing something. Change the file to read:

```{code-block} flitter
:emphasize-lines: 3

!window size=1920;1080
    !canvas
        !text text="Hello world!" font_size=100 point=960;540 color=1
```

You should see the words "Hello world!" appear in white in the middle of the
window. Lets unpick what is happening here: Each line beginning `!` and followed
by a name creates a *node* of that *kind*. These can be followed by any number
of *name*`=`*value* pairs, each of which adds an *attribute* to the preceding
node. Values are *vectors* of any combination of numbers or Unicode strings
surrounded by single or double quote characters. Semicolons `;` are used to
separate multiple items in a vector. Here the `size` attribute of `!window` and
the `point` attribute of `!text` are both two item numeric vectors. All of the
other values are single item vectors. There are no non-vector values in
**Flitter**.

Nodes form a *tree*, with each allowed to have multiple child nodes. The
structure of this tree is created through indentation. Any number of indented
nodes given below another node will become children of that node. All children
of a specific node must be indented to the same level. Indenting in further,
as has been done with the `!text` node, causes a new parent/child relationship.
So the `!canvas` node is a child of `!window` and the `!text` node is a child
of `!canvas`.

Names, including node kinds and attributes, may be any Unicode alphabetic
character or an underscore, followed by any number of Unicode alphanumeric
characters or underscores. The **Flitter** language itself only constructs
trees of nodes and has no notion of which kinds or attributes are correct at any
point in those trees. However, after a tree has been created, it is passed off
to the rendering framework which then interprets the kinds, attributes and
values to render some output.

In this case the renderer is doing the following:

- A window with a 16:9 aspect ratio has been created. This will be kept open
  as long as **Flitter** is running and the program continues to contain the
  `!window` node. The window can be resized and moved around, but will retain
  this 16:9 aspect and cannot be manually closed. The window has a black
  background.
- Inside this window, a [Skia](https://skia.org) 1920 x 1080 pixel `!canvas` has
  been created. It is initially filled with transparent pixels.
- The canvas renderer draws some text (`!text text="Hello world!"`), in
  Helvetica (the default typeface family) at 100px (`font_size=100`), centred
  within the window (`point=960;540`), and in white (`color=1`).

The rendering roughly proceeds down through the tree drawing everything that
needs to be drawn, and then works back up compositing the elements together. So
the text is drawn inside the canvas, and then this canvas is drawn into the
window.

Add the following line to the code:

```{code-block} flitter
:emphasize-lines: 4

!window size=1920;1080
    !canvas
        !text text="Hello world!" font_size=100 point=960;540 color=1
        !text text="Hello world!" font_size=100 point=962;538 color=1;0;0
```

You should see the window now displays the message in red, overlaid on top of
the original white text. There are some immediately useful things to learn from
this:

- Drawing generally proceeds downwards through the file, with later elements
  drawn on top of earlier ones.
- The drawing canvas follows the common document convention of the origin being
  at the top left and the *y* axis pointing down. So `538` is higher up in the
  window than `540` and `962` is further right than `960`.
- Colors can be specified as either 1 or 3 item vectors. If given as a single
  value then the number represents a gray level from 0 to 1, if given as three
  values then the number represents an RGB triplet also in the range 0 to 1.

We can pull out some of the duplicated values in these two text nodes so that
the intention is more clear. Try changing the code to the following (be careful
of the new indentation):

```{code-block} flitter
:emphasize-lines: 3,4,5

!window size=1920;1080
    !canvas
        !group font_size=100 translate=960;540 color=1;0;0
            !text text="Hello world!" color=1
            !text text="Hello world!" point=2;-2
```

Here we place the two `!text` nodes inside a `!group` node that abstracts out
the common `font_size`, changes the drawing origin with `translate` and sets
a default `color`. The first `!text` node overrides this default color and,
without a `point` attribute, will be drawn at the new origin. The second
specifies a drawing point offset from this origin 2px to the right and 2px up.
Outside of this `!group` node, the previous font size, drawing origin and paint
color apply.

`!group` nodes alter the drawing context for the nodes that they contain. They
are able to change the local transformation matrix that establishes the drawing
coordinate system, including rotating and scaling; change the default drawing
color, and various other *paint* properties like line width; and set default
font properties, including font size, the typeface family and weight. We still
have to specify the actual text to be drawn at both nodes as this is individual
to each `!text` node.

An important lesson to learn from this tiny example is that both structure (this
is *in* that) and context (like origin and paint color) are managed through
indentation in **Flitter**. Whereas other drawing systems might have a drawing
context that needs to be explicitly saved and restored, in **Flitter** one only
needs to change the indentation level.
