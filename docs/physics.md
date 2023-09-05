# Physics simulation

**flitter** supports a simple system for running physics simulations of particle
systems. The physics engine is configured with a set of nodes and will store the
current position and velocity of each particle in the state dictionary, allowing
it to be linked back to rendering code and nodes.

A new physics system is introduced by a `!physics` node at the top level,
containing a number of `!particle` or `!anchor` nodes along with force applier
nodes that define the specific physics of the system.

## Nodes

### `!physics`

A `!physics` node at the top level in a **flitter** script creates a new
simulation system. The attributes are:

- `state` is a required attribute giving a state prefix key; all of the state
of the system will be stored in the **flitter** state dictionary with keys
prefixed by this
- `dimensions` is a required attribute giving the number of dimensions of the
system, i.e., the length of the position, value and force vectors
- `time` is an optional attribute providing the simulation clock
- `resolution` is an optional attribute specifying a minimum simulation step
time

If `time` and `resolution` are not specified then the system will default to
**flitter**'s internal frame clock and the target frame-rate interval. If
`time` is specified, then you should also provide a sensible matching
`resolution` value. `time` cannot run backwards and the simulation will simply
stop running in this instance. If `time` jumps backwards, or forwards by more
than `resolution`, then the simulation will continue from the new `time` point.

> :warning: **Note**
>
> If the physics engine is called with sequential `time` values (or the actual
> frame times) that have a delta greater than `resolution`, then the simulation
> will be run with `resolution` as the time delta, i.e., if **flitter** is
> unable to call the engine rapidly enough, the simulation will subjectively
> appear to slow down. This is to avoid numerical instability, for example if
> two objects overlap each other because of processing a large step at their
> current velocity and a collision force applier is in place, then this may
> generate massive repelling forces that will cause the particles to fly apart.

All other nodes described below must be contained with a `!physics` node.

### `!particle`

A `!particle` node specifies a point/spherical object to be simulated. At each
step, the physics engine will compute the forces to be applied to each particle
based on their properties, current positions and velocities. These forces will
then be applied to the current velocity to generate a new velocity and the
position updated based on this.

```math
\vec{v}_{t+\Delta t} = \vec{v}_t +
\left( \sum_{i=0}^n \vec{F}_i \right) {\Delta t \over M}
```

```math
\vec{p}_{t+\Delta t} = \vec{p}_t + \vec{v}_{t+\Delta t} \Delta t
```

The attributes that specify properties of the particle are:

- `id` - each particle must have a unique `id` attribute and this will be
combined with the `!physics`' `state_prefix` value to produce keys against
which the current state of the particle will be stored in the state dictionary
- `position` - specifies an initial position to use for the particle at the
first simulation step
- `velocity` - specifies an initial velocity
- `force` - specifies a constant force vector to be applied to the particle,
this may be changed at any point during the simulation to create custom forces,
e.g., thrust from an engine
- `radius` - specifies a radius for a spherical particle, this is used both
in collision detection and when calculating drag force (defaults to 1)
- `mass` - specifies a value to be used both as the inertial component of
converting forces into accelleration and as the mass for computing gravitic
attraction (defaults to 1)
- `charge` - specifies a value to be used for calculating electrostatic force
(defaults to 1)

### `!anchor`

An `!anchor` is a particle that is considered for the purposes of calculating
forces, but will not be affected by any force – including a `force`
attribute. The attributes are otherwise the same as for `!particle`, with the
added difference that `position` specifies the *current* position of the object
rather than an initial value. Thus an `!anchor` may be arbitrarily moved around.

### `!constant`

Specifies a constant force to be applied to all particles. This is useful for
simulating the "down" acceleration from the gravity of a large body.

- `direction` - specifies a vector representing the force direction (which
will be normalized)
- `strength` - specifies the magnitude of the constant force along the
direction vector
- `force` - alternative to `direction` and `strength` for giving the force
vector directly

```math
\vec{F} = \textbf{strength} . \vec{d}
```

`direction` and `strength` are available rather than just `force` as it allows
one to easily specify a direction in terms of two points, e.g.:

```
!constant direction=bottom-top strength=100
```

is equivalent to:

```
!constant force=100*normalize(bottom-top)
```

but possibly more obvious.

### `!distance`

Specifies a force to be applied between two specific particles that linearly
scales with displacement from a minimum or maximum distance. This can be used
to simulate various tethers, rubber bands and springs.

- `from` - the `id` of the first particle
- `to` - the `id` of the second particle
- `min` - a minimum distance that the two objects can be apart
- `max` - a maximum distance that the two objects can be apart
- `fixed` - a shortcut for setting both `min` and `max` to the same value
- `strength` - force magnitude coefficient

```math
l = \left| \vec{p}_\textbf{to} - \vec{p}_\textbf{from} \right|
```

```math
\vec{d} = {\vec{p}_\textbf{to} - \vec{p}_\textbf{from} \over l}
```

```math
\vec{F} = \begin{cases}
\textbf{strength} . (l - \textbf{min}) . \vec{d}
& \text{if $l < \textbf{min}$} \\
\textbf{strength} . (l - \textbf{max}) . \vec{d}
& \text{if $l > \textbf{max}$}
\end{cases}
```

```math
\vec{F}_\textbf{from} = \vec{F}
```

```math
\vec{F}_\textbf{to} = -\vec{F}
```

### `!collision`

This creates an implicit `!distance` force applier between all pairs of
particles, with `min` set to the sum of the `radius` attributes of each
particle. Particles with zero or negative `radius` will be ignored.

- `strength` - force magnitude coefficient

The `strength` attribute is inversely proportional to the elasticity of the
particles: lower values mean the particles can overlap more before bouncing
apart. Setting this value too high can cause wild instability – especially if
random starting positions are chosen that might cause particles to
overlap each other.

### `!gravity`

`!gravity` creates an attractive force that applies to all pairs of particles
in proportion to the product of their `mass` attributes and inversely
proportional to the square of the distance between the particles. Particles
with zero or negative `mass` will be ignored.

- `strength` - force magnitude coefficient

```math
\vec{F} = \textbf{strength} . \vec{d} .
{ \textbf{mass}_{from} . \textbf{mass}_{to} \over l^2}
```

```math
\vec{F}_{from} = \vec{F}
```

```math
\vec{F}_{to} = -\vec{F}
```

As gravity falls off with the square of the distance, `strength` will normally
need to be quite a large number to see any noticable effect.

### `!electrostatic`

`!electrostatic` creates an attractive force that applies to all pairs of
particles in proportion to the *signed* product of their `charge` attributes and
inversely proportional to the square of the distance between the particles.
Particles with zero `charge` will be ignored. Particles with the same sign
charge will repel each other and particles with oppositely signed charges will
attract each other.

- `strength` - force magnitude coefficient

```math
\vec{F} = \textbf{strength} . \vec{d} .
{ | \textbf{charge}_{from} | . \textbf{charge}_{to} \over l^2}
```

```math
\vec{F}_{from} = \vec{F}
```

```math
\vec{F}_{to} = -\vec{F}
```

As electrostatic force falls off with the square of the distance, `strength`
will normally need to be quite a large number to see any noticable effect.

### `!drag`

`!drag` simulates the effect of particles moving through a liquid/gas by
applying a force to each particle, against the direction of movement, in
proportion to the square of the speed and square of the `radius`. This is
very useful for taking energy out of a simulation, otherwise particles will
tend to bounce around forever. Particles with zero `radius` will be ignored.

- `strength` - force magnitude coefficient

```math
{speed} = |\vec{v}_t|
```

```math
\vec{d} = { \vec{v}_t \over {speed} }
```

```math
\vec{F} = \textbf{strength} . (-\vec{d}) . {speed}^2 . \textbf{radius}^2
```

As drag scales with the square of both the speed and particle radius,
`strength` should normally be a *very* small number. This force applier is
limited to ensure that simulation granularity issues cannot cause a particle to
reverse direction.

## State interaction

For a `!physics` system with `state` set to *prefix*, and a particle with `id`
set to *id*, the following key/value pairs will be stored in the state
dictionary:

- *prefix* - the last simulation timestamp (as provided with the `time`
attribute or the internal frame time
- *prefix*`;`*id* - the last position of the particle
- *prefix*`;`*id*`;:velocity` - the last velocity of the particle

`!anchor` particles still store their position and velocity in the state
dictionary, even though position will always be whatever was provided with the
`position` attribute and velocity will always be a zero vector.

`state` and `id` can be any non-null vectors, but `id` must be unique within
the system and `state` must be unique if multiple simultaneous systems are
used. Obviously one should avoid using a `state` prefix that might collide with
other users of the state dictionary, such as MIDI controllers.

## Example

This example creates a "petri dish" of "cell" particles with
normally-distributed random charges. An `!electrostatic` force applier makes
the cells drift together or apart, clumping up into different shapes. Chains of
cells with alternating charge will form and break up.

A `!collision` force applier stops them from overlapping with each other and a
`!distance` force applier is used to constrain the particles within the dish
by setting a maximum distance for each from an anchor in the middle. A `!drag`
force applier slows the particles as if they are moving through a liquid.

Without any external forces, this system will quickly come to a static
equilibirum, so random forces are derived from the `noise()` function and
applied directly to each particle with the `force` attribute.

The `beat` clock is used as the simulation time, allowing the simulation to be
sped up or slowed down by altering the tempo. `resolution` is calculated to
match the current tempo to the target frame-rate.

```flitter
%pragma tempo 60

let SIZE=1080;1080
    NBUBBLES=200
    RADIUS=15
    DISH=500

!physics state=:cells dimensions=2 time=beat resolution=tempo/60/fps
    let ease_in=linear(beat/10)
    !anchor id=:middle position=0;0
    for i in ..NBUBBLES
        let start=(beta(:start;i)[..2]-0.5)*2*DISH
            random=2*RADIUS*(noise(:x;i, beat/2);noise(:y;i, beat/2))
            charge=10*normal(:charge)[i]
        !particle id=i charge=charge radius=RADIUS position=start force=random
        !distance strength=1000 max=DISH-RADIUS from=i to=:middle
    !electrostatic strength=1000*ease_in
    !collision strength=200*ease_in
    !drag strength=0.0001

!window size=SIZE
    !canvas color=1 translate=SIZE/2
        !path
            for i in ..NBUBBLES
                !ellipse point=$(:cells;i) radius=RADIUS
            !fill color=0;0.5;0
        !path
            !ellipse point=$(:cells;:middle) radius=DISH
            !stroke stroke_width=10
```

Note that the `strength` coefficients for the `!electrostatic` and `!collision`
force appliers are "eased in" by increasing them linearly over the first 10
beats. This allows any particles with overlapping start positions to gently
move apart at the beginning.
