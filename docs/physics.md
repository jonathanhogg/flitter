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
- `resolution` is an optional attribute specifying a *minimum* simulation step
interval

The simulation maintains an internal simulation clock which begins at zero and
advances in steps equal to the minimum of `resolution` or the difference between
successive values of `time`. This means that if the engine frame rate falls
below that necessary to maintain `time` deltas greater than `resolution`, the
simulation will advance in time steps of `resolution`. This allows `resolution`
to be used to avoid numerical instability caused by slow frames. In this
situation, the internal simulation clock will advance more slowly than `time`.
The internal simulation clock can be read from the state and used to track the
actual amount of simulation time that has passed (see **State interaction**
below).

If `time` and `resolution` are not specified then the system will default to
**flitter**'s internal frame clock and the target frame-rate interval
respectively. This means that the simulation time units will be seconds and the
simulation will advance in time steps equal to the frame interval.

If `time` *is* specified then `resolution` should be set to a sensible matching
value somewhere at or above the expected increment in `time` at the engine
frame-rate.

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
\vec{p}_{t+\Delta t} = \vec{p}_t + \vec{v}_{t+\Delta t} . \Delta t
```

The attributes that specify properties of the particle are:

- `id` - each particle must have a unique `id` attribute and this will be
combined with the `!physics`' `state_prefix` value to produce keys against
which the current state of the particle will be stored in the state dictionary
- `position` - specifies an initial position vector to use for the particle at
the first simulation step (defaults to zero)
- `velocity` - specifies an initial velocity vector (defaults to zero)
- `force` - specifies a constant force vector to be applied to the particle,
this may be changed at any point during the simulation to create custom forces,
e.g., thrust from an engine (defaults to zero)
- `ease` - specifies an amount of simulation time over which to ramp up the
force vector (does nothing if `force` is not specified)
- `radius` - specifies a radius for a spherical particle, this is used both
in collision detection and when calculating drag force (defaults to 1 and will
be clamped to zero if negative)
- `mass` - specifies a value to be used both as the inertial component of
converting forces into accelleration and as the mass for computing gravitic
attraction (defaults to 1 and will be clamped to zero if negative)
- `charge` - specifies a value to be used for calculating electrostatic force
(defaults to 1)

As `mass` is used when calculating acceleration, particles with zero mass
cannot be the subject of a force, meaning they will always continue travelling
at their initial velocity (or remain fixed at their initial position).
However, particles with zero mass will still be considered when computing
forces on other particles.

> **A note on "easing"**
>
> Starting a simulation with forces immediately applied can cause wild
> instability due to massive forces being computed. This applies particularly
> when using a collision force applier and particle starting positions that map
> overlap. To avoid this, all forces can be "eased"-in with the `ease`
> attribute. This specifies an amount of simulation time to linearly ramp up
> the strength of the force applier, giving an amount of time for particles to
> settle into a more stable position.

### `!anchor`

An `!anchor` is a particle that is considered for the purposes of calculating
forces, but will not be affected by any force – including a `force`
attribute. The attributes are otherwise the same as for `!particle`, with the
added difference that `position` specifies the *current* position of the object
rather than an initial value. Thus an `!anchor` may be arbitrarily moved around.

While a zero-mass particle is similar to an anchor, a zero-mass particle cannot
be moved once the simulation has started. An anchor can usefully have zero mass,
for example if it is to be one side of a distance force but should be ignored
for the purposes of calculating attraction due to gravity.

### `!barrier`

A `!barrier` constrains all particles to be on one side of it. In the case of
a system with 3 dimensions, this will be an infinite plane; with 2 dimensions,
a line; and with 1 dimension, a point. Particles that hit a barrier will
"bounce" by reflecting the velocity.

- `position` - specifies the origin for the barrier
- `normal` - specifies the orientation of the barrier; particles are
constrained to be on the side of the barrier in the direction of this vector
- `restitution` - the coefficient of restitution (default is `1`)

Particles bouncing off a barrier will have reflected speed in proportion to
their original speed multiplied by the coefficient of restitution: a value of
`1` will result in a perfectly elastic collision, whereas `0` would mean all of
the particle's velocity is absorbed.

### `!constant`

Specifies a constant force or acceleration to be applied to all particles. This
is useful for simulating global forces such as fields, winds, or gravity.

- `force` - specifies a constant force vector (such as an electric field)
- `acceleration` - specifies a constant acceleration vector (such as large-body
gravity)
- `strength` - specifies a multiplier for `force`/`acceleration` vector
(default is `1`)
- `ease` - specifies an amount of simulation time over which to ramp up
`strength`

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
- `ease` - specifies an amount of simulation time over which to ramp up
`strength`

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

### `!collision`

This creates an implicit `!distance` force applier between all pairs of
particles, with `min` set to the sum of the `radius` attributes of each
particle. Particles with zero `radius` will be ignored.

- `strength` - force magnitude coefficient
- `ease` - specifies an amount of simulation time over which to ramp up
`strength`

The `strength` attribute is inversely proportional to the elasticity of the
particles: lower values mean the particles can overlap more before bouncing
apart. Setting this value too high can cause wild instability – especially if
random starting positions are chosen that might cause particles to
overlap each other.

### `!gravity`

`!gravity` creates an attractive force that applies to all pairs of particles
in proportion to the product of their `mass` attributes and inversely
proportional to the square of the distance between the particles. Particles
with zero `mass` will be ignored.

- `strength` - force magnitude coefficient
- `ease` - specifies an amount of simulation time over which to ramp up
`strength`
- `max_distance` - pairs of particles further apart than this will be ignored

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
need to be quite a large number to see any noticeable effect. `max_distance` is
an optimization feature that allows running the simulation slightly faster by
skipping over particle pairings where they are far apart and will have only a
minimal effect on each other.

Gravitational forces are ignored for overlapping particles, i.e., the minimum
distance over which gravity will be calculated is the sum of the `radius` of
each particle. This is to avoid the wild instability caused by massive forces
when the distance is very small.

### `!electrostatic`

`!electrostatic` creates an attractive force that applies to all pairs of
particles in proportion to the *signed* product of their `charge` attributes and
inversely proportional to the square of the distance between the particles.
Particles with zero `charge` will be ignored. Particles with the same sign
charge will repel each other and particles with oppositely signed charges will
attract each other.

- `strength` - force magnitude coefficient
- `ease` - specifies an amount of simulation time over which to ramp up
`strength`
- `max_distance` - pairs of particles further apart than this will be ignored

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

Except for the ability to have negative charges, electrostatic force operates
in the same way as gravity – including being ignored for overlapping particles.

> **Note**
>
> The `!collision`, `!gravity` and `!electrostatic` force appliers are all
> compute-intensive as they have to consider all particle *pairings* and thus
> have $O(n^2)$ time-complexity.

### `!drag`

`!drag` simulates the effect of particles moving through a liquid/gas by
applying a force to each particle, against the direction of movement, in
proportion to the square of the speed and square of the `radius`. This is
very useful for taking energy out of a simulation, otherwise particles will
tend to bounce around forever. Particles with zero `radius` will be ignored.

- `strength` - force magnitude coefficient
- `ease` - specifies an amount of simulation time over which to ramp up
`strength`

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

- *prefix* - the last simulation timestamp (either the last value of the `time`
attribute or the internal engine frame time)
- *prefix*`;:clock` - the internal simulation clock
- *prefix*`;`*id* - the last position of the particle
- *prefix*`;`*id*`;:velocity` - the last velocity of the particle

`!anchor` particles still store their position and velocity in the state
dictionary, even though position will always be whatever was provided with the
`position` attribute and velocity will always be a zero vector.

`state` and `id` can be any non-null vectors, but `id` must be unique within
the system and `state` must be unique if multiple simultaneous systems are
used. Obviously one should avoid using a `state` prefix that might collide with
other users of the state dictionary, such as MIDI controllers. One should also
not use `:clock` as the id of a particle.

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
    !anchor id=:middle position=0;0
    for i in ..NBUBBLES
        let start=(beta(:start;i)[..2]-0.5)*2*DISH
            random=2*RADIUS*(noise(:x;i, beat/2);noise(:y;i, beat/2))
            charge=10*normal(:charge)[i]
        !particle id=i charge=charge radius=RADIUS position=start force=random
        !distance strength=1000 max=DISH-RADIUS from=i to=:middle
    !electrostatic strength=1000 ease=10
    !collision strength=200 ease=10
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

## Non-realtime mode

If the **flitter** engine is run in non-realtime mode, with the `--lockstep`
command-line option, then the simulation behaviour with regard to the
`resolution` attribute is slightly different. In non-realtime mode,
`resolution` still represents a minimum interval that the simulation will use
but instead of lagging if the frame-rate is slower than this, additional
simulation steps will be inserted to keep the simulation up to date.

For example, if resolution is set to `1/60` and the engine is run with
`--lockstep --fps=30` – for instance, to record a clean output video – then
the simulation will advance *two* steps at each frame instead of subjectively
slowing down.

This behaviour is only supported in non-realtime mode as, when running realtime,
if the engine is unable to keep up with the `resolution` interval then running
additional frames of the simulation would make the problem worse and quickly
result in the engine slowing to a halt.
