# cython: language_level=3, profile=False, boundscheck=False, wraparound=False, cdivision=True

"""
Flitter physics engine
"""

import asyncio
from loguru import logger

from .. import name_patch
from ..model cimport Vector, Node, StateDict, null_
from ..language.functions cimport normal, uniform

from libc.math cimport sqrt, isinf, isnan, abs, floor
from libc.stdint cimport uint64_t, int64_t
from cpython cimport PyObject

cdef extern from "Python.h":
    # Note: I am explicitly (re)defining these as not requiring the GIL for speed.
    # This appears to work completely fine, but is obviously making assumptions
    # on the Python ABI that may not be correct.
    #
    Py_ssize_t PyList_GET_SIZE(object list) nogil
    PyObject* PyList_GET_ITEM(object list, Py_ssize_t i) nogil

logger = name_patch(logger, __name__)

cdef Vector VELOCITY = Vector._symbol('velocity')
cdef Vector CLOCK = Vector._symbol('clock')
cdef Vector RUN = Vector._symbol('run')

cdef normal NormalRandomSource = normal(Vector.symbol('_physics_normal'))
cdef uint64_t NormalRandomIndex = 0

cdef uniform UniformRandomSource = uniform(Vector.symbol('_physics_uniform'))
cdef uint64_t UniformRandomIndex = 0


cdef inline double random_normal() nogil:
    global NormalRandomSource, NormalRandomIndex
    cdef double d = NormalRandomSource._item(NormalRandomIndex)
    NormalRandomIndex += 1
    return d


cdef inline double random_uniform() nogil:
    global UniformRandomSource, UniformRandomIndex
    cdef double d = UniformRandomSource._item(UniformRandomIndex)
    UniformRandomIndex += 1
    return d


cdef class Particle:
    cdef Vector id
    cdef Vector position_state_key
    cdef Vector position
    cdef Vector velocity_state_key
    cdef Vector velocity
    cdef Vector initial_force
    cdef Vector force
    cdef double radius
    cdef double mass
    cdef double charge
    cdef double ease

    def __cinit__(self, Node node, Vector id, Vector zero, Vector prefix, StateDict state):
        self.id = id
        self.position_state_key = prefix.concat(self.id).intern()
        cdef Vector position = state.get_item(self.position_state_key)
        if position.length == zero.length and position.numbers != NULL:
            self.position = Vector._copy(position)
        else:
            self.position = Vector._copy(node.get_fvec('position', zero.length, zero))
        self.velocity_state_key = self.position_state_key.concat(VELOCITY).intern()
        cdef Vector velocity = state.get_item(self.velocity_state_key)
        if velocity.length == zero.length and velocity.numbers != NULL:
            self.velocity = Vector._copy(velocity)
        else:
            self.velocity = Vector._copy(node.get_fvec('velocity', zero.length, zero))
        self.initial_force = node.get_fvec('force', zero.length, zero)
        self.force = Vector._copy(self.initial_force)
        self.radius = max(0, node.get_float('radius', 1))
        self.mass = max(0, node.get_float('mass', 1))
        self.charge = node.get_float('charge', 1)
        self.ease = node.get_float('ease', 0)

    cdef void update(self, double speed_of_light, double clock, double delta) noexcept nogil:
        cdef double speed, d, k
        cdef int64_t i, n=self.force.length
        if self.mass:
            for i in range(n):
                if isinf(self.force.numbers[i]) or isnan(self.force.numbers[i]):
                    break
            else:
                k = delta / self.mass
                if self.ease > 0 and clock < self.ease:
                    k *= clock / self.ease
                for i in range(n):
                    self.velocity.numbers[i] = self.velocity.numbers[i] + self.force.numbers[i] * k
                if speed_of_light > 0:
                    speed = 0
                    for i in range(n):
                        d = self.velocity.numbers[i]
                        speed += d * d
                    if speed > speed_of_light * speed_of_light:
                        k = speed_of_light / sqrt(speed)
                        for i in range(n):
                            self.velocity.numbers[i] = self.velocity.numbers[i] * k
        for i in range(n):
            self.position.numbers[i] = self.position.numbers[i] + self.velocity.numbers[i] * delta
            self.force.numbers[i] = self.initial_force.numbers[i]


cdef class Anchor(Particle):
    def __cinit__(self, Node node, Vector id, Vector zero, Vector prefix, StateDict state):
        self.position = node.get_fvec('position', zero.length, zero)
        self.velocity = Vector._copy(zero)

    cdef void update(self, double speed_of_light, double clock, double delta) noexcept nogil:
        pass


cdef class Barrier:
    cdef Vector position
    cdef Vector normal
    cdef double restitution

    def __cinit__(self, Node node, Vector zero):
        self.position = node.get_fvec('position', zero.length, zero)
        self.normal = node.get_fvec('normal', zero.length, zero).normalize()
        self.restitution = node.get_float('restitution', 1)

    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        if self.normal.length == 0:
            return
        cdef int i, dimensions=self.position.length
        cdef double d
        d = -particle.radius
        for i in range(dimensions):
            d = d + (particle.position.numbers[i] - self.position.numbers[i]) * self.normal.numbers[i]
        if d >= 0:
            return
        for i in range(dimensions):
            particle.position.numbers[i] = particle.position.numbers[i] - d*self.normal.numbers[i]
        d = 0
        for i in range(dimensions):
            d = d + particle.velocity.numbers[i] * self.normal.numbers[i]
        for i in range(dimensions):
            particle.velocity.numbers[i] = (particle.velocity.numbers[i] - 2*d*self.normal.numbers[i]) * self.restitution


cdef class ForceApplier:
    cdef double strength

    def __cinit__(self, Node node, double strength, Vector zero):
        self.strength = strength


cdef class PairForceApplier(ForceApplier):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        raise NotImplementedError()


cdef class ParticleForceApplier(ForceApplier):
    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        raise NotImplementedError()


cdef class MatrixPairForceApplier(PairForceApplier):
    cdef double max_distance_squared

    def __cinit__(self, Node node, double strength, Vector zero):
        self.max_distance_squared = max(0, node.get_float('max_distance', 0)) ** 2


cdef class SpecificPairForceApplier(PairForceApplier):
    cdef Vector from_particle_id
    cdef Vector to_particle_id
    cdef int64_t from_index
    cdef int64_t to_index

    def __cinit__(self, Node node, double strength, Vector zero):
        self.from_particle_id = <Vector>node._attributes.get('from') if node._attributes else None
        self.to_particle_id = <Vector>node._attributes.get('to') if node._attributes else None


cdef class DistanceForceApplier(SpecificPairForceApplier):
    cdef double power
    cdef double minimum
    cdef double maximum

    def __cinit__(self, Node node, double strength, Vector zero):
        self.power = max(0, node.get_float('power', 1))
        cdef double fixed
        if (fixed := node.get_float('fixed', 0)) != 0:
            self.minimum = fixed
            self.maximum = fixed
        else:
            self.minimum = node.get_float('min', 0)
            self.maximum = node.get_float('max', 0)

    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        cdef double f, k
        cdef int64_t i
        if self.minimum and distance < self.minimum:
            k = self.strength * (self.minimum - distance) ** self.power
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] - f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] + f
        elif self.maximum and distance > self.maximum:
            k = self.strength * (distance - self.maximum) ** self.power
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class DragForceApplier(ParticleForceApplier):
    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        cdef double speed_squared=0, v, k
        cdef int64_t i
        if particle.radius:
            for i in range(particle.velocity.length):
                v = particle.velocity.numbers[i]
                speed_squared += v * v
            k = min(self.strength * sqrt(speed_squared) * particle.radius**(particle.force.length-1), particle.mass / delta)
            for i in range(particle.velocity.length):
                particle.force.numbers[i] = particle.force.numbers[i] - particle.velocity.numbers[i] * k


cdef class BuoyancyForceApplier(ParticleForceApplier):
    cdef double density
    cdef Vector gravity

    def __cinit__(self, Node node, double strength, Vector zero):
        self.density = node.get_float('density', 1)
        self.gravity = node.get_fvec('gravity', zero.length, zero)
        if self.gravity is zero:
            self.gravity = Vector._copy(self.gravity)
            self.gravity.numbers[zero.length-1] = -1

    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        cdef double displaced_mass, k
        cdef int64_t i
        if particle.radius and particle.mass:
            displaced_mass = particle.radius**particle.force.length * self.density
            k = self.strength * (particle.mass - displaced_mass)
            for i in range(particle.force.length):
                particle.force.numbers[i] = particle.force.numbers[i] + self.gravity.numbers[i] * k


cdef class ConstantForceApplier(ParticleForceApplier):
    cdef Vector force
    cdef Vector acceleration

    def __cinit__(self, Node node, double strength, Vector zero):
        self.force = node.get_fvec('force', zero.length, zero)
        self.acceleration = node.get_fvec('acceleration', zero.length, zero)

    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        cdef int64_t i
        for i in range(self.force.length):
            particle.force.numbers[i] = particle.force.numbers[i] + self.force.numbers[i]*self.strength
            particle.velocity.numbers[i] = particle.velocity.numbers[i] + self.acceleration.numbers[i]*self.strength*delta


cdef class RandomForceApplier(ParticleForceApplier):
    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        global RandomIndex
        cdef int64_t i
        for i in range(particle.force.length):
            particle.force.numbers[i] = particle.force.numbers[i] + self.strength * random_normal()


cdef class CollisionForceApplier(MatrixPairForceApplier):
    cdef double power

    def __cinit__(self, Node node, double strength, Vector zero):
        self.power = max(0, node.get_float('power', 1))

    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        cdef double min_distance, f, k
        cdef int64_t i
        if from_particle.radius and to_particle.radius:
            min_distance = from_particle.radius + to_particle.radius
            if distance < min_distance:
                k = self.strength * (min_distance - distance) ** self.power
                for i in range(direction.length):
                    f = direction.numbers[i] * k
                    from_particle.force.numbers[i] = from_particle.force.numbers[i] - f
                    to_particle.force.numbers[i] = to_particle.force.numbers[i] + f


cdef class GravityForceApplier(MatrixPairForceApplier):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        cdef double f, k
        cdef int64_t i
        if from_particle.mass and to_particle.mass and distance > max(from_particle.radius, to_particle.radius):
            k = self.strength * from_particle.mass * to_particle.mass / distance_squared
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class ElectrostaticForceApplier(MatrixPairForceApplier):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        cdef double f, k
        cdef int64_t i
        if from_particle.charge and to_particle.charge and distance > max(from_particle.radius, to_particle.radius):
            k = self.strength * -from_particle.charge * to_particle.charge / distance_squared
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class AdhesionForceApplier(MatrixPairForceApplier):
    cdef double overlap

    def __cinit__(self, Node node, double strength, Vector zero):
        self.overlap = min(max(0, node.get_float('overlap', 0.25)), 1)

    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        cdef double min_distance, max_distance, overlap_distance, adhesion_distance, f, k
        cdef int64_t i
        if from_particle.radius and to_particle.radius:
            max_distance = from_particle.radius + to_particle.radius
            if distance < max_distance:
                overlap_distance = max_distance - distance
                min_distance = abs(from_particle.radius - to_particle.radius)
                adhesion_distance = min_distance*self.overlap + max_distance*(1-self.overlap)
                k = self.strength * overlap_distance * (distance - adhesion_distance)
                for i in range(direction.length):
                    f = direction.numbers[i] * k
                    from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                    to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class PhysicsSystem:
    cdef set state_keys
    cdef int64_t dimensions
    cdef double start_time
    cdef double resolution

    def __init__(self, **kwargs):
        self.state_keys = set()

    def destroy(self):
        pass

    def purge(self):
        pass

    async def update(self, engine, Node node, double clock, double performance, bint slow_frame, **kwargs):
        cdef int64_t run = node.get_int('run', 0)
        self.dimensions = node.get_int('dimensions', 0)
        if self.dimensions < 1:
            return
        cdef Vector state_prefix = <Vector>node._attributes.get('state') if node._attributes else None
        if state_prefix is None:
            return
        cdef double time = node.get_float('time', clock)
        self.resolution = node.get_float('resolution', 1/engine.target_fps)
        if self.resolution <= 0:
            return
        cdef double speed_of_light = max(0, node.get_float('speed_of_light', 0))
        cdef StateDict state = engine.state
        cdef Vector time_vector = state.get_item(state_prefix.concat(CLOCK))
        if time_vector.length == 1 and time_vector.numbers != NULL:
            clock = time_vector.numbers[0]
        else:
            clock = 0
        cdef Vector last_run_vector = state.get_item(state_prefix.concat(RUN))
        cdef int64_t last_run
        if last_run_vector.length == 1 and last_run_vector.numbers != NULL:
            last_run = <int64_t>floor(last_run_vector.numbers[0])
        else:
            last_run = run
        cdef Vector state_key
        if run != last_run:
            logger.debug("Reset physics system {!r} for run {}", state_prefix, run)
            for state_key in self.state_keys:
                state.set_item(state_key, null_)
            self.state_keys = set()
        cdef Vector run_vector = Vector.__new__(Vector)
        run_vector.allocate_numbers(1)
        run_vector.numbers[0] = run
        state.set_item(state_prefix.concat(RUN), run_vector)
        cdef Vector zero = Vector.__new__(Vector)
        zero.allocate_numbers(self.dimensions)
        cdef int64_t i
        for i in range(self.dimensions):
            zero.numbers[i] = 0
        cdef list particles=[], non_anchors=[], particle_forces=[], matrix_forces=[], specific_forces=[], barriers=[]
        cdef Node child
        cdef Vector id
        cdef double strength, ease
        cdef dict particles_by_id = {}
        cdef Particle particle
        cdef Barrier barrier
        cdef set old_state_keys = self.state_keys
        cdef set new_state_keys = set()
        for child in node._children:
            if child.kind == 'particle':
                id = <Vector>child._attributes.get('id')
                if id is not None:
                    particle = Particle.__new__(Particle, child, id, zero, state_prefix, state)
                    particles_by_id[id] = len(particles)
                    particles.append(particle)
                    non_anchors.append(particle)
                    new_state_keys.add(particle.position_state_key)
                    old_state_keys.discard(particle.position_state_key)
                    new_state_keys.add(particle.velocity_state_key)
                    old_state_keys.discard(particle.velocity_state_key)
            elif child.kind == 'anchor':
                id = <Vector>child._attributes.get('id')
                if id is not None:
                    particle = Anchor.__new__(Anchor, child, id, zero, state_prefix, state)
                    particles_by_id[id] = len(particles)
                    particles.append(particle)
                    new_state_keys.add(particle.position_state_key)
                    old_state_keys.discard(particle.position_state_key)
                    new_state_keys.add(particle.velocity_state_key)
                    old_state_keys.discard(particle.velocity_state_key)
            elif child.kind == 'barrier':
                barrier = Barrier.__new__(Barrier, child, zero)
                barriers.append(barrier)
            else:
                strength = child.get_float('strength', 1)
                ease = child.get_float('ease', 0)
                if ease > 0 and ease < clock:
                    strength *= clock/ease
                if child.kind == 'distance':
                    specific_forces.append(DistanceForceApplier.__new__(DistanceForceApplier, child, strength, zero))
                elif child.kind == 'drag':
                    particle_forces.append(DragForceApplier.__new__(DragForceApplier, child, strength, zero))
                elif child.kind == 'buoyancy':
                    particle_forces.append(BuoyancyForceApplier.__new__(BuoyancyForceApplier, child, strength, zero))
                elif child.kind == 'constant':
                    particle_forces.append(ConstantForceApplier.__new__(ConstantForceApplier, child, strength, zero))
                elif child.kind == 'random':
                    particle_forces.append(RandomForceApplier.__new__(RandomForceApplier, child, strength, zero))
                elif child.kind == 'collision':
                    matrix_forces.append(CollisionForceApplier.__new__(CollisionForceApplier, child, strength, zero))
                elif child.kind == 'gravity':
                    matrix_forces.append(GravityForceApplier.__new__(GravityForceApplier, child, strength, zero))
                elif child.kind == 'electrostatic':
                    matrix_forces.append(ElectrostaticForceApplier.__new__(ElectrostaticForceApplier, child, strength, zero))
                elif child.kind == 'adhesion':
                    matrix_forces.append(AdhesionForceApplier.__new__(AdhesionForceApplier, child, strength, zero))
        cdef SpecificPairForceApplier specific_force
        for specific_force in specific_forces:
            specific_force.from_index = particles_by_id.get(specific_force.from_particle_id, -1)
            specific_force.to_index = particles_by_id.get(specific_force.to_particle_id, -1)
        time_vector = state.get_item(state_prefix)
        if time_vector.length == 1 and time_vector.numbers != NULL:
            self.start_time = time_vector.numbers[0]
        else:
            logger.debug("New {}D physics {!r} with {} particles and {} forces", self.dimensions, state_prefix, len(particles),
                         len(particle_forces) + len(matrix_forces) + len(specific_forces))
            self.start_time = time
            time_vector = Vector.__new__(Vector)
            time_vector.allocate_numbers(1)
            time_vector.numbers[0] = self.start_time
            state.set_item(state_prefix, time_vector)
        cdef double behind = (time - self.start_time - clock) / self.resolution
        cdef bint extra_frame = performance > 1 and not slow_frame and random_uniform() > min(100 / behind, 0.9)
        cdef tuple objects
        if time > self.start_time + clock:
            objects = particles, non_anchors, particle_forces, matrix_forces, specific_forces, barriers
            clock = await asyncio.to_thread(self.calculate, objects, engine.realtime, extra_frame, speed_of_light, time, clock)
            for particle in particles:
                state.set_item(particle.position_state_key, particle.position)
                state.set_item(particle.velocity_state_key, particle.velocity)
        time_vector = Vector.__new__(Vector)
        time_vector.allocate_numbers(1)
        time_vector.numbers[0] = clock
        state.set_item(state_prefix.concat(CLOCK), time_vector)
        for state_key in old_state_keys:
            state.set_item(state_key, null_)
        self.state_keys = new_state_keys

    cdef double calculate(self, tuple objects, bint realtime, bint extra_frame, double speed_of_light, double time, double clock):
        cdef list particles, non_anchors, particle_forces, matrix_forces, specific_forces, barriers
        particles, non_anchors, particle_forces, matrix_forces, specific_forces, barriers = objects
        cdef int64_t i, j, k, ii, m, n=len(particles), o=len(non_anchors)
        cdef double delta, last_time = self.start_time + clock
        cdef Vector direction = Vector.__new__(Vector)
        direction.allocate_numbers(self.dimensions)
        cdef double d, distance, distance_squared
        cdef PyObject* from_particle
        cdef PyObject* to_particle
        cdef PyObject* force
        cdef PyObject* barrier
        with nogil:
            while True:
                delta = min(self.resolution, time-last_time)
                m = PyList_GET_SIZE(particle_forces)
                for i in range(m):
                    for j in range(o):
                        (<ParticleForceApplier>PyList_GET_ITEM(particle_forces, i)).apply((<Particle>PyList_GET_ITEM(non_anchors, j)), delta)
                m = PyList_GET_SIZE(specific_forces)
                for i in range(m):
                    force = PyList_GET_ITEM(specific_forces, i)
                    if (<SpecificPairForceApplier>force).from_index != -1 and (<SpecificPairForceApplier>force).to_index != -1:
                        from_particle = PyList_GET_ITEM(particles, (<SpecificPairForceApplier>force).from_index)
                        to_particle = PyList_GET_ITEM(particles, (<SpecificPairForceApplier>force).to_index)
                        distance_squared = 0
                        for k in range(self.dimensions):
                            d = (<Particle>to_particle).position.numbers[k] - (<Particle>from_particle).position.numbers[k]
                            direction.numbers[k] = d
                            distance_squared += d * d
                        distance = sqrt(distance_squared)
                        for k in range(self.dimensions):
                            direction.numbers[k] /= distance
                        (<SpecificPairForceApplier>force).apply(<Particle>from_particle, <Particle>to_particle, direction, distance, distance_squared)
                m = PyList_GET_SIZE(matrix_forces)
                if m:
                    for i in range(1, n):
                        from_particle = PyList_GET_ITEM(particles, i)
                        for j in range(i):
                            to_particle = PyList_GET_ITEM(particles, j)
                            distance_squared = 0
                            for k in range(self.dimensions):
                                d = (<Particle>to_particle).position.numbers[k] - (<Particle>from_particle).position.numbers[k]
                                direction.numbers[k] = d
                                distance_squared += d * d
                            distance = -1
                            for k in range(m):
                                force = PyList_GET_ITEM(matrix_forces, k)
                                if not (<MatrixPairForceApplier>force).max_distance_squared or \
                                        distance_squared < (<MatrixPairForceApplier>force).max_distance_squared:
                                    if distance == -1:
                                        distance = sqrt(distance_squared)
                                        for ii in range(self.dimensions):
                                            direction.numbers[ii] /= distance
                                    (<MatrixPairForceApplier>force).apply(<Particle>from_particle, <Particle>to_particle,
                                                                          direction, distance, distance_squared)
                for i in range(n):
                    (<Particle>PyList_GET_ITEM(particles, i)).update(speed_of_light, clock, delta)
                m = PyList_GET_SIZE(barriers)
                for i in range(m):
                    barrier = PyList_GET_ITEM(barriers, i)
                    for j in range(o):
                        to_particle = PyList_GET_ITEM(non_anchors, j)
                        (<Barrier>barrier).apply(<Particle>to_particle, delta)
                last_time += delta
                clock += delta
                if last_time >= time or (realtime and not extra_frame):
                    break
                extra_frame = False
        return clock


RENDERER_CLASS = PhysicsSystem
