# cython: boundscheck=False, wraparound=False, cdivision=True, cpow=True

"""
Flitter physics engine
"""

import asyncio
from loguru import logger

from .. import name_patch
from ..model cimport Vector, Node, StateDict, null_
from ..language.functions cimport normal

import cython
from libc.math cimport sqrt, isinf, isnan, abs, floor
from libc.stdint cimport uint64_t, int64_t
from cpython cimport PyObject
from cpython.dict cimport PyDict_GetItem

cdef extern from "Python.h":
    # Note: I am explicitly (re)defining these as not requiring the GIL for speed.
    # This appears to work completely fine, but is obviously making assumptions
    # on the Python ABI that may not be correct.
    #
    Py_ssize_t PyList_GET_SIZE(object list) nogil
    PyObject* PyList_GET_ITEM(object list, Py_ssize_t i) nogil

logger = name_patch(logger, __name__)

cdef Vector VELOCITY = Vector._symbol('velocity')
cdef Vector LAST = Vector._symbol('last')
cdef Vector CLOCK = Vector._symbol('clock')
cdef Vector RUN = Vector._symbol('run')
cdef Vector ITERATION = Vector._symbol('iteration')


cdef class PhysicsGroup:
    cdef PhysicsGroup parent
    cdef list particles
    cdef list non_anchors
    cdef list particle_forces
    cdef list matrix_forces
    cdef list specific_forces
    cdef list barriers

    def __cinit__(self):
        self.particles = []
        self.non_anchors = []
        self.particle_forces = []
        self.matrix_forces = []
        self.specific_forces = []
        self.barriers = []


cdef class Particle:
    cdef Vector id
    cdef Vector position_state_key
    cdef Vector position
    cdef Vector velocity_state_key
    cdef Vector velocity
    cdef Vector acceleration
    cdef Vector initial_force
    cdef Vector force
    cdef double radius
    cdef double mass
    cdef double charge
    cdef double ease

    @cython.profile(False)
    def __cinit__(self, Node node, Vector id, Vector zero, Vector prefix, StateDict state):
        self.id = id
        self.position_state_key = prefix.concat(self.id).intern()
        cdef Vector position = state.get_item(self.position_state_key)
        if position.length == zero.length and position.numbers != NULL:
            self.position = position.copy()
        else:
            self.position = node.get_fvec('position', zero.length, zero).copy()
        self.velocity_state_key = self.position_state_key.concat(VELOCITY).intern()
        cdef Vector velocity = state.get_item(self.velocity_state_key)
        if velocity.length == zero.length and velocity.numbers != NULL:
            self.velocity = velocity.copy()
        else:
            self.velocity = node.get_fvec('velocity', zero.length, zero).copy()
        self.initial_force = node.get_fvec('force', zero.length, zero)
        self.force = self.initial_force.copy()
        self.acceleration = zero.copy()
        self.radius = max(0, node.get_float('radius', 1))
        self.mass = max(0, node.get_float('mass', 1))
        self.charge = node.get_float('charge', 1)
        self.ease = node.get_float('ease', 0)

    @cython.profile(False)
    cdef void step(self, double speed_of_light, double clock, double delta) noexcept nogil:
        cdef double speed_squared, v, k
        cdef int64_t i, n=self.force.length
        for i in range(n):
            self.acceleration.numbers[i] = self.velocity.numbers[i]
        if self.mass:
            for i in range(n):
                if isinf(self.force.numbers[i]) or isnan(self.force.numbers[i]):
                    break
            else:
                k = delta / self.mass
                if self.ease > 0 and clock < self.ease:
                    k *= clock / self.ease
                speed_squared = 0
                for i in range(n):
                    v = self.velocity.numbers[i] + self.force.numbers[i] * k
                    self.velocity.numbers[i] = v
                    speed_squared += v * v
                if speed_of_light > 0 and speed_squared > speed_of_light * speed_of_light:
                    k = speed_of_light / sqrt(speed_squared)
                    for i in range(n):
                        self.velocity.numbers[i] = self.velocity.numbers[i] * k
        for i in range(n):
            v = self.acceleration.numbers[i]
            self.acceleration.numbers[i] = (self.velocity.numbers[i] - v) / delta
            v = (v + self.velocity.numbers[i]) / 2
            self.position.numbers[i] = self.position.numbers[i] + v * delta

    @cython.profile(False)
    cdef void reset_force(self) noexcept nogil:
        cdef int64_t i, n=self.force.length
        for i in range(n):
            self.force.numbers[i] = self.initial_force.numbers[i]


cdef class Anchor(Particle):
    @cython.profile(False)
    def __cinit__(self, Node node, Vector id, Vector zero, Vector prefix, StateDict state):
        self.position = node.get_fvec('position', zero.length, zero)
        self.velocity = zero.copy()

    @cython.profile(False)
    cdef void step(self, double speed_of_light, double clock, double delta) noexcept nogil:
        pass


cdef class Barrier:
    cdef Vector position
    cdef Vector normal
    cdef double restitution

    @cython.profile(False)
    def __cinit__(self, Node node, Vector zero):
        self.position = node.get_fvec('position', zero.length, zero)
        self.normal = node.get_fvec('normal', zero.length, zero).normalize()
        self.restitution = min(max(0, node.get_float('restitution', 1)), 1)

    @cython.profile(False)
    cdef void apply(self, Particle particle, double speed_of_light, double clock, double delta) noexcept nogil:
        if self.normal.length == 0:
            return
        cdef int i, dimensions=self.position.length
        cdef double n, velocity=0, distance=-particle.radius, acceleration=0
        for i in range(dimensions):
            n = self.normal.numbers[i]
            distance = distance + (particle.position.numbers[i] - self.position.numbers[i]) * n
            velocity = velocity + particle.velocity.numbers[i] * n
            acceleration = acceleration + particle.acceleration.numbers[i] * n
        if distance >= 0:
            return
        cdef double vv=velocity*velocity, ad2=2*acceleration*distance, k=distance/velocity
        cdef double rewind_t = (velocity + sqrt(vv - ad2)) / acceleration if acceleration and ad2 < vv else k
        if velocity >= 0 or rewind_t > delta:
            for i in range(dimensions):
                particle.position.numbers[i] = particle.position.numbers[i] - distance * self.normal.numbers[i]
            return
        cdef double v
        velocity = 0
        for i in range(dimensions):
            particle.position.numbers[i] = particle.position.numbers[i] - k * particle.velocity.numbers[i]
            v = (particle.velocity.numbers[i] - rewind_t * particle.acceleration.numbers[i]) * self.restitution
            particle.velocity.numbers[i] = v
            velocity = velocity + v * self.normal.numbers[i]
        for i in range(dimensions):
            n = self.normal.numbers[i]
            particle.velocity.numbers[i] = particle.velocity.numbers[i] - 2 * velocity * n
        velocity = -velocity
        cdef double t
        if acceleration < 0 and (t := 2 * velocity / -acceleration) < t:
            rewind_t = t
        particle.step(speed_of_light, clock, rewind_t)


cdef class ForceApplier:
    cdef double strength

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.strength = strength


cdef class PairForceApplier(ForceApplier):
    @cython.profile(False)
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        raise NotImplementedError()


cdef class ParticleForceApplier(ForceApplier):
    @cython.profile(False)
    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        raise NotImplementedError()


cdef class MatrixPairForceApplier(PairForceApplier):
    cdef double max_distance_squared

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.max_distance_squared = max(0, node.get_float('max_distance', 0)) ** 2


cdef class SpecificPairForceApplier(PairForceApplier):
    cdef Vector from_particle_id
    cdef Vector to_particle_id

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.from_particle_id = <Vector>node._attributes.get('from') if node._attributes else None
        self.to_particle_id = <Vector>node._attributes.get('to') if node._attributes else None


cdef class DistanceForceApplier(SpecificPairForceApplier):
    cdef double power
    cdef double minimum
    cdef double maximum

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.power = max(0, node.get_float('power', 1))
        cdef double fixed
        if (fixed := node.get_float('fixed', 0)) != 0:
            self.minimum = fixed
            self.maximum = fixed
        else:
            self.minimum = node.get_float('min', 0)
            self.maximum = node.get_float('max', 0)

    @cython.profile(False)
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
    @cython.profile(False)
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

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.density = node.get_float('density', 1)
        self.gravity = node.get_fvec('gravity', zero.length, zero)
        if self.gravity is zero:
            self.gravity = self.gravity.copy()
            self.gravity.numbers[zero.length-1] = -1

    @cython.profile(False)
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

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.force = node.get_fvec('force', zero.length, zero)
        self.acceleration = node.get_fvec('acceleration', zero.length, zero)

    @cython.profile(False)
    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        cdef int64_t i
        cdef double f
        for i in range(self.force.length):
            f = self.force.numbers[i] + self.acceleration.numbers[i] * particle.mass
            particle.force.numbers[i] = particle.force.numbers[i] + f*self.strength


cdef class RandomForceApplier(ParticleForceApplier):
    cdef normal random_source
    cdef uint64_t index

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.index = 0

    @cython.profile(False)
    cdef void apply(self, Particle particle, double delta) noexcept nogil:
        cdef int64_t i
        for i in range(particle.force.length):
            particle.force.numbers[i] = particle.force.numbers[i] + self.strength * self.random_source._item(self.index)
            self.index = self.index + 1


cdef class CollisionForceApplier(MatrixPairForceApplier):
    cdef double power

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.power = max(0, node.get_float('power', 1))

    @cython.profile(False)
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
    @cython.profile(False)
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        cdef double f, k, d=from_particle.radius + to_particle.radius
        cdef int64_t i
        if from_particle.mass and to_particle.mass:
            k = self.strength * from_particle.mass * to_particle.mass / max(distance_squared, d*d)
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class ElectrostaticForceApplier(MatrixPairForceApplier):
    @cython.profile(False)
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared) noexcept nogil:
        cdef double f, k, d=from_particle.radius + to_particle.radius
        cdef int64_t i
        if from_particle.charge and to_particle.charge:
            k = self.strength * from_particle.charge * to_particle.charge / max(distance_squared, d*d)
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] - f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] + f


cdef class AdhesionForceApplier(MatrixPairForceApplier):
    cdef double overlap

    @cython.profile(False)
    def __cinit__(self, Node node, double strength, Vector zero):
        self.overlap = min(max(0, node.get_float('overlap', 0.25)), 1)

    @cython.profile(False)
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
    cdef double resolution
    cdef double speed_of_light

    def __init__(self, **kwargs):
        self.state_keys = set()

    def destroy(self):
        pass

    def purge(self):
        pass

    async def update(self, engine, Node node, double time, double performance, **kwargs):
        cdef int64_t run = node.get_int('run', 0)
        self.dimensions = node.get_int('dimensions', 0)
        if self.dimensions < 1:
            return
        cdef Vector state_prefix = <Vector>node._attributes.get('state') if node._attributes else None
        if state_prefix is None:
            return
        time = node.get_float('time', time)
        self.resolution = node.get_float('resolution', 1/engine.target_fps)
        if self.resolution <= 0:
            return
        self.speed_of_light = max(0, node.get_float('speed_of_light', 0))
        cdef StateDict state = engine.state
        cdef Vector time_vector = state.get_item(state_prefix.concat(CLOCK))
        if time_vector.length == 1 and time_vector.numbers != NULL:
            clock = time_vector.numbers[0]
        else:
            clock = 0
        cdef double last_time
        time_vector = state.get_item(state_prefix.concat(LAST))
        if time_vector.length == 1 and time_vector.numbers != NULL:
            last_time = time_vector.numbers[0]
        else:
            last_time = time
        cdef Vector last_run_vector = state.get_item(state_prefix.concat(RUN))
        cdef int64_t last_run
        if last_run_vector.length == 1 and last_run_vector.numbers != NULL:
            last_run = <int64_t>floor(last_run_vector.numbers[0])
        else:
            last_run = run
        cdef Vector iteration_vector = state.get_item(state_prefix.concat(ITERATION))
        cdef int64_t iteration
        if iteration_vector.length == 1 and iteration_vector.numbers != NULL:
            iteration = <int64_t>floor(iteration_vector.numbers[0]) + 1
        else:
            iteration = 0
        cdef Vector state_key
        if run != last_run:
            logger.debug("Reset physics system {!r} for run {}", state_prefix, run)
            for state_key in self.state_keys:
                state.set_item(state_key, null_)
            self.state_keys = set()
            time_vector = Vector.__new__(Vector)
            time_vector.allocate_numbers(1)
            time_vector.numbers[0] = time
            state.set_item(state_prefix, time_vector)
            clock = 0
            iteration = 0
        cdef Vector run_vector = Vector.__new__(Vector)
        run_vector.allocate_numbers(1)
        run_vector.numbers[0] = run
        state.set_item(state_prefix.concat(RUN), run_vector)
        iteration_vector = Vector.__new__(Vector)
        iteration_vector.allocate_numbers(1)
        iteration_vector.numbers[0] = iteration
        state.set_item(state_prefix.concat(ITERATION), iteration_vector)
        cdef Vector seed = state_prefix.concat(run_vector).concat(iteration_vector)
        cdef PhysicsGroup group, top_group=PhysicsGroup.__new__(PhysicsGroup)
        cdef list groups = [top_group]
        cdef dict particles_by_id = {}
        cdef set old_state_keys = self.state_keys
        cdef set new_state_keys = set()
        self.collect(groups, top_group, node, particles_by_id, old_state_keys, new_state_keys, state, state_prefix, clock, seed)
        for state_key in old_state_keys:
            state.set_item(state_key, null_)
        self.state_keys = new_state_keys
        time_vector = state.get_item(state_prefix)
        cdef int64_t i
        if time_vector.length != 1 or time_vector.numbers == NULL:
            if len(groups) == 1:
                logger.debug("New {}D physics {!r} with {} particles, {} forces and {} barriers", self.dimensions, state_prefix, len(top_group.particles),
                             len(top_group.particle_forces) + len(top_group.matrix_forces) + len(top_group.specific_forces), len(top_group.barriers))
            else:
                logger.debug("New {}D physics {!r} with {} particles over {} groups", self.dimensions, state_prefix, len(particles_by_id), len(groups))
                for i, group in enumerate(groups):
                    logger.trace("Group {} with {} particles, {} forces and {} barriers", i, len(group.particles),
                                 len(group.particle_forces) + len(group.matrix_forces) + len(group.specific_forces), len(group.barriers))
            time_vector = Vector.__new__(Vector)
            time_vector.allocate_numbers(1)
            time_vector.numbers[0] = time
            state.set_item(state_prefix, time_vector)
        cdef bint extra = performance > 1.5
        cdef Particle particle
        if time > last_time:
            clock = await asyncio.to_thread(self.calculate, groups, particles_by_id, engine.realtime, extra, time, last_time, clock)
            for group in groups:
                for particle in group.particles:
                    if particle.position.is_finite():
                        state.set_item(particle.position_state_key, particle.position)
                    if particle.velocity.is_finite():
                        state.set_item(particle.velocity_state_key, particle.velocity)
        time_vector = Vector.__new__(Vector)
        time_vector.allocate_numbers(1)
        time_vector.numbers[0] = time
        state.set_item(state_prefix.concat(LAST), time_vector)
        time_vector = Vector.__new__(Vector)
        time_vector.allocate_numbers(1)
        time_vector.numbers[0] = clock
        state.set_item(state_prefix.concat(CLOCK), time_vector)

    cdef void collect(self, list groups, PhysicsGroup group, Node node, dict particles_by_id, set old_state_keys, set new_state_keys,
                      StateDict state, Vector state_prefix, double clock, Vector seed):
        cdef Node child
        cdef Vector id
        cdef double strength, ease
        cdef Particle particle
        cdef ParticleForceApplier particle_force
        cdef Barrier barrier
        cdef PhysicsGroup child_group
        cdef Vector zero = Vector.__new__(Vector)
        zero.allocate_numbers(self.dimensions)
        cdef int64_t i
        for i in range(self.dimensions):
            zero.numbers[i] = 0
        for child in node._children:
            if child.kind is 'particle':
                id = <Vector>child._attributes.get('id')
                if id is not None:
                    particle = Particle.__new__(Particle, child, id, zero, state_prefix, state)
                    particles_by_id[id] = particle
                    group.particles.append(particle)
                    group.non_anchors.append(particle)
                    new_state_keys.add(particle.position_state_key)
                    old_state_keys.discard(particle.position_state_key)
                    new_state_keys.add(particle.velocity_state_key)
                    old_state_keys.discard(particle.velocity_state_key)
            elif child.kind is 'anchor':
                id = <Vector>child._attributes.get('id')
                if id is not None:
                    particle = Anchor.__new__(Anchor, child, id, zero, state_prefix, state)
                    particles_by_id[id] = particle
                    group.particles.append(particle)
                    new_state_keys.add(particle.position_state_key)
                    old_state_keys.discard(particle.position_state_key)
                    new_state_keys.add(particle.velocity_state_key)
                    old_state_keys.discard(particle.velocity_state_key)
            elif child.kind is 'group':
                child_group = PhysicsGroup.__new__(PhysicsGroup)
                child_group.parent = group
                groups.append(child_group)
                self.collect(groups, child_group, child, particles_by_id, old_state_keys, new_state_keys, state, state_prefix, clock, seed)
            elif child.kind is 'barrier':
                barrier = Barrier.__new__(Barrier, child, zero)
                group.barriers.append(barrier)
            else:
                strength = child.get_float('strength', 1)
                ease = child.get_float('ease', 0)
                if ease > 0 and ease < clock:
                    strength *= clock/ease
                if child.kind is 'distance':
                    group.specific_forces.append(DistanceForceApplier.__new__(DistanceForceApplier, child, strength, zero))
                elif child.kind is 'drag':
                    group.particle_forces.append(DragForceApplier.__new__(DragForceApplier, child, strength, zero))
                elif child.kind is 'buoyancy':
                    group.particle_forces.append(BuoyancyForceApplier.__new__(BuoyancyForceApplier, child, strength, zero))
                elif child.kind is 'constant':
                    group.particle_forces.append(ConstantForceApplier.__new__(ConstantForceApplier, child, strength, zero))
                elif child.kind is 'random':
                    particle_force = RandomForceApplier.__new__(RandomForceApplier, child, strength, zero)
                    (<RandomForceApplier>particle_force).random_source = normal(seed)
                    group.particle_forces.append(particle_force)
                elif child.kind is 'collision':
                    group.matrix_forces.append(CollisionForceApplier.__new__(CollisionForceApplier, child, strength, zero))
                elif child.kind is 'gravity':
                    group.matrix_forces.append(GravityForceApplier.__new__(GravityForceApplier, child, strength, zero))
                elif child.kind is 'electrostatic':
                    group.matrix_forces.append(ElectrostaticForceApplier.__new__(ElectrostaticForceApplier, child, strength, zero))
                elif child.kind is 'adhesion':
                    group.matrix_forces.append(AdhesionForceApplier.__new__(AdhesionForceApplier, child, strength, zero))

    cpdef double calculate(self, list groups, dict particles_by_id, bint realtime, bint extra, double time, double last_time, double clock):
        cdef list all_particles, particles, non_anchors, particle_forces, matrix_forces, barriers
        cdef int64_t i, j, k, m, n, o, p
        cdef double delta
        cdef Vector direction = Vector.__new__(Vector)
        direction.allocate_numbers(self.dimensions)
        cdef double d, distance, distance_squared
        cdef PyObject* from_particle
        cdef PyObject* to_particle
        cdef PyObject* force
        cdef PyObject* barrier
        cdef PhysicsGroup group
        cdef SpecificPairForceApplier specific_force
        while True:
            delta = min(self.resolution, time-last_time)
            for group in groups:
                for specific_force in group.specific_forces:
                    from_particle = PyDict_GetItem(particles_by_id, specific_force.from_particle_id)
                    if from_particle == NULL:
                        continue
                    to_particle = PyDict_GetItem(particles_by_id, specific_force.to_particle_id)
                    if to_particle == NULL:
                        continue
                    distance_squared = 0
                    for k in range(self.dimensions):
                        d = (<Particle>to_particle).position.numbers[k] - (<Particle>from_particle).position.numbers[k]
                        direction.numbers[k] = d
                        distance_squared += d * d
                    distance = sqrt(distance_squared)
                    for k in range(self.dimensions):
                        direction.numbers[k] /= distance
                    specific_force.apply(<Particle>from_particle, <Particle>to_particle, direction, distance, distance_squared)
                particles = group.particles
                non_anchors = group.non_anchors
                if group.parent is None:
                    all_particles = group.particles
                    particle_forces = group.particle_forces
                    matrix_forces = group.matrix_forces
                    barriers = group.barriers
                else:
                    all_particles = group.particles.copy()
                    particle_forces = group.particle_forces.copy()
                    matrix_forces = group.matrix_forces.copy()
                    barriers = group.barriers.copy()
                    while (group := group.parent) is not None:
                        all_particles.extend(group.particles)
                        particle_forces.extend(group.particle_forces)
                        matrix_forces.extend(group.matrix_forces)
                        barriers.extend(group.barriers)
                with nogil:
                    n = PyList_GET_SIZE(all_particles)
                    o = PyList_GET_SIZE(non_anchors)
                    p = PyList_GET_SIZE(particles)
                    m = PyList_GET_SIZE(particle_forces)
                    if m and o:
                        for i in range(m):
                            for j in range(o):
                                (<ParticleForceApplier>PyList_GET_ITEM(particle_forces, i)).apply((<Particle>PyList_GET_ITEM(non_anchors, j)), delta)
                    m = PyList_GET_SIZE(matrix_forces)
                    if m and p:
                        for i in range(p):
                            from_particle = PyList_GET_ITEM(all_particles, i)
                            for j in range(i+1, n):
                                to_particle = PyList_GET_ITEM(all_particles, j)
                                distance_squared = 0
                                for k in range(self.dimensions):
                                    d = (<Particle>to_particle).position.numbers[k] - (<Particle>from_particle).position.numbers[k]
                                    direction.numbers[k] = d
                                    distance_squared += d * d
                                distance = sqrt(distance_squared)
                                for k in range(self.dimensions):
                                    direction.numbers[k] /= distance
                                for k in range(m):
                                    force = PyList_GET_ITEM(matrix_forces, k)
                                    if not (<MatrixPairForceApplier>force).max_distance_squared or \
                                            distance_squared < (<MatrixPairForceApplier>force).max_distance_squared:
                                        (<MatrixPairForceApplier>force).apply(<Particle>from_particle, <Particle>to_particle,
                                                                              direction, distance, distance_squared)
                    if o:
                        for i in range(o):
                            (<Particle>PyList_GET_ITEM(non_anchors, i)).step(self.speed_of_light, clock, delta)
                        m = PyList_GET_SIZE(barriers)
                        if m:
                            for i in range(m):
                                barrier = PyList_GET_ITEM(barriers, i)
                                for j in range(o):
                                    to_particle = PyList_GET_ITEM(non_anchors, j)
                                    (<Barrier>barrier).apply(<Particle>to_particle, self.speed_of_light, clock, delta)
            last_time += delta
            clock += delta
            if last_time >= time or (realtime and not extra):
                break
            extra = False
            for group in groups:
                non_anchors = group.non_anchors
                with nogil:
                    for j in range(o):
                        to_particle = PyList_GET_ITEM(non_anchors, j)
                        (<Particle>to_particle).reset_force()
        return clock
