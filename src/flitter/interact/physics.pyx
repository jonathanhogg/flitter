# cython: language_level=3, profile=False, boundscheck=False, wraparound=False, cdivision=True

"""
Flitter physics engine
"""

from loguru import logger

from .. import name_patch
from ..model cimport Vector, Node
from ..language.vm cimport StateDict

from libc.math cimport sqrt, isinf, isnan, abs


logger = name_patch(logger, __name__)

cdef Vector VELOCITY = Vector('velocity')


cdef class Particle:
    cdef Vector id
    cdef Vector position_state_key
    cdef Vector position
    cdef Vector velocity_state_key
    cdef Vector velocity
    cdef Vector force
    cdef double radius
    cdef double mass
    cdef double charge

    def __cinit__(self, Node node, Vector id, Vector zero, Vector prefix, StateDict state):
        self.id = id
        self.position_state_key = Vector._compose([prefix, self.id], 0, 2)
        cdef Vector position = state.get_item(self.position_state_key)
        if position.length == zero.length and position.numbers != NULL:
            self.position = Vector._copy(position)
        else:
            self.position = Vector._copy(node.get_fvec('position', zero.length, zero))
        self.velocity_state_key = Vector._compose([self.position_state_key, VELOCITY], 0, 2)
        cdef Vector velocity = state.get_item(self.velocity_state_key)
        if velocity.length == zero.length and velocity.numbers != NULL:
            self.velocity = Vector._copy(velocity)
        else:
            self.velocity = Vector._copy(node.get_fvec('velocity', zero.length, zero))
        self.force = Vector._copy(zero)
        self.radius = max(0, node.get_float('radius', 1))
        self.mass = max(0, node.get_float('mass', 1))
        self.charge = node.get_float('charge', 1)

    cdef void update(self, double speed_of_light, double delta, StateDict state):
        cdef double speed, d, k
        cdef int i, n=self.force.length
        if self.mass:
            for i in range(n):
                if isinf(self.force.numbers[i]) or isnan(self.force.numbers[i]):
                    break
            else:
                k = delta / self.mass
                speed = 0
                for i in range(n):
                    d = self.velocity.numbers[i] + self.force.numbers[i] * k
                    self.velocity.numbers[i] = d
                    speed += d * d
                if speed > speed_of_light * speed_of_light:
                    k = speed_of_light / sqrt(speed)
                    for i in range(n):
                        self.velocity.numbers[i] = self.velocity.numbers[i] * k
        for i in range(n):
            self.position.numbers[i] = self.position.numbers[i] + self.velocity.numbers[i] * delta
        state.set_item(self.position_state_key, self.position)
        state.set_item(self.velocity_state_key, self.velocity)


cdef class Anchor(Particle):
    def __cinit__(self, Node node, Vector id, Vector zero, Vector prefix, StateDict state):
        self.position = node.get_fvec('position', zero.length, zero)
        self.velocity = node.get_fvec('velocity', zero.length, zero)

    cdef void update(self, double speed_of_light, double delta, StateDict state):
        state.set_item(self.position_state_key, self.position)
        state.set_item(self.velocity_state_key, self.velocity)


cdef class ForceApplier:
    cdef double strength

    def __cinit__(self, Node node, Vector zero):
        self.strength = node.get_float('strength', 1)


cdef class ParticleForceApplier(ForceApplier):
    cdef void apply(self, Particle particle):
        raise NotImplementedError()


cdef class PairForceApplier(ForceApplier):
    cdef double max_distance

    def __cinit__(self, Node node, Vector zero):
        self.max_distance = max(0, node.get_float('max_distance', 0))

    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        raise NotImplementedError()


cdef class SpecificPairForceApplier(PairForceApplier):
    cdef Vector from_particle_id
    cdef Vector to_particle_id

    def __cinit__(self, Node node, Vector zero):
        self.from_particle_id = node._attributes.get('from')
        self.to_particle_id = node._attributes.get('to')


cdef class DistanceForceApplier(SpecificPairForceApplier):
    cdef double minimum
    cdef double maximum

    def __cinit__(self, Node node, Vector zero):
        cdef double fixed
        if (fixed := node.get_float('fixed', 0)) != 0:
            self.minimum = fixed
            self.maximum = fixed
        else:
            self.minimum = node.get_float('min', 0)
            self.maximum = node.get_float('max', 0)

    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        cdef double f, k
        cdef int i
        if self.minimum and distance < self.minimum:
            k = self.strength * (self.minimum - distance)
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] - f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] + f
        elif self.maximum and distance > self.maximum:
            k = self.strength * (distance - self.maximum)
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class DragForceApplier(ParticleForceApplier):
    cdef void apply(self, Particle particle):
        cdef double speed_squared=0, v, k
        cdef int i
        if particle.radius:
            for i in range(particle.velocity.length):
                v = particle.velocity.numbers[i]
                speed_squared += v * v
            k = self.strength * sqrt(speed_squared) * (particle.radius * particle.radius)
            for i in range(particle.velocity.length):
                particle.force.numbers[i] = particle.force.numbers[i] - particle.velocity.numbers[i] * k


cdef class ConstantForceApplier(ParticleForceApplier):
    cdef Vector force

    def __cinit__(self, Node node, Vector zero):
        cdef Vector force = node.get_fvec('direction', zero.length, zero).normalize().fmul(self.strength)
        self.force = force if force.as_bool() else None

    cdef void apply(self, Particle particle):
        cdef int i
        if self.force is not None:
            for i in range(self.force.length):
                particle.force.numbers[i] = particle.force.numbers[i] + self.force.numbers[i]


cdef class CollisionForceApplier(PairForceApplier):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        cdef double min_distance, f, k
        cdef int i
        if from_particle.radius and to_particle.radius:
            min_distance = from_particle.radius + to_particle.radius
            if distance < min_distance:
                k = self.strength * (min_distance - distance)
                for i in range(direction.length):
                    f = direction.numbers[i] * k
                    from_particle.force.numbers[i] = from_particle.force.numbers[i] - f
                    to_particle.force.numbers[i] = to_particle.force.numbers[i] + f


cdef class GravityForceApplier(PairForceApplier):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        cdef double f, k
        cdef int i
        if (not self.max_distance or distance < self.max_distance) and from_particle.mass and to_particle.mass:
            k = self.strength * from_particle.mass * to_particle.mass / distance_squared
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class ElectrostaticForceApplier(PairForceApplier):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        cdef double f, k
        cdef int i
        if from_particle.charge and to_particle.charge:
            k = self.strength * -from_particle.charge * to_particle.charge / distance_squared
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class PhysicsSystem:
    cdef double last_beat

    def __init__(self):
        logger.debug("Create physics system")
        self.last_beat = 0

    def destroy(self):
        pass

    def purge(self):
        pass

    async def update(self, engine, Node node, now):
        cdef int dimensions = node.get_int('dimensions', 0)
        if dimensions < 1:
            return
        cdef Vector state_prefix = <Vector>node._attributes.get('state')
        if state_prefix is None:
            return
        cdef double resolution = node.get_float('resolution', 1/(<double>engine.target_fps))
        if resolution <= 0:
            return
        cdef double speed_of_light = node.get_float('speed_of_light', 1e9)
        if speed_of_light <= 0:
            return
        cdef int i, j, k
        cdef StateDict state = engine.state
        cdef Vector zero = Vector.__new__(Vector)
        zero.allocate_numbers(dimensions)
        for i in range(dimensions):
            zero.numbers[i] = 0
        cdef list particles=[], particle_constraints=[], pair_constraints=[], specific_pair_constraints=[]
        cdef Node child = node.first_child
        cdef Vector id
        while child is not None:
            if child.kind == 'particle':
                id = <Vector>child._attributes.get('id')
                if id is not None:
                    particles.append(Particle.__new__(Particle, child, id, zero, state_prefix, state))
            elif child.kind == 'anchor':
                id = <Vector>child._attributes.get('id')
                if id is not None:
                    particles.append(Anchor.__new__(Anchor, child, id, zero, state_prefix, state))
            elif child.kind == 'distance':
                specific_pair_constraints.append(DistanceForceApplier.__new__(DistanceForceApplier, child, zero))
            elif child.kind == 'drag':
                particle_constraints.append(DragForceApplier.__new__(DragForceApplier, child, zero))
            elif child.kind == 'constant':
                particle_constraints.append(ConstantForceApplier.__new__(ConstantForceApplier, child, zero))
            elif child.kind == 'collision':
                pair_constraints.append(CollisionForceApplier.__new__(CollisionForceApplier, child, zero))
            elif child.kind == 'gravity':
                pair_constraints.append(GravityForceApplier.__new__(GravityForceApplier, child, zero))
            elif child.kind == 'electrostatic':
                pair_constraints.append(ElectrostaticForceApplier.__new__(ElectrostaticForceApplier, child, zero))
            child = child.next_sibling
        cdef bint realtime = engine.realtime
        cdef double beat = engine.counter.beat_at_time(now)
        if not self.last_beat:
            self.last_beat = beat
        cdef double delta
        cdef Particle particle1, particle2
        cdef Vector direction = Vector.__new__(Vector)
        direction.allocate_numbers(dimensions)
        cdef double distance, distance_squared
        cdef dict particles_by_id
        cdef double d
        while True:
            if particle_constraints or pair_constraints:
                for i in range(len(particles)):
                    particle1 = <Particle>particles[i]
                    for constraint in particle_constraints:
                        (<ParticleForceApplier>constraint).apply(particle1)
                    if pair_constraints:
                        for j in range(i):
                            particle2 = <Particle>particles[j]
                            distance_squared = 0
                            for k in range(dimensions):
                                d = particle2.position.numbers[k] - particle1.position.numbers[k]
                                direction.numbers[k] = d
                                distance_squared += d * d
                            distance = sqrt(distance_squared)
                            for k in range(dimensions):
                                direction.numbers[k] /= distance
                            for constraint in pair_constraints:
                                if not (<PairForceApplier>constraint).max_distance or distance < (<PairForceApplier>constraint).max_distance:
                                    (<PairForceApplier>constraint).apply(particle1, particle2, direction, distance, distance_squared)
            if specific_pair_constraints:
                particles_by_id = {}
                for particle in particles:
                    particles_by_id[(<Particle>particle).id] = particle
                for constraint in specific_pair_constraints:
                    particle1 = <Particle>particles_by_id.get((<SpecificPairForceApplier>constraint).from_particle_id)
                    particle2 = <Particle>particles_by_id.get((<SpecificPairForceApplier>constraint).to_particle_id)
                    if particle1 is not None and particle2 is not None:
                        distance_squared = 0
                        for k in range(dimensions):
                            d = particle2.position.numbers[k] - particle1.position.numbers[k]
                            direction.numbers[k] = d
                            distance_squared += d * d
                        distance = sqrt(distance_squared)
                        if not (<PairForceApplier>constraint).max_distance or distance < (<PairForceApplier>constraint).max_distance:
                            for k in range(dimensions):
                                direction.numbers[k] /= distance
                            (<PairForceApplier>constraint).apply(particle1, particle2, direction, distance, distance_squared)
            delta = min(resolution, beat-self.last_beat) if realtime else resolution
            for particle in particles:
                (<Particle>particle).update(speed_of_light, delta, state)
            self.last_beat += delta
            if realtime or self.last_beat >= beat:
                break
        if realtime:
            self.last_beat = beat


INTERACTOR_CLASS = PhysicsSystem
