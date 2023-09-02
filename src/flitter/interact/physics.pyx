# cython: language_level=3, profile=False, boundscheck=False, wraparound=False, cdivision=True

"""
Flitter physics engine
"""

from loguru import logger

from .. import name_patch
from ..model cimport Vector, Node
from ..language.vm cimport StateDict

from libc.math cimport sqrt, isinf, isnan


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
            self.position = position
        else:
            self.position = node.get_fvec('initial', zero.length, zero)
        self.velocity_state_key = Vector._compose([self.position_state_key, VELOCITY], 0, 2)
        cdef Vector velocity = state.get_item(self.velocity_state_key)
        if velocity.length == zero.length and velocity.numbers != NULL:
            self.velocity = velocity
        else:
            self.velocity = zero
        self.force = zero
        self.radius = max(0, node.get_float('radius', 0))
        self.mass = max(0, node.get_float('mass', 1))
        self.charge = node.get_float('charge', 0)

    cdef void update(self, double speed_of_light, double delta, StateDict state):
        cdef int i
        if self.mass:
            for i in range(self.force.length):
                if isinf(self.force.numbers[i]) or isnan(self.force.numbers[i]):
                    break
            else:
                self.velocity = self.velocity.add(self.force.fmul(delta / self.mass))
        cdef double speed = sqrt(self.velocity.squared_sum())
        if speed > speed_of_light:
            self.velocity = self.velocity.fmul(speed_of_light / speed)
        self.position = self.position.add(self.velocity.fmul(delta))
        state.set_item(self.position_state_key, self.position)
        state.set_item(self.velocity_state_key, self.velocity)


cdef class Anchor(Particle):
    def __cinit__(self, Node node, Vector id, Vector zero, Vector prefix, StateDict state):
        self.position = node.get_fvec('position', zero.length, zero)
        self.velocity = zero
        self.mass = max(0, node.get_float('mass', 0))

    cdef void update(self, double speed_of_light, double delta, StateDict state):
        state.set_item(self.position_state_key, self.position)
        state.set_item(self.velocity_state_key, self.velocity)


cdef class Constraint:
    cdef double coefficient

    def __cinit__(self, Node node, Vector zero):
        self.coefficient = node.get_float('coefficient', 1)


cdef class ParticleConstraint(Constraint):
    cdef void apply(self, Particle particle):
        raise NotImplementedError()


cdef class PairConstraint(Constraint):
    cdef double max_distance

    def __cinit__(self, Node node, Vector zero):
        self.max_distance = max(0, node.get_float('max_distance', 0))

    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        raise NotImplementedError()


cdef class SpecificPairConstraint(PairConstraint):
    cdef Vector from_particle_id
    cdef Vector to_particle_id

    def __cinit__(self, Node node, Vector zero):
        self.from_particle_id = node._attributes.get('from')
        self.to_particle_id = node._attributes.get('to')


cdef class DistanceConstraint(SpecificPairConstraint):
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
        cdef Vector force
        if self.minimum and distance < self.minimum:
            force = direction.fmul(self.coefficient * (self.minimum - distance))
            from_particle.force = from_particle.force.sub(force)
            to_particle.force = to_particle.force.add(force)
        elif self.maximum and distance > self.maximum:
            force = direction.fmul(self.coefficient * (distance - self.maximum))
            from_particle.force = from_particle.force.add(force)
            to_particle.force = to_particle.force.sub(force)


cdef class DragConstraint(ParticleConstraint):
    cdef void apply(self, Particle particle):
        cdef Vector force = particle.velocity.abs().fadd(1).mul(particle.velocity).fmul(self.coefficient)
        particle.force = particle.force.sub(force)


cdef class ConstantConstraint(ParticleConstraint):
    cdef Vector force

    def __cinit__(self, Node node, Vector zero):
        self.force = node.get_fvec('direction', zero.length, zero).normalize().fmul(self.coefficient)

    cdef void apply(self, Particle particle):
        if self.force.as_bool():
            particle.force = particle.force.add(self.force)


cdef class CollisionConstraint(PairConstraint):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        cdef double min_distance
        if from_particle.radius and to_particle.radius:
            min_distance = from_particle.radius + to_particle.radius
            if distance < min_distance:
                force = direction.fmul(self.coefficient * (min_distance - distance))
                from_particle.force = from_particle.force.sub(force)
                to_particle.force = to_particle.force.add(force)


cdef class GravityConstraint(PairConstraint):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        if (not self.max_distance or distance < self.max_distance) and from_particle.mass and to_particle.mass:
            force = direction.fmul(self.coefficient * from_particle.mass * to_particle.mass / distance_squared)
            from_particle.force = from_particle.force.add(force)
            to_particle.force = to_particle.force.sub(force)


cdef class ElectrostaticConstraint(PairConstraint):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        if from_particle.charge and to_particle.charge:
            force = direction.fmul(self.coefficient * -from_particle.charge * to_particle.charge / distance_squared)
            from_particle.force = from_particle.force.add(force)
            to_particle.force = to_particle.force.sub(force)


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
        cdef Vector state_prefix = node._attributes.get('state')
        if state_prefix is None or state_prefix.length == 0:
            return
        cdef double resolution = node.get_float('resolution', 1/engine.target_fps)
        if resolution <= 0:
            return
        cdef double speed_of_light = node.get_float('speed_of_light', 1e9)
        if speed_of_light <= 0:
            return
        cdef int i, j
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
                id = child._attributes.get('id')
                if id is not None and id.length != 0:
                    particle = Particle.__new__(Particle, child, id, zero, state_prefix, state)
                    particles.append(particle)
            elif child.kind == 'anchor':
                id = child._attributes.get('id')
                if id is not None and id.length != 0:
                    particle = Anchor.__new__(Anchor, child, id, zero, state_prefix, state)
                    particles.append(particle)
            elif child.kind == 'distance':
                specific_pair_constraints.append(DistanceConstraint.__new__(DistanceConstraint, child, zero))
            elif child.kind == 'drag':
                particle_constraints.append(DragConstraint.__new__(DragConstraint, child, zero))
            elif child.kind == 'constant':
                particle_constraints.append(ConstantConstraint.__new__(ConstantConstraint, child, zero))
            elif child.kind == 'collision':
                pair_constraints.append(CollisionConstraint.__new__(CollisionConstraint, child, zero))
            elif child.kind == 'gravity':
                pair_constraints.append(GravityConstraint.__new__(GravityConstraint, child, zero))
            elif child.kind == 'electrostatic':
                pair_constraints.append(ElectrostaticConstraint.__new__(ElectrostaticConstraint, child, zero))
            child = child.next_sibling
        cdef double beat = engine.counter.beat_at_time(now)
        cdef double last_beat = self.last_beat if self.last_beat else beat
        cdef double delta = min(resolution, beat-last_beat)
        cdef Particle particle1, particle2
        cdef Vector direction
        cdef double distance, distance_squared
        cdef PairConstraint pair_constraint
        if particle_constraints or pair_constraints:
            for i in range(len(particles)):
                particle1 = <Particle>particles[i]
                for constraint in particle_constraints:
                    (<ParticleConstraint>constraint).apply(particle1)
                if pair_constraints:
                    for j in range(i):
                        particle2 = <Particle>particles[j]
                        direction = particle2.position.sub(particle1.position)
                        distance_squared = direction.squared_sum()
                        distance = sqrt(distance_squared)
                        direction = direction.ftruediv(distance)
                        for constraint in pair_constraints:
                            pair_constraint = <PairConstraint>constraint
                            if not pair_constraint.max_distance or distance < pair_constraint.max_distance:
                                pair_constraint.apply(particle1, particle2, direction, distance, distance_squared)
        cdef dict particles_by_id
        # cdef double planck_length = 1 / (speed_of_light * speed_of_light)
        if specific_pair_constraints:
            particles_by_id = {}
            for particle1 in particles:
                particles_by_id[particle1.id] = particle1
            for constraint in specific_pair_constraints:
                particle1 = particles_by_id.get((<SpecificPairConstraint>constraint).from_particle_id)
                particle2 = particles_by_id.get((<SpecificPairConstraint>constraint).to_particle_id)
                pair_constraint = <PairConstraint>constraint
                if particle1 is not None and particle2 is not None:
                    direction = particle2.position.sub(particle1.position)
                    distance_squared = direction.squared_sum()
                    distance = sqrt(distance_squared)
                    direction = direction.ftruediv(distance)
                    if not pair_constraint.max_distance or distance < pair_constraint.max_distance:
                        pair_constraint.apply(particle1, particle2, direction, distance, distance_squared)
        for particle1 in particles:
            particle1.update(speed_of_light, delta, state)
        self.last_beat = beat


INTERACTOR_CLASS = PhysicsSystem
