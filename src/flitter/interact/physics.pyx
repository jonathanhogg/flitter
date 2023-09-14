# cython: language_level=3, profile=False, boundscheck=False, wraparound=False, cdivision=True

"""
Flitter physics engine
"""

from loguru import logger

from .. import name_patch
from ..model cimport Vector, Node, null_
from ..language.functions cimport Normal
from ..language.vm cimport StateDict

from libc.math cimport sqrt, isinf, isnan, abs


logger = name_patch(logger, __name__)

cdef Vector VELOCITY = Vector('velocity')

cdef Normal RandomSource = Normal('_physics')
cdef unsigned long long RandomIndex = 0


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
        self.force = Vector._copy(node.get_fvec('force', zero.length, zero))
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
        self.velocity = Vector._copy(zero)

    cdef void update(self, double speed_of_light, double delta, StateDict state):
        state.set_item(self.position_state_key, self.position)
        state.set_item(self.velocity_state_key, self.velocity)


cdef class ForceApplier:
    cdef double strength

    def __cinit__(self, Node node, Vector zero):
        self.strength = node.get_float('strength', 1)


cdef class PairForceApplier(ForceApplier):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        raise NotImplementedError()


cdef class ParticleForceApplier(ForceApplier):
    cdef void apply(self, Particle particle, double delta):
        raise NotImplementedError()


cdef class MatrixPairForceApplier(PairForceApplier):
    cdef double max_distance

    def __cinit__(self, Node node, Vector zero):
        self.max_distance = max(0, node.get_float('max_distance', 0))


cdef class SpecificPairForceApplier(PairForceApplier):
    cdef Vector from_particle_id
    cdef Vector to_particle_id

    def __cinit__(self, Node node, Vector zero):
        self.from_particle_id = <Vector>node._attributes.get('from')
        self.to_particle_id = <Vector>node._attributes.get('to')


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
    cdef void apply(self, Particle particle, double delta):
        cdef double speed_squared=0, v, k
        cdef int i
        if particle.radius:
            for i in range(particle.velocity.length):
                v = particle.velocity.numbers[i]
                speed_squared += v * v
            k = min(self.strength * sqrt(speed_squared) * (particle.radius * particle.radius), particle.mass / delta)
            for i in range(particle.velocity.length):
                particle.force.numbers[i] = particle.force.numbers[i] - particle.velocity.numbers[i] * k


cdef class ConstantForceApplier(ParticleForceApplier):
    cdef Vector force

    def __cinit__(self, Node node, Vector zero):
        cdef Vector force
        cdef int i
        force = node.get_fvec('force', zero.length, null_)
        if force.length == 0:
            force = node.get_fvec('direction', zero.length, zero).normalize()
            for i in range(force.length):
                force.numbers[i] = force.numbers[i] * self.strength
        self.force = force

    cdef void apply(self, Particle particle, double delta):
        cdef int i
        for i in range(self.force.length):
            particle.force.numbers[i] = particle.force.numbers[i] + self.force.numbers[i]


cdef class RandomForceApplier(ParticleForceApplier):
    cdef void apply(self, Particle particle, double delta):
        global RandomIndex
        cdef int i
        for i in range(particle.force.length):
            particle.force.numbers[i] = particle.force.numbers[i] + self.strength * RandomSource._item(RandomIndex)
            RandomIndex += 1


cdef class CollisionForceApplier(MatrixPairForceApplier):
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


cdef class GravityForceApplier(MatrixPairForceApplier):
    cdef void apply(self, Particle from_particle, Particle to_particle, Vector direction, double distance, double distance_squared):
        cdef double f, k
        cdef int i
        if (not self.max_distance or distance < self.max_distance) and from_particle.mass and to_particle.mass:
            k = self.strength * from_particle.mass * to_particle.mass / distance_squared
            for i in range(direction.length):
                f = direction.numbers[i] * k
                from_particle.force.numbers[i] = from_particle.force.numbers[i] + f
                to_particle.force.numbers[i] = to_particle.force.numbers[i] - f


cdef class ElectrostaticForceApplier(MatrixPairForceApplier):
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
    def destroy(self):
        pass

    def purge(self):
        pass

    async def update(self, engine, Node node, double now):
        self._update(engine.state, node, now, engine.target_fps, engine.realtime)

    cdef _update(self, StateDict state, Node node, double now, double target_fps, bint realtime):
        cdef int dimensions = node.get_int('dimensions', 0)
        if dimensions < 1:
            return
        cdef Vector state_prefix = <Vector>node._attributes.get('state')
        if state_prefix is None:
            return
        cdef double time = node.get_float('time', now)
        cdef double resolution = node.get_float('resolution', 1/target_fps)
        if resolution <= 0:
            return
        cdef double speed_of_light = node.get_float('speed_of_light', 1e9)
        if speed_of_light <= 0:
            return
        cdef int i, j, k, n
        cdef Vector zero = Vector.__new__(Vector)
        zero.allocate_numbers(dimensions)
        for i in range(dimensions):
            zero.numbers[i] = 0
        cdef list particles=[], particle_forces=[], matrix_forces=[], specific_forces=[]
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
                specific_forces.append(DistanceForceApplier.__new__(DistanceForceApplier, child, zero))
            elif child.kind == 'drag':
                particle_forces.append(DragForceApplier.__new__(DragForceApplier, child, zero))
            elif child.kind == 'constant':
                particle_forces.append(ConstantForceApplier.__new__(ConstantForceApplier, child, zero))
            elif child.kind == 'random':
                particle_forces.append(RandomForceApplier.__new__(RandomForceApplier, child, zero))
            elif child.kind == 'collision':
                matrix_forces.append(CollisionForceApplier.__new__(CollisionForceApplier, child, zero))
            elif child.kind == 'gravity':
                matrix_forces.append(GravityForceApplier.__new__(GravityForceApplier, child, zero))
            elif child.kind == 'electrostatic':
                matrix_forces.append(ElectrostaticForceApplier.__new__(ElectrostaticForceApplier, child, zero))
            child = child.next_sibling
        cdef double last_time
        cdef Vector time_vector = state.get_item(state_prefix)
        if time_vector.length == 1 and time_vector.numbers != NULL:
            last_time = min(time, time_vector.numbers[0])
        else:
            logger.debug("Create physics {!r} with {} particles and {} forces", state_prefix, len(particles),
                         len(particle_forces) + len(matrix_forces) + len(specific_forces))
            last_time = time
        cdef double delta
        cdef Particle from_particle, to_particle
        cdef Vector direction = Vector.__new__(Vector)
        direction.allocate_numbers(dimensions)
        cdef double distance, distance_squared
        cdef dict particles_by_id
        cdef double d
        cdef MatrixPairForceApplier matrix_force
        while True:
            delta = min(resolution, time-last_time)
            for force in particle_forces:
                for particle in particles:
                    (<ParticleForceApplier>force).apply(<Particle>particle, delta)
            n = len(matrix_forces)
            if n:
                for i in range(1, len(particles)):
                    from_particle = <Particle>particles[i]
                    for j in range(i):
                        to_particle = <Particle>particles[j]
                        distance_squared = 0
                        for k in range(dimensions):
                            d = to_particle.position.numbers[k] - from_particle.position.numbers[k]
                            direction.numbers[k] = d
                            distance_squared += d * d
                        distance = sqrt(distance_squared)
                        for k in range(dimensions):
                            direction.numbers[k] /= distance
                        for k in range(n):
                            matrix_force = <MatrixPairForceApplier>matrix_forces[k]
                            if not matrix_force.max_distance or distance < matrix_force.max_distance:
                                matrix_force.apply(from_particle, to_particle, direction, distance, distance_squared)
            if specific_forces:
                particles_by_id = {}
                for particle in particles:
                    particles_by_id[(<Particle>particle).id] = particle
                for force in specific_forces:
                    from_particle = <Particle>particles_by_id.get((<SpecificPairForceApplier>force).from_particle_id)
                    to_particle = <Particle>particles_by_id.get((<SpecificPairForceApplier>force).to_particle_id)
                    if from_particle is not None and to_particle is not None:
                        distance_squared = 0
                        for k in range(dimensions):
                            d = to_particle.position.numbers[k] - from_particle.position.numbers[k]
                            direction.numbers[k] = d
                            distance_squared += d * d
                        distance = sqrt(distance_squared)
                        for k in range(dimensions):
                            direction.numbers[k] /= distance
                        (<SpecificPairForceApplier>force).apply(from_particle, to_particle, direction, distance, distance_squared)
            for particle in particles:
                (<Particle>particle).update(speed_of_light, delta, state)
            last_time += delta
            if realtime or last_time >= time:
                break
        time_vector = Vector.__new__(Vector)
        time_vector.allocate_numbers(1)
        time_vector.numbers[0] = time
        state.set_item(state_prefix, time_vector)


INTERACTOR_CLASS = PhysicsSystem
